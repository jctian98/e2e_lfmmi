import k2
import torch
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.lm.lm_utils import make_lexical_tree
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm


class MMILookaheadScorer(PartialScorerInterface):
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list):
        
        self.lang = lang
        self.device = device
        
        self.lexicon = Lexicon(lang)
        self.graph_compiler = MmiTrainingGraphCompiler(self.lexicon, self.device)
        self.phone_ids = self.lexicon.phone_symbols()
      
        self.lo = torch.nn.Linear(idim, len(self.phone_ids) + 1) 
        self.lm_scores = None
        self.load_weight(rank)

        self.P = create_bigram_phone_lm(self.phone_ids)
        self.P.set_scores_stochastic_(self.lm_scores)
       
        # We use a char-level lexicon root but search in it by BPE
        alphabet = [chr(i) for i in range(65, 91)] + ["\'"] # upper class
        self.alphabet_dict = {c: i+1 for i, c in enumerate(alphabet)} 
        self.word_dict = self.lexicon.words._sym2id
        self.word_unk_id = int(open(self.lang / 'oov.int').read().strip()) 
        self.lexroot = make_lexical_tree(self.word_dict, self.alphabet_dict, self.word_unk_id) # 3 is unknown-id
        print("end of lex building", flush=True)

        self.char_list = char_list # BPE char list
        self.bpe_space = char_list[-2][0]

        # special value
        self.logzero = -10000.0
        self.eos = sos_id

    def load_weight(self, rank):
        # load lo weight and lm_scores
        ckpt_dict = torch.load(self.lang / f"mmi_param.{rank}.pth")
        for v in ckpt_dict.values():
            v.requires_grad=False
        self.lm_scores = ckpt_dict["lm_scores"]
        lo_dict = {"weight": ckpt_dict["lo.1.weight"],
                   "bias": ckpt_dict["lo.1.bias"]}
        self.lo.load_state_dict(lo_dict)

    def init_state(self, x):
        # (1) nnet_output
        nnet_output = self.lo(x.unsqueeze(0))

        # (2) den_graph
        texts = ["<UNK>"]
        _, den = self.graph_compiler.compile(texts, self.P, replicate_den=True)

        # (3) den_scores. Use a loop to avoid memory spark
        T = nnet_output.size()[1]
        den_scores = []
        for t in range(T, 0, -1):
            supervision = torch.Tensor([[0, 0, t]]).to(torch.int32) # [idx, start, length]
            dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
            den_lats = k2.intersect_dense(den, dense_fsa_vec, output_beam=10.0)
            den_tot_scores = den_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
            den_scores.append(den_tot_scores)
        den_scores = torch.cat(den_scores).unsqueeze(0) # [T] -> [B, T]

        # (4) others
        prev_score = torch.Tensor([0]).to(torch.float32)

        return nnet_output, den_scores, prev_score, self.lexroot

    def select_state(self, states, j):
        # only the prev_scores and next_nodes should be selected
        nnet_output_single, den_scores, prev_scores, next_nodes = states
        return nnet_output_single, den_scores, prev_scores[j], next_nodes[j]

    def score(**kargs):
        raise NotImplementedError

    def score_partial(self, y, next_tokens, state, hs_pad):
        print("start a new partial score", flush=True)
        torch.set_printoptions(sci_mode=False)
        # (1) unpack state
        nnet_output_single, den_frame_scores, prev_score, node = state
        beam_size = len(next_tokens)
        T = nnet_output_single.size()[1]   

        # (2) build numerator graph
        
        prefix = "".join([self.char_list[x.item()] for x in y[1:]])\
                 .replace(" ", "").replace(self.bpe_space, " ").strip().split(" ")

        intervals, next_nodes, index_to_kill = self.search_lexical_tree(node, next_tokens)
        # if proposed token does not start with '_', y[-1] should be removed during graph composition
        # '_' means the proposal of new word. <eos> cannot be seen as a new word. other special tokens
        # would be killed
        drop_prefix_tail = [int(not self.char_list[x.item()].startswith(self.bpe_space)) for x in next_tokens]
        # some intervals are too long. we need to split them before computation and then combine 
        split_intervals, interval_indexes, split_drop_prefix_tail = self.split_intervals(intervals, drop_prefix_tail)
        print("intervals: \n", intervals, "split intervals: \n", split_intervals, "interval_index: \n", interval_indexes)
        graphs = self.graph_compiler.compile_nums_for_prefix_scoring(
                     prefix, split_intervals, self.P, split_drop_prefix_tail)

        # (3) loop to compute frame-level prob. cannot do this in one go due to memory limitation
        split_size = len(split_intervals)
        split_frame_scores = []
        nnet_output = nnet_output_single.expand(split_size, -1, -1)
        for t in range(T, 0, -1):
            supervision = torch.stack([
                          torch.arange(split_size),
                          torch.zeros(split_size),
                          torch.ones(split_size) * t,
                          ], dim=-1).cpu().int()
            dense_fsa_vec = k2.DenseFsaVec(nnet_output[:, :t, :], supervision)
            # split_lats = k2.intersect_dense(graphs, dense_fsa_vec, output_beam=3.0)
            # must use a pruned version! very slow here especially for large interval
            split_lats = k2.intersect_dense_pruned(graphs,
                                         dense_fsa_vec,
                                         search_beam=10.0,
                                         output_beam=5.0,
                                         min_active_states=30,
                                         max_active_states=10000)
            split_frame_score = split_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
            split_frame_scores.append(split_frame_score)
            print(f"intersection t = {t}", flush=True)
        split_frame_scores = torch.stack(split_frame_scores, dim=-1) # [split_size, T]
        
        # (4) combine split and then den_scores to compute scores
        num_frame_scores = []
        for interval_index in interval_indexes:
            num_frame_score = torch.logsumexp(split_frame_scores[interval_index], dim=0)
            num_frame_scores.append(num_frame_score)
        num_frame_scores = torch.stack(num_frame_scores, dim=0) # [beam_size, T]

        frame_scores = num_frame_scores - den_frame_scores
        scores = torch.logsumexp(frame_scores, dim=-1)

        # (5) postprocess
        # A. <eos> should only aligned with last frame
        eos_pos = torch.where(next_tokens == self.eos)[0]
        if len(eos_pos) > 0:
            print("found eos position: ", eos_pos)
            scores[eos_pos] = frame_scores[eos_pos.item(), 0]

        # B. kill hypothesis in "index_to_kill"
        print("index to kill: ", index_to_kill)
        if len(index_to_kill) > 0:
            for idx in index_to_kill:
                scores[idx] = self.logzero
        
        # (6) return
        token_scores = scores - prev_score
        print("token score: ", token_scores, flush=True) 
        states = nnet_output_single, den_frame_scores, scores, next_nodes
        return token_scores, states

   
    def search_lexical_tree(self, node, next_tokens):
        if node is None:
            print("None node given!")

        intervals, next_nodes = [], []
        # some tokens are invalid (e.g., invalid word combination). however, we should
        # still compute the score for it to be compatible. We will force that score
        # to logzero in postprocess stage, but need to use the index_to_kill make a record.
        index_to_kill = [] 

        for idx, i in enumerate(next_tokens):
            # node is the previous one if _ is not proposed else root
            subword = self.char_list[i]
            print("subword: ", subword)
            # case (1): '_' or <eos> is proposed, which means end of the word
            if subword == self.bpe_space or subword == "<eos>":
                this_node = node # keep 'node' unchanged
                # Invalid and kill. Previous node cannot be root
                if this_node == self.lexroot:
                    interval = [self.word_unk_id-1, self.word_unk_id]
                    this_node = None
                    index_to_kill.append(idx)
                # score is for a word, not a word prefix -> interval for only one word 
                else:
                    interval = [this_node[2][0], this_node[2][0] + 1]
                    # next_node is root so the next token is valid even though it is not
                    # start with '_' 
                    this_node = self.lexroot

            # case (2): impossible token. kill them
            elif subword == "<blank>" or subword == "<unk>":
                this_node = None
                interval = [self.word_unk_id-1, self.word_unk_id]
                index_to_kill.append(idx)

            # case (3): ordinary tokens. All special token should never reach this branch
            else:
                # subword start with '_' means a prefix of new word -> search from root
                this_node = self.lexroot if subword.startswith(self.bpe_space) else node

                subword = subword.replace(self.bpe_space, "")
                for c in subword:
                    cid = self.alphabet_dict[c]
                    # descent to successor
                    if cid in this_node[0]:
                        this_node = this_node[0][cid]
                    # no valid successor found. kill this hypothesis
                    else:
                        this_node = None
                        break
            
                if this_node is not None and this_node[2] is not None:
                    interval = this_node[2]
                else:
                    interval = [self.word_unk_id-1, self.word_unk_id]
                    index_to_kill.append(idx)
            
            # plus one to correct the interval. see building process of lexroot
            interval = [interval[0] + 1, interval[1] + 1]  
            intervals.append(interval)
            # this_node == None always means a kill
            next_nodes.append(this_node)

        return intervals, next_nodes, index_to_kill

    def split_intervals(self, intervals, is_new_word, max_len=20000):
        """
        large interval (length > 2000) will cause errors in k2. 
        Split it and keep the index
        """
        new_intervals = []
        interval_indexes = []
        interval_is_new_word = []
        for idx, (left, right) in enumerate(intervals):
            interval_index = []
            new_word = is_new_word[idx]
            while left < right:
                new_intervals.append([left, min(left + max_len, right)])   
                left += max_len       
                interval_index.append(idx) 
                interval_is_new_word.append(new_word) # means this split is a new_word
            interval_indexes.append(interval_index)

        return new_intervals, interval_indexes, interval_is_new_word

    def final_score(self, state):
        return 0
             
