import k2
import torch
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.lm.lm_utils import make_lexical_tree
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm
from espnet.nets.scorers.lookahead import parse_lookahead, build_word_fsa_mat

class MMILookaheadScorer(PartialScorerInterface):
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list):
        
        self.lang = lang
        self.device = device
        
        self.lexicon = Lexicon(lang)
        self.oov = self.oovid = open(self.lang / 'oov.txt').read().strip()
        self.graph_compiler = MmiTrainingGraphCompiler(self.lexicon, self.device, self.oov)
        self.phone_ids = self.lexicon.phone_symbols()
      
        self.lo = torch.nn.Linear(idim, len(self.phone_ids) + 1) 
        self.lm_scores = None
        self.load_weight(rank)

        self.P = create_bigram_phone_lm(self.phone_ids)
        self.P.set_scores_stochastic_(self.lm_scores)
       
        # We use a char-level lexicon root but search in it by BPE
        alphabet = [chr(i) for i in range(65, 91)] + ["\'"] if "A" in char_list else\
                   [chr(i) for i in range(97, 123)] + ["\'"]
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

        return nnet_output, den_scores, prev_score

    def select_state(self, states, j):
        # only the prev_scores and next_nodes should be selected
        nnet_output_single, den_scores, prev_scores = states
        return nnet_output_single, den_scores, prev_scores[j]

    def score(**kargs):
        raise NotImplementedError

    def score_partial(self, y, next_tokens, state, hs_pad):
        # (1) unpack the state
        nnet_output_single, den_frame_scores, prev_score = state
        beam_size = len(next_tokens)        

        # (2) build numerator graph
        yseqs = torch.cat([
                y.unsqueeze(0).repeat(beam_size, 1),
                next_tokens.unsqueeze(1)], dim=1).cpu().tolist()
        prefix_and_interval = [parse_lookahead(yseq, self.lexroot,
                               self.char_list, self.alphabet_dict,
                               self.word_dict, self.bpe_space)
                               for yseq in yseqs]
        word_fsa_mats = [build_word_fsa_mat(*x) for x in prefix_and_interval]
        word_fsa = [k2.Fsa.from_dict({"arcs": mat}) for mat in word_fsa_mats]
        word_fsa = k2.create_fsa_vec(word_fsa)
        num_graphs = self.graph_compiler.compile_lookahead_numerators(word_fsa, self.P) 

        # (3) loop to compute frame-level prob. cannot do this in one go due to memory limitation
        T = nnet_output_single.size()[1]
        num_frame_scores = []
        nnet_output = nnet_output_single.expand(beam_size, -1, -1)
        for t in range(T, 0, -1):
            supervision = torch.stack([
                          torch.arange(beam_size),
                          torch.zeros(beam_size),
                          torch.ones(beam_size) * t,
                          ], dim=-1).cpu().int()
            dense_fsa_vec = k2.DenseFsaVec(nnet_output[:, :t, :], supervision)
            
            # A pruned version. Or it would be much slow. parameters are tunable
            lats = k2.intersect_dense_pruned(num_graphs,
                                         dense_fsa_vec,
                                         search_beam=10.0,
                                         output_beam=5.0,
                                         min_active_states=30,
                                         max_active_states=10000)
            num_frame_score = lats.get_tot_scores(log_semiring=True, use_double_scores=True)
            num_frame_scores.append(num_frame_score)
        num_frame_scores = torch.stack(num_frame_scores, dim=-1) # [beam, T]
        # Important: exclude all -inf
        num_frame_scores = torch.clamp(num_frame_scores, min=self.logzero)
 
        frame_scores = num_frame_scores - den_frame_scores
        scores = torch.logsumexp(frame_scores, dim=-1)

        # (4) postprocess
        # (4.1) <eos> should only aligned with last frame
        eos_pos = torch.where(next_tokens == self.eos)[0]
        if len(eos_pos) > 0:
            scores[eos_pos] = frame_scores[eos_pos.item(), 0]

        # TODO: kill some valid token like <blank>
        
        # (5) return
        token_scores = scores - prev_score
        states = nnet_output_single, den_frame_scores, scores
        return token_scores, states

    """  
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
    """
    def final_score(self, state):
        return 0
     
