import k2
import torch
from espnet.lm.lm_utils import make_lexical_tree
from espnet.nets.scorer_interface import PartialScorerInterface
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm


class MMIFramePrefixScorer(PartialScorerInterface):
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list):
        
        self.lang = lang
        # Although the main decoding process adopt CPU. MMI computation may
        # Adopt GPU to accelerate
        #self.device = torch.device(f"cuda:{rank-1}") if device == "cuda" \
        #              else torch.device("cpu")
        self.device = torch.device("cpu")
        print("MMI scorer device: ", self.device)
        
        self.lexicon = Lexicon(lang)
        self.graph_compiler = MmiTrainingGraphCompiler(self.lexicon, self.device)
        self.phone_ids = self.lexicon.phone_symbols()
      
        self.lo = torch.nn.Linear(idim, len(self.phone_ids) + 1) 
        self.lm_scores = None
        self.load_weight(rank)

        self.P = create_bigram_phone_lm(self.phone_ids)
        self.P.set_scores_stochastic_(self.lm_scores)
        
        self.blank = 0 # by default 0 means CTC blank 
        self.logzero = -10000

        # special tokens and lexical root
        self.word2id = self.lexicon.words._sym2id 
        self.char2id = {c: cid for cid, c in enumerate(char_list)}
        self.id2char = {cid: c for cid, c in enumerate(char_list)}
        self.eos = sos_id
        self.word_unk_id = self.word2id["<UNK>"]
        self.char_space_id = self.char2id["<space>"]
        self.lexroot = make_lexical_tree(self.word2id, self.char2id, "<UNK>")
        print(f"lexroot succ: {self.lexroot[0].keys()}")

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
        x = x.unsqueeze(0)
        nnet_output = self.lo(x)

        # (2) den_scores
        texts = ["<UNK>"] # use a random text, just to get den graph
        _, den = self.graph_compiler.compile(texts, self.P, replicate_den=True)

        T = x.size()[1]
        den_scores = []
        # use a loop since denominator would consume much memory
        # in descending order
        for t in range(T, 0, -1):
            supervision = torch.Tensor([[0, 0, t]]).to(torch.int32) # [idx, start, length]
            dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
            den_lats = k2.intersect_dense(den, dense_fsa_vec, output_beam=10.0)
            den_tot_scores = den_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
            den_scores.append(den_tot_scores)
        den_scores = torch.cat(den_scores).unsqueeze(0).cpu() # [T] -> [B, T]
        print("Den scores: ", den_scores)

        # (3) Prev Score is zero
        prev_score = torch.Tensor([0]).to(torch.float32)
        return nnet_output, den_scores, prev_score, self.lexroot

    def select_state(self, states, j):
        nnet_output_single, den_scores, prev_scores, next_roots = states
        return nnet_output_single, den_scores, prev_scores[j], next_roots[j]

    def score(**kargs):
        raise NotImplementedError

    def split_intervals(self, olds, max_interval=2000):
        # it may cause error in k2 if the interval is too large. e.g., > 5k
        news, indices = [], []
        cnt = 0
        for start, end in olds:
            group_idx = []
            while start <= end:
                news.append([start, min(end, start + max_interval)])
                start += max_interval
                group_idx.append(cnt)
                cnt += 1
            indices.append(group_idx)
        return news, indices

    def score_partial(self, y, next_tokens, state, hs_pad):
        # (1) unpack the state
        nnet_output_single, den_scores, prev_score, root = state
    
        # (2) process the prefix and intervals; build numerator graphs       
        prefix = "".join([self.id2char[c.item()] for c in y]).replace("<eos>", "").replace("<space>", " ")
        prefix = [self.word2id.get(tok, self.word_unk_id) for tok in prefix.strip().split()]
        if y[-1] == self.char_space_id:
            prefix = prefix # end with <space>
        else:
            prefix = prefix[:-1] # the last part is a sub-word
        print(f"prefix: {prefix}")

        intervals = []
        force_zero = [] # invalid case. Set prob to logzero finally
        next_roots = []
        for idx, tok in enumerate(next_tokens):
            tok = tok.item()
            # case 1: <space>, <eos> indicate the end of a word
            # This reduce the probability from a prefix prob
            # to an exact probability of a word
            if tok == self.char_space_id or tok == self.eos:
                if root[1] > -1:
                    intervals.append([root[1]-1, root[1]])
                    next_roots.append(self.lexroot)
                    print(f"{idx}-th: {tok}, space / eos. This is a valid space")
                else:
                    intervals.append((self.word_unk_id-1, self.word_unk_id))
                    force_zero.append(idx)
                    next_roots.append(self.lexroot)
                    print(f"{idx}-th: {tok}, space / eos. This is an invalid space")
            # case 2: OOV. kill it
            elif not tok in root[0]:
                intervals.append((self.word_unk_id-1, self.word_unk_id))
                force_zero.append(idx)
                next_roots.append(self.lexroot)
                print(f"{idx}-th: {tok}, oov")
            # case 3: A valid intra-word transition. 
            # shift to next lexicon node
            else:
                intervals.append(root[0][tok][2])
                next_roots.append(root[0][tok])
                print(f"{idx}-th: {tok}, intra-trans")
       
        # Being compatible with lexroot format 
        intervals = [(l+1, r+1) for l, r in intervals]
        # Long intervals may cause error in k2. split them into
        split_intervals, interval_indices = self.split_intervals(intervals)
        print(f"intervals: {intervals}, split intervals: {split_intervals}", flush=True)
        num_split = len(split_intervals)

        num_graphs = self.graph_compiler.compile_nums_for_prefix_scoring(
                                          prefix, split_intervals, self.P)
        
        # (3) frame-wise intersection
        # calculate frame-by-frame to avoid OOM
        T = nnet_output_single.size()[1]
        nnet_output = nnet_output_single.repeat(num_split, 1, 1)
        score_segment = []
   
        for t in range(1, T+1):
            supervision = torch.stack([
                          torch.arange(num_split),
                          torch.zeros(num_split),
                          torch.ones(num_split) * t
                          ], dim=1).to(torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(nnet_output[:, :t], supervision)
            num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=10.0) 
            num_tot_scores_split = num_lats.get_tot_scores(log_semiring=True,
                                                           use_double_scores=True)
            score_segment.append(num_tot_scores_split)
        
        score_segment = torch.stack(score_segment, dim=1) # [num_split, T]
        num_scores = torch.zeros(len(interval_indices), T) # [batch_size, T]
        for i in range(len(interval_indices)):
            num_scores[i] = torch.logsumexp(score_segment[interval_indices[i]], dim=0)
            if i in force_zero:
                num_scores[i] = self.logzero

        # (4) finalize
        tot_scores = num_scores - den_scores.unsqueeze(0)
        tot_scores = torch.logsumexp(tot_scores, dim=-1)[0]
        tok_scores = tot_scores - prev_score
        state = (nnet_output_single, den_scores, tot_scores, next_roots)
        return tok_scores.numpy(), state

    def final_score(self, state):
        return 0     
                
