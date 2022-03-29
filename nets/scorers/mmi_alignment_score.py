import k2
import torch
import math
from espnet.nets.scorer_interface import PartialScorerInterface
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm
from espnet.nets.scorers.mmi_utils import step_intersect

class MMIRNNTScorer():
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list, mas_lookahead=1):
        
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
        
        self.char_list = char_list
        self.eos = sos_id  # <sos> is identical to <eos>
        self.blank = 0 # by default 0 means CTC blank 
        self.logzero = -10000
     
        self.mas_lookahead = mas_lookahead

    def load_weight(self, rank):
        # load lo weight and lm_scores
        ckpt_dict = torch.load(self.lang / f"mmi_param.{rank}.pth")
        for v in ckpt_dict.values():
            v.requires_grad=False
        self.lm_scores = ckpt_dict["lm_scores"]
        lo_dict = {"weight": ckpt_dict["lo.1.weight"],
                   "bias": ckpt_dict["lo.1.bias"]}
        self.lo.load_state_dict(lo_dict)

    def den_scores(self, x):
        # (1) nnet_output
        x = x.unsqueeze(0)
        nnet_output = self.lo(x)

        # (2) den_scores
        texts = ["<UNK>"] # use a random text, just to get den graph
        _, den = self.graph_compiler.compile(texts, self.P, replicate_den=True)

        T = nnet_output.size(1)
        supervision = torch.Tensor([[0, 0, T]]).int()
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        den_scores = step_intersect(den, dense_fsa_vec)[0]

        return nnet_output, den_scores

    def batch_rescore(self, A, nnet_output, den_scores):
        batch, T = len(A), nnet_output.size(1)
        if batch == 0:
            return A

        texts = [h.yseq[1:] for h in A]
        texts = [" ".join([self.char_list[x] for x in text]) for text in texts]
        num, _ = self.graph_compiler.compile(texts, self.P, replicate_den=False)

        supervision = torch.stack([torch.arange(batch),
                                   torch.zeros(batch),
                                   torch.ones(batch) * T], dim=1).int()
        dense_fsa_vec = k2.DenseFsaVec(nnet_output.repeat(batch, 1, 1), supervision)

        num_lats = k2.intersect_dense_pruned(num,
                                             dense_fsa_vec,
                                             search_beam=20.0,
                                             output_beam=10.0,
                                             min_active_states=30,
                                             max_active_states=100000)
        num_scores = num_lats.get_tot_scores(True, True)
        tot_scores = num_scores - den_scores[-1] # num_frame -> index

        for h, s in zip(A, tot_scores):
            h.mmi_prev_score = s.item()
        
        return A

    def batch_score(self, A, nnet_output, den_scores, tu_sum, mmi_weight):
        if self.mas_lookahead > 15:
            return self.Imp_A(A, nnet_output, den_scores, tu_sum, mmi_weight)
        else:
            return self.Imp_B(A, nnet_output, den_scores, tu_sum, mmi_weight)

    def Imp_B(self, A, nnet_output, den_scores, tu_sum, mmi_weight):
        batch = len(A)
        if batch == 0:
            return A

        # reorder: increasing order in u means decreasing order in t
        #          this is required by k2 supervision
        A.sort(key=lambda h: len(h.yseq))

        # (1) get ts: the alignment length in t-axis 
        # +1 since frame start with 0; +1 since redundant blank
        ts = [tu_sum - len(h.yseq) + 2 for h in A]
        ts = torch.Tensor(ts).long()

        # (2) compile numerator graph
        texts = [h.yseq[1:] for h in A] # exclude starting <sos>
        texts = [" ".join([self.char_list[x] for x in text]) for text in texts] # need modification for BPE
        num_graphs, _ = self.graph_compiler.compile(texts, self.P, replicate_den=False)

        # (3) intersection
        lookahead_range = (0, self.mas_lookahead)
        tot_scores_collection = []
        T = nnet_output.size()[1]
        for s in range(lookahead_range[0], lookahead_range[1] + 1):
            ts_shift = torch.clamp(ts + s, min=1, max=T)
            supervision = torch.stack([torch.arange(batch),
                                       torch.zeros(batch),
                                       ts_shift
                                       ], dim=1).to(torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(nnet_output.repeat(batch, 1, 1),
                                           supervision)
            num_lats = k2.intersect_dense_pruned(num_graphs,
                                             dense_fsa_vec,
                                             search_beam=20.0,
                                             output_beam=10.0,
                                             min_active_states=30,
                                             max_active_states=20000)
            num_scores = num_lats.get_tot_scores(True, True)
            # num_scores = torch.where(num_scores == -math.inf, 0.0, num_scores)
            tot_scores = num_scores - den_scores[ts_shift-1] # num_frame -> idx_frame
            tot_scores_collection.append(tot_scores)
        tot_scores = torch.stack(tot_scores_collection, dim=1) # [beam, T]

        # hint: we can only use top-1 score rather than logsumexp or top-k-sum
        # since torch.clamp leads to repeatition of these scores at boundaries
        tot_scores, _ = torch.topk(tot_scores, 1, dim=-1)

        # (4) assign and post-process
        idx_to_empty_str = [j for j, h in enumerate(A) if len(h.yseq) == 1]
        for j in idx_to_empty_str:
            tot_scores[j] = 0.0

        for j in range(batch):
            h = A[j]
            h.score += (tot_scores[j].item() - h.mmi_prev_score) * mmi_weight
            h.mmi_prev_score = tot_scores[j].item()

        A = [h for h in A if h.score > -1e8]
        return A
    

    "Version using step_intersect. Very slow. Only used when lookahead is very large"
    def Imp_A(self, A, nnet_output, den_scores, tu_sum, mmi_weight):
        batch = len(A)
        if batch == 0 or mmi_weight == 0:
            return A
        
        # For hypotheses without mmi_tot_score, compute the score sequences and assign
        texts = [h.yseq[1:] for h in A if h.mmi_tot_score is None]
        texts = [" ".join([self.char_list[x] for x in text]) for text in texts]
        indices = [i for i, h in enumerate(A) if h.mmi_tot_score is None]

        T, num_texts = nnet_output.size(1), len(texts)
        if num_texts > 0:
            num, _ = self.graph_compiler.compile(texts, self.P, replicate_den=False)
            supervision = torch.stack([torch.arange(num_texts),
                                       torch.zeros(num_texts),
                                       torch.ones(num_texts) * T], dim=1).int()
            dense_fsa_vec = k2.DenseFsaVec(nnet_output.repeat(num_texts, 1, 1), supervision)
            num_scores = step_intersect(num, dense_fsa_vec)
            assert len(num_scores) == num_texts
    
            tot_scores = [x - den_scores for x in num_scores]
            
            for i, score in zip(indices, tot_scores):
                A[i].mmi_tot_score = score.tolist()

        # selecting the scores accordingly
        assert all([h.mmi_tot_score is not None for h in A])
        ts = [tu_sum - len(h.yseq[1:]) for h in A]
        curr_scores = [max(h.mmi_tot_score[t: min(t + self.mas_lookahead, T)])
                      for t, h in zip(ts, A)]
        prev_scores = [h.mmi_prev_score for h in A]
        diff_scores = [a - b for a, b in zip(curr_scores, prev_scores)]

        for curr_s, diff_s, h in zip(curr_scores, diff_scores, A):
            h.score += mmi_weight * diff_s
            h.mmi_prev_score = curr_s
       
        # exclude all hypotheses whose end-states are not reachable 
        A = [h for h in A if h.score > -1e8]
        
        return A

    
