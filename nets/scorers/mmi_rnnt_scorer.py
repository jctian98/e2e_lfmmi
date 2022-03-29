import os
import k2
import torch
import math
from espnet.nets.scorer_interface import PartialScorerInterface
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm


class MMIRNNTScorer():
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list, weight_path, lookahead=0):
        
        self.lang = lang
        self.device = device
        
        self.lexicon = Lexicon(lang)
        self.oov = self.oovid = open(self.lang / 'oov.txt').read().strip()
        self.graph_compiler = MmiTrainingGraphCompiler(self.lexicon, self.device, self.oov)
        self.phone_ids = self.lexicon.phone_symbols()
      
        self.lo = torch.nn.Linear(idim, len(self.phone_ids) + 1) 
        self.lm_scores = None
        self.load_weight(rank, weight_path)

        self.P = create_bigram_phone_lm(self.phone_ids)
        self.P.set_scores_stochastic_(self.lm_scores)
        
        self.char_list = char_list
        self.eos = sos_id  # <sos> is identical to <eos>
        self.blank = 0 # by default 0 means CTC blank 
        self.logzero = -10000

        self.lookahead = lookahead

    def load_weight(self, rank, path):
        # load lo weight and lm_scores
        ckpt_path = os.path.join(path, f"mmi_param.{rank}.pth") 
        ckpt_dict = torch.load(ckpt_path)
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

        T = x.size()[1]
        den_scores = []
        # use a loop since denominator would consume much memory
        # in acscending order
        for t in range(1, T+1):
            supervision = torch.Tensor([[0, 0, t]]).to(torch.int32) # [idx, start, length]
            dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
            den_lats = k2.intersect_dense(den, dense_fsa_vec, output_beam=10.0)
            den_tot_scores = den_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
            den_scores.append(den_tot_scores)
        den_scores = torch.cat(den_scores).unsqueeze(0) # [T] -> [B, T]

        return nnet_output, den_scores

    def batch_rescore(self, A, h):
        ans = []
        start = 0
        while start < len(A):
            ans.append(self._batch_rescore(A[start: start + 50], h))
            start += 50

        return A

    def _batch_rescore(self, A, h):
        nnet_output = self.lo(h.unsqueeze(0))
        batch, T = len(A), nnet_output.size(1)
        if batch == 0:
            return A

        texts = [h.yseq[1:] for h in A]
        texts = [" ".join([self.char_list[x] for x in text]) for text in texts]
        num_graphs, _ = self.graph_compiler.compile(texts, self.P, replicate_den=False)

        supervision = torch.stack([torch.arange(batch),
                                   torch.zeros(batch),
                                   torch.ones(batch) * T
                                  ], dim=1).to(torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output.repeat(batch, 1, 1),
                                       supervision)

        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=30.0)
        num_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)

        for h, s in zip(A, num_scores):
            h.mmi_tot_score = s.item()
        
        return A

    # batch score without time-axis lookahead. TODO: remove the indices as the order is not important
    def batch_score(self, A, nnet_output, den_scores, tu_sum, mmi_weight):
        
        batch, T = len(A), nnet_output.size(1)
        if batch == 0:
            return A

        # (1) supervision
        # +1 since frame start with 0; +1 since redundant <sos>
        # the supervision must be descending order.
        ts = [tu_sum - len(h.yseq) + 2 for h in A]
        ts = torch.Tensor(ts).long()
        supervision = torch.stack([torch.arange(batch),
                                   torch.zeros(batch),
                                   ts
                                  ], dim=1).to(torch.int32) 
        indices = torch.argsort(supervision[:, 2], descending=True)
        supervision = supervision[indices]
        # dense_fsa_vec = k2.DenseFsaVec(nnet_output.repeat(batch, 1, 1),
        #                                supervision)

        # (2) texts
        texts = [h.yseq[1:] for h in A] # exclude starting <sos>
        texts = [" ".join([self.char_list[x] for x in text]) for text in texts] # need modification for BPE
        texts = [texts[idx] for idx in indices] # reorder
        num_graphs, _ = self.graph_compiler.compile(texts, self.P, replicate_den=False)

        # (3) intersection. 
        num_score_collection = []
        for i in range(self.lookahead + 1):
            supervision = torch.stack([torch.arange(batch),
                                       torch.zeros(batch),
                                       torch.clamp(ts + i, min=1, max=T)
                                       ], dim=1).to(torch.int32)[indices]
            dense_fsa_vec = k2.DenseFsaVec(nnet_output.repeat(batch, 1, 1),
                                           supervision)
            num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=200.0)
            num_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
            num_scores = torch.where(num_scores == -math.inf, 0.0, num_scores)
            num_score_collection.append(num_scores)

        num_scores = torch.stack(num_score_collection, dim=1).max(1)[0]
        ts = torch.Tensor([ts[j] for j in indices]).long()
        tot_scores = num_scores - den_scores[0][ts-1] # -1: num_frames -> idx_frames

        # (4) assign and post-process
        idx_to_empty_str = [j for j, x in enumerate(texts) if x == ""]
        for j in idx_to_empty_str:
            tot_scores[j] = 0.0

        for j in range(batch):
            h = A[indices[j]]
            h.score += (tot_scores[j].item() - h.mmi_tot_score) * mmi_weight
            h.mmi_tot_score = tot_scores[j].item()
        
        return A
