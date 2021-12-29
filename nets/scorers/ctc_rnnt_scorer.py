import k2
import torch
import math
from snowfall.training.ctc_graph import CtcTrainingGraphCompiler
from snowfall.lexicon import Lexicon


class CTCRNNTScorer():
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list):
        
        self.device = device

        # compiler
        self.lang = lang
        self.lexicon = Lexicon(lang)
        self.graph_compiler = CtcTrainingGraphCompiler(
                              L_inv=self.lexicon.L_inv,
                              phones=self.lexicon.phones,
                              words=self.lexicon.words
                              )

        # linear
        self.phone_ids = self.lexicon.phone_symbols()
        self.lo = torch.nn.Linear(idim, len(self.phone_ids) + 1) 
        self.load_weight(rank)

        self.char_list = char_list

    def load_weight(self, rank):
        # load lo weight and lm_scores
        ckpt_dict = torch.load(self.lang / f"ctc_param.{rank}.pth")
        for v in ckpt_dict.values():
            v.requires_grad=False
        lo_dict = {"weight": ckpt_dict["lo.1.weight"],
                   "bias": ckpt_dict["lo.1.bias"]}
        self.lo.load_state_dict(lo_dict)

    def den_scores(self, x):
        # (1) nnet_output
        x = x.unsqueeze(0)
        nnet_output = self.lo(x)
        nnet_output = torch.nn.functional.log_softmax(nnet_output, dim=-1)

        # None is to be compatible with MMIRNNTScorer
        return nnet_output, None 

    def batch_score(self, A, nnet_output, den_scores, tu_sum, mmi_weight):
        print("tu_sum: ", tu_sum)

        batch = len(A)
        if batch == 0:
            return A

        # (1) supervision
        # +1 since frame start with 0; +1 since redundant blank
        # the supervision must be descending order.
        ts = [tu_sum - len(h.yseq) + 2 for h in A]
        ts = torch.Tensor(ts).long()
        supervision = torch.stack([torch.arange(batch),
                                   torch.zeros(batch),
                                   ts
                                  ], dim=1).to(torch.int32) 
        indices = torch.argsort(supervision[:, 2], descending=True)
        supervision = supervision[indices]
        dense_fsa_vec = k2.DenseFsaVec(nnet_output.repeat(batch, 1, 1),
                                       supervision)

        # (2) texts
        texts = [h.yseq[1:] for h in A] # exclude starting <sos>
        texts = [" ".join([self.char_list[x] for x in text]) for text in texts] # need modification for BPE
        texts = [texts[idx] for idx in indices] # reorder
        graphs = self.graph_compiler.compile(texts)

        # (3) intersection.  
        lats = k2.intersect_dense(graphs, dense_fsa_vec, output_beam=10.0)
        tot_scores = lats.get_tot_scores(log_semiring=True, use_double_scores=True)
        tot_scores = torch.where(tot_scores == -math.inf, 0.0, tot_scores)
        
        # (4) assign and post-process
        # Question: How to deal with the hypothesis with empty yseq
        idx_to_empty_str = [j for j, x in enumerate(texts) if x == ""]
        for j in idx_to_empty_str:
            tot_scores[j] = 0.0

        for j in range(batch):
            h = A[indices[j]]

            # step_score = (tot_scores[j] - h.mmi_tot_score)*mmi_weight
            h.score += (tot_scores[j].item() - h.mmi_tot_score) * mmi_weight
            h.mmi_tot_score = tot_scores[j].item()
            # print(f"idx: {indices[j]} | Hypothesis: {texts[j]} | CTC Score: {h.mmi_tot_score} | Tot Score: {h.score} | CTC step Score: {step_score}", flush=True)
        
        return A
         
