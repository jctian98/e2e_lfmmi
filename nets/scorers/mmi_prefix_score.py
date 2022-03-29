import os
import k2
import torch
from espnet.nets.scorer_interface import PartialScorerInterface
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm
from espnet.nets.scorers.mmi_utils import step_intersect

class MMIFrameScorer(PartialScorerInterface):
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list, weight_path):
        
        self.lang = lang
        self.device = device
        
        self.lexicon = Lexicon(lang)
        self.oov = self.oovid = open(self.lang / 'oov.txt').read().strip()
        self.graph_compiler = MmiTrainingGraphCompiler(self.lexicon, self.device, self.oov)
        self.phone_ids = self.lexicon.phone_symbols()
      
        self.lo = torch.nn.Linear(idim, len(self.phone_ids) + 1) 
        self.lm_scores = None

        for i in range(10):
            try:
                self.load_weight(rank, weight_path)
            except:
                print(f"{i}-th trail to load MMI matrix weight but fail")

        self.P = create_bigram_phone_lm(self.phone_ids)
        self.P.set_scores_stochastic_(self.lm_scores)
        
        self.char_list = char_list
        self.eos = sos_id  # <sos> is identical to <eos>
        self.blank = 0 # by default 0 means CTC blank 
        self.logzero = -10000

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

    def init_state(self, x):
        # (1) den_graphs
        texts = ["<UNK>"] # use a random text, just to get den graph
        _, den = self.graph_compiler.compile(texts, self.P, replicate_den=True)

        # (2) DenseFsaVec
        nnet_output = self.lo(x.unsqueeze(0))
        supervision = torch.Tensor([[0, 0, nnet_output.size(1)]]).int()
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

        # (3) den_scores: [B, T]
        den_scores = step_intersect(den, dense_fsa_vec)[0].unsqueeze(0)

        # (4) initialize prev_score by 0.0
        prev_score = torch.Tensor([0.0]).to(torch.float32)
        return nnet_output, den_scores, prev_score 

    def select_state(self, states, j):
        nnet_output_single, den_scores, prev_scores = states
        return nnet_output_single, den_scores, prev_scores[j]

    def score(**kargs):
        raise NotImplementedError

    def score_partial(self, y, next_tokens, state, hs_pad):
        nnet_output_single, den_scores, prev_score = state
        batch_size = len(next_tokens)

        # (1) num_graphs
        ys = torch.cat([y.unsqueeze(0).repeat(batch_size, 1), next_tokens.unsqueeze(1)], dim=1)
        texts = [" ".join([self.char_list[tid] for tid in text[1:]]) for text in ys]
        texts = [text.replace("<eos>", "").strip() for text in texts]
        num, _ = self.graph_compiler.compile(texts, self.P, replicate_den=False)

        # (2) DenseFsaVec
        supervision = torch.stack([
                      torch.arange(batch_size),
                      torch.zeros(batch_size),
                      torch.ones(batch_size) * nnet_output_single.size(1)
                      ], dim=1).to(torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output_single.repeat(batch_size, 1, 1), supervision)

        # (3) num_scores: [B, T]
        num_scores = torch.stack(step_intersect(num, dense_fsa_vec), dim=0) 

        # (4) compute frame scores and accumulate along t-axis
        tot_scores_frame = num_scores - den_scores
        tot_scores = torch.logsumexp(tot_scores_frame, dim=-1)

        # (5) post-process
        # <eos> means the exact probability rather than the prefix probability 
        eos_pos = torch.where(next_tokens == self.eos)[0]
        if len(eos_pos) > 0:
            tot_scores[eos_pos] = tot_scores_frame[eos_pos.item(), -1]

        # CTC blank is never allowed in hypothesis. kill it
        blk_pos = torch.where(next_tokens == self.blank)[0]
        if len(blk_pos) > 0:
            tot_scores[blk_pos] = self.logzero
        
        # (6) finalize
        tok_scores = tot_scores - prev_score
        state = nnet_output_single, den_scores, tot_scores
        return tok_scores, state

    def final_score(self, state):
        return 0     
                
