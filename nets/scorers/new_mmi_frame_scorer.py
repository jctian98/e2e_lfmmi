import k2
import torch
from espnet.nets.scorer_interface import PartialScorerInterface
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm
from espnet.nets.scorers.trace_frame import trace_frame

class MMIFrameScorer(PartialScorerInterface):
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
        
        self.char_list = char_list
        self.eos = sos_id  # <sos> is identical to <eos>
        self.blank = 0 # by default 0 means CTC blank 
        self.logzero = -10000

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
        torch.set_printoptions(sci_mode=False)

        x = x[:50]
        # (1) nnet_output
        x = x.unsqueeze(0)
        nnet_output = self.lo(x)

        # (2) den_scores
        texts = ["<UNK>"] # use a random text, just to get den graph
        _, den = self.graph_compiler.compile(texts, self.P, replicate_den=True)

        T = x.size()[1]

        den_scores = trace_frame(nnet_output, den, "<UNK>")
        # [T] -> [B, T]
        den_scores = den_scores.unsqueeze(0)
        print("den_score: ", den_scores)

        # (3) Prev Score is zero
        prev_score = torch.Tensor([0]).to(torch.float32)
        return nnet_output, den_scores, prev_score 

    def select_state(self, states, j):
        nnet_output_single, den_scores, prev_scores = states
        return nnet_output_single, den_scores, prev_scores[j]

    def score(**kargs):
        raise NotImplementedError

    def score_partial(self, y, next_tokens, state, hs_pad):
        # Warning: All frame-level scores are adopted in reverse order in time-axis 
        # since k2 requires a descending input length

        # (1) unpack state
        nnet_output_single, den_scores, prev_score = state
        batch_size = len(next_tokens)

        # (3) texts
        y = y.unsqueeze(0).repeat(batch_size, 1)
        next_tokens = next_tokens.unsqueeze(1)
        ys = torch.cat([y, next_tokens], dim=1)
        # This is for Chinese. Need more tuning on English
        if not "<space>" in self.char_list:
            texts = [" ".join([self.char_list[tid] for tid in text[1:]]) for text in ys]
            texts = [text.replace("<eos>", "").strip() for text in texts]
        else:
            texts = ["".join([self.char_list[tid] for tid in text[1:]]) for text in ys]
            texts = [text.replace("<eos>", "").replace("<space>", "<space> ").strip() for text in texts]

        num_scores = []
        for text in texts:
             print(text, flush=True)
             num_graph, _ = self.graph_compiler.compile([text], self.P, replicate_den=False)
             num_graph[0].draw(f"{text}_graph.svg")
             num_score = trace_frame(nnet_output_single, num_graph, text)
             num_scores.append(num_score)
        num_scores = torch.stack(num_scores, dim=0)

        #     minus the denominator scores
        # we should keep the frame-level result for <eos>
        tot_scores_frame = num_scores - den_scores
        tot_scores = torch.logsumexp(tot_scores_frame, dim=-1)
        print(tot_scores_frame)       
 
        # (5) treat <eos> and ctc <blk> specailly
        # <eos> means the exact probability rather than the prefix probability 
        eos_pos = torch.where(next_tokens == self.eos)[0]
        if len(eos_pos) > 0:
            tot_scores[eos_pos] = tot_scores_frame[eos_pos.item(), 0]

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
                
