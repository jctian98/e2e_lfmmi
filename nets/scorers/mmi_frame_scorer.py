import k2
import torch
from espnet.nets.scorer_interface import PartialScorerInterface
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm
# from espnet.nets.scorers.trace_frame import trace_frame
from espnet.nets.scorers.mmi_utils import step_intersect


class MMIFrameScorer(PartialScorerInterface):
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list):
        
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
                self.load_weight(rank)
            except:
                print(f"{i}-th trail to load MMI matrix weight but fail")

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
        # (1) nnet_output
        x = x.unsqueeze(0)
        nnet_output = self.lo(x)

        # (2) den_scores
        texts = ["<UNK>"] # use a random text, just to get den graph
        _, den = self.graph_compiler.compile(texts, self.P, replicate_den=True)

        #torch.set_printoptions(sci_mode=False)
        #fake_den_scores = trace_frame(nnet_output, den)

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
        den_scores = torch.cat(den_scores).unsqueeze(0) # [T] -> [B, T]
        print("den_scores: ", den_scores)

        ### DEBUG ###
        supervision = torch.Tensor([[0, 0, T]]).to(torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        den_scores_ = step_intersect(den, dense_fsa_vec)[0].unsqueeze(0)
        den_scores_ = torch.flip(den_scores_, [1]) 

        max_diff = torch.max(torch.abs(den_scores - den_scores_)).item()
        if abs(max_diff) > 0.02:
            print("denominator error: ", den_scores, den_scores_)
            raise ValueError
        ### END DEBUG ###

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

        # (2) acoustic
        T = nnet_output_single.size()[1]
        batch_size = len(next_tokens)
        num_egs = T * batch_size
        supervision = torch.stack([
                      torch.arange(num_egs),
                      torch.zeros(num_egs),
                      torch.arange(T, 0, -1).unsqueeze(1).repeat(1, batch_size).view(-1)
                      ], dim=1).to(torch.int32) 
        nnet_output = nnet_output_single.repeat(num_egs, 1, 1)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

        # (3) texts
        y = y.unsqueeze(0).repeat(num_egs, 1)
        next_tokens = next_tokens.unsqueeze(1).repeat(T, 1)
        ys = torch.cat([y, next_tokens], dim=1)
        
        # This is for Chinese. Need more tuning on English
        #if not "<space>" in self.char_list:
        #    texts = [" ".join([self.char_list[tid] for tid in text[1:]]) for text in ys]
        #    texts = [text.replace("<eos>", "").strip() for text in texts]
        #else:
        #    texts = ["".join([self.char_list[tid] for tid in text[1:]]) for text in ys]
        #    texts = [text.replace("<eos>", "").replace("<space>", "<space> ").strip() for text in texts]
        texts = [" ".join([self.char_list[tid] for tid in text[1:]]) for text in ys]
        texts = [text.replace("<eos>", "").strip() for text in texts]

        # (4) compute and accumulate
        num_graphs, _ = self.graph_compiler.compile(texts, self.P, replicate_den=False)
        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=10.0)         
        num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
        num_tot_scores = num_tot_scores.view(T, batch_size).transpose(0, 1) # -> [B, T]
        
        ### DEBUG ###
        supervision = torch.stack([
                      torch.arange(batch_size),
                      torch.zeros(batch_size),
                      torch.ones(batch_size).int() * T
                      ], dim=1).to(torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

        ys = ys[: batch_size]
        texts = [" ".join([self.char_list[tid] for tid in text[1:]]) for text in ys]
        texts = [text.replace("<eos>", "").strip() for text in texts]
        num_graphs, _ = self.graph_compiler.compile(texts, self.P, replicate_den=False)
        num_tot_scores_ = torch.stack(step_intersect(num_graphs, dense_fsa_vec), dim=0)
        num_tot_scores_ = torch.flip(num_tot_scores_, [1])

        max_diff = torch.max(torch.abs(num_tot_scores - num_tot_scores_)).item()
        if abs(max_diff) > 0.02:
            print("numerator error: ", num_tot_scores, num_tot_scores_)
            raise ValueError 
        ### END DEBUG ### 

        #     minus the denominator scores. 
        tot_scores_frame = num_tot_scores - den_scores
        tot_scores = torch.logsumexp(tot_scores_frame, dim=-1)

        # (5) treat <eos> and ctc <blk> specailly
        next_tokens = next_tokens.squeeze(1)[:batch_size] # recover the initail next_tokens
       
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
                
