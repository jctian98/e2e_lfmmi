import k2
import torch
import torch.nn.functional as F
import sys
import jieba
from espnet.nets.scorer_interface import PartialScorerInterface
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.warpper.mmi_utils import build_word_mapping, convert_transcription

# All methods are overrided
class MMIPrefixScores(PartialScorerInterface):
    def __init__(self, lang, device, idim, sos_id, rank, use_segment, char_list):
        """
        lang: Path object of lang dir
        device: torch device
 
        We do not refer K2MMI module in model to avoid device conflict
        """
        self.lang = lang
        self.device = device
        self.sos_id = sos_id # sos and eos are the same
        self.char_list = char_list
        self.oovid = int(open(self.lang / 'oov.int').read().strip())

        self.lexicon = Lexicon(lang)
        self.graph_compiler = MmiTrainingGraphCompiler(
                                 self.lexicon, device=device)    

        self.phone_ids = self.lexicon.phone_symbols()
        # ignore dropout here; +1 for blank; maybe need log_softmax
        self.lo = torch.nn.Linear(idim, len(self.phone_ids) + 1) 
        self.lm_scores = None
        self.load_weight(rank)

        self.P = create_bigram_phone_lm(self.phone_ids)
        self.P.set_scores_stochastic_(self.lm_scores)

        # if True, the numerator score would be calculated on segments instead of single characters
        self.use_segment = use_segment 

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
        """
        x: 2-D tensor, utterance encoder output without batch
        
        return: nnet_output: [B, T, D], log_distribution without being normalized (exp-sum!=1)
                supervision: supervision for single utterance
                prev_score: initial score

        The denominator score is independent to the prefix. It is a constant for any hypothesis.
        So it would be ignored during decoding: We only consider the numerator, and the initial 
        score is set 0.
        """
        
        x = x.unsqueeze(0)
        nnet_output = self.lo(x)
        
        T = x.size()[1]
        supervision = torch.Tensor([[0, 0, T]]).to(torch.int32)

        prev_score = torch.Tensor([0]).to(torch.float32)
        return nnet_output, supervision, prev_score 

    def select_state(self, states, j):
        nnet_output_single, supervision_single, prev_scores = states
        prev_score = torch.Tensor([prev_scores[j]])
        return nnet_output_single, supervision_single, prev_score

    def score(**kargs):        
        raise NotImplementedError

    def score_partial(self, y, next_tokens, state, hs_pad):
        """
        y: prefix hypothesis, start with sos
        next_tokens: candidates for next token
	state: decoding state
	hs_pad: encoder output, ignore

        return:
        tok_scores: prefix g, new token c, hypothesis h. token_scores = score(h) - score(g)
                    which is the score for c.
        state: directly copy nnet_output_single and supervision_single. 
               save the scores for diffeent score(h), it would be score(g) in next round.
        """
        nnet_output_single, supervision_single, prev_score = state
        batch_size = next_tokens.size()[0]

        # acoustic
        supervision = supervision_single.repeat(batch_size, 1)
        nnet_output = nnet_output_single.repeat(batch_size, 1, 1)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

        # texts
        y = y.unsqueeze(0).repeat(batch_size, 1)
        next_tokens = next_tokens.unsqueeze(1)
        ys = torch.cat([y, next_tokens], dim=-1)
        # texts = convert_transcription(ys, self.word_mapping, self.lexicon.words,
        #                               self.oovid, [self.sos_id])
        texts = ["".join([self.char_list[tid] for tid in text[1:]]) for text in ys]
        texts = [text.replace("<space>", " ") for text in texts]
        print(texts)

        if self.use_segment:
            texts = self.segmentation(texts)
        
        num_graphs, _ = self.graph_compiler.compile(
                            texts, self.P, replicate_den=True)
        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=10.0)
        
        num_tot_scores = num_lats.get_tot_scores(log_semiring=True,
                                                 use_double_scores=True)
        tok_scores = num_tot_scores - prev_score
        
        return tok_scores, (nnet_output_single, supervision_single, num_tot_scores)

    def score(self, state):
        raise NotImplementedError

    def final_score(self, state):
        # This will add a score for the <eos>
        # We do not give a special score for the last <eos>
        return 0

    def segmentation(self, ys):
        ys = [y.replace(" ", "") for y in ys]
        ys = [jieba.cut(y, cut_all=False) for y in ys]
        ys = [" ".join(list(y)) for y in ys]
        return ys
