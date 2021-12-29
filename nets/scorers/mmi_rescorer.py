import k2
import torch
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm
from espnet.asr.asr_utils import parse_hypothesis

class MMIRescorer(object):
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
        
        self.char_list = char_list

        if not char_list[-2][0].isalnum():
            self.bpe_space = char_list[-2][0]
            print("use bpe. bpe space is: ", self.bpe_space)
        else:
            self.bpe_space = ""
        
    def load_weight(self, rank):
        # load lo weight and lm_scores
        ckpt_dict = torch.load(self.lang / f"mmi_param.{rank}.pth")
        for v in ckpt_dict.values():
            v.requires_grad=False
        self.lm_scores = ckpt_dict["lm_scores"]
        lo_dict = {"weight": ckpt_dict["lo.1.weight"],
                   "bias": ckpt_dict["lo.1.bias"]}
        self.lo.load_state_dict(lo_dict)

    def score(self, x, nbest_hyps, v2=False):
        batch_size = len(nbest_hyps)        

        # (1) acoustic
        x = x.unsqueeze(0)
        nnet_output = self.lo(x)
        T = x.size()[1]
        supervision = torch.Tensor([[0, 0, T]]).to(torch.int32)
        
        nnet_output = nnet_output.repeat(batch_size, 1, 1)
        supervision = supervision.repeat(batch_size, 1)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

        # (2) texts
        """
        texts, scores = [], []
        for idx, hyp in enumerate(nbest_hyps):
            if v2:
                tokenid = hyp.yseq.tolist()[1:]
                text = " ".join([char_list[x] for x in tokenid])
            else:
                text, token, tokenid, score = parse_hypothesis(hyp, char_list)
            text = text.replace("<eos>", "").strip()
            texts.append(text)
            text_split = text.strip().split()
            for tok in text_split:
                if not tok in self.lexicon.words:
                    print(f"{idx}: Found oov {tok}")
        """
        texts = []
        for idx, hyp in enumerate(nbest_hyps):
            if v2:
                tokenid = hyp.yseq[1:]
                tokens = [self.char_list[x] for x in tokenid]
                if isinstance(tokenid, torch.Tensor):
                    tokenid = tokenid.tolist()
                if "<space>" in self.char_list: # English
                    text = "".join(tokens).replace("<space>", " ")
                else: # Mandarin, BPE
                    text = " ".join(tokens)
                text = text.replace("<eos>", "") 
            else:
                text, _, tokenid, _ = parse_hypothesis(hyp, self.char_list)

            # BPE space:
            if self.bpe_space is not None:
                text = text.replace(" ", "").replace(self.bpe_space, " ").strip() 

            texts.append(text)

        # (3) computation
        num_graphs, _ = self.graph_compiler.compile(texts, self.P, replicate_den=True)
        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=10.0)
        num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
        
                
        # (4) add MMI scores
        new_hyps = []
        for i, hyp in enumerate(nbest_hyps):
            if v2:
                if hasattr(hyp, "scores"):
                    hyp.scores["mmi_tot_score"] = num_tot_scores[i].item()
                else:
                    setattr(hyp, 'mmi_tot_score', num_tot_scores[i].item())
            else:
                hyp["mmi_tot_score"] = num_tot_scores[i].item()
            new_hyps.append(hyp)
        return new_hyps 
