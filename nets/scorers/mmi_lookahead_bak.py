import k2
import torch
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.lm.lm_utils import make_lexical_tree
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm


class MMILookaheadScorer(PartialScorerInterface):
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

        self.wdict = self.lexicon.words._sym2id
        self.pdict = self.lexicon.phones._sym2id
        self.lexroot = make_lexical_tree(self.wdict, self.pdict, "<UNK>")
        self.char_list = char_list
        self.space_str = "<space>"
        self.oovid = int(open(self.lang / 'oov.int').read().strip()) 
        self.unk_id = self.wdict["<UNK>"]
        self.sos_id = sos_id
        
        self.logzero = -10000.0

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
        # TODO: determine the format of state
        x = x.unsqueeze(0)
        nnet_output = self.lo(x)

        T = x.size()[1]
        supervision = torch.Tensor([[0, 0, T]]).to(torch.int32)

        prev_score = torch.Tensor([0]).to(torch.float32)
        return nnet_output, supervision, prev_score, self.lexroot 

    def select_state(self, states, j):
        # TODO: select the state
        pass

    def score(**kargs):
        raise NotImplementedError

    def score_partial(self, y, next_tokens, state, hs_pad):
        # (1) unpack state
        nnet_output_single, supervision_single, prev_score, node = state
        batch_size = next_tokens.size()[0]

        # (2) handle the prefix and candidate, find the prefix_ids and intervals
        # prefix
        prefix = [self.char_list[x] for x in y[1:]]
        # prefix = [char_list[x] for x in y[1:]]
        prefix = "".join(prefix).replace(self.space_str, " ").strip().split(' ')
        word_transition = False
        if self.char_list[y[-1]] == self.space_str:
            # end with <space>, word transition
            prefix = prefix[:-1]
            word_transition = True
        prefix = [self.wdict.get(w, self.oovid)  for w in prefix]
        
        # candidate: any interval: start-1, end -> [start, end)
        intervals, next_nodes = [], []
        for tok in next_tokens:
            tok = self.char_list[tok]
            
            if tok == self.space_str: 
                next_nodes.append(self.lexroot)
            # attention symbol but not mmi symbol
            if not tok in self.pdict:
                intervals.append((self.unk_id - 1, self.unk_id))
                next_nodes.append(None)
            else:
                pid = self.pdict[tok]
                if pid in node[0]: # successors exists: valid word
                    intervals.append(node[0][pid][2])
                    next_nodes.append(node[0][pid])
                else: # OOV: 
                    intervals.append((self.unk_id - 1, self.unk_id))
                    next_nodes.append(None)
        intervals = [(l+1, r+1) for l, r in intervals] # work around: to be compatible with lex-tree 
        split_intervals, indices = self.split_intervals(intervals)
        num_split = len(split_intervals)

        # (3) acoustic information
        supervision = supervision_single.repeat(num_split, 1)
        nnet_output = nnet_output_single.repeat(num_split, 1, 1)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        
        # (4) accumulate probability
        num_graphs = self.graph_compiler.compile_nums_for_prefix_scoring(prefix, split_intervals, self.P)
        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=5.0)
        num_tot_scores_split = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
        num_tot_scores = torch.zeros(batch_size).to(y.device)
        for i in range(batch_size):
            num_tot_scores[i] = torch.logsumexp(num_tot_scores_split[indices[i]], dim=-1)

        num_tok_scores = num_tot_scores - prev_score
        return num_tok_scores, (nnet_output_single, supervision_single, num_tot_scores, next_nodes)    
         

    def split_intervals(self, olds, max_interval=2000):
        # it may cause error in k2 if the interval is too large. e.g., > 5k
        # so a large interval should be split into multiple small intervals.
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
