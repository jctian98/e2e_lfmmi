import k2
import torch
import numpy as np
from pathlib import Path
from espnet.snowfall.lexicon import Lexicon
from espnet.snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from espnet.snowfall.warpper.mmi_utils import encode_supervision
from espnet.snowfall.training.mmi_graph import create_bigram_phone_lm


def build_word_fsa(prefix_ids, candidate_intervals):
    batch = len(candidate_intervals)    

    # Prefix part 
    prefix_len = len(prefix_ids)
    start_state = np.arange(prefix_len)
    end_state = np.arange(prefix_len) + 1
    labels = np.array(prefix_ids)
    scores = np.zeros(prefix_len)
    
    prefix_part = np.stack([start_state, end_state, labels, scores], axis=1)

    # candidate part
    candidate_parts = []
    for start, end in candidate_intervals: 
        num_candidate = end - start
        start_state = np.ones(num_candidate) * prefix_len 
        end_state = np.ones(num_candidate) * (prefix_len + 1)
        labels = np.arange(start, end)
        scores = np.zeros(num_candidate)
        candidate_part = np.stack([start_state, end_state, labels, scores], axis=1)
        candidate_parts.append(candidate_part)

    # end arc
    end_arc = np.array([[prefix_len + 1, prefix_len + 2, -1, 0]])
   
    # assemble: do not need to arc_sort 
    num_vec = []
    for i, candidate_part in enumerate(candidate_parts):
        num_mat = np.concatenate([prefix_part, candidate_part, end_arc], axis=0)
        num_mat = torch.from_numpy(num_mat).to(torch.int32)
        num_vec.append(num_mat)

    num_vec = [k2.Fsa.from_dict({"arcs": num}) for num in num_vec]
    num_vec = k2.create_fsa_vec(num_vec)
    return num_vec
    
if __name__ == '__main__':
    lang = Path("data/lang_char")
    device = torch.device("cpu")
    lexicon = Lexicon(lang)
    compiler = MmiTrainingGraphCompiler(lexicon, device)
    phones = lexicon.phone_symbols()
    P = create_bigram_phone_lm(phones).to(device)

    prefix_ids = [1] 
    candidate_intervals =  [[58968, 60968], [60968, 62968], [62968, 64968], [64968, 66298]]
    num_graphs = compiler.compile_nums_for_prefix_scoring(prefix_ids, candidate_intervals, P)

    batch = len(candidate_intervals)
    nnet_output = torch.randn(batch, 500, len(phones) + 1).to(device)
    nnet_output = torch.nn.functional.log_softmax(nnet_output, dim=-1)
    hlens = torch.ones(batch) * 500
    supervision, _ = encode_supervision(hlens)
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
    num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=5.0)
    print(num_lats[0].as_dict())
    num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
    print(num_tot_scores)
