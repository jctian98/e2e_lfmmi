import torch
import k2
from pathlib import Path
from snowfall.lexicon import Lexicon
from snowfall.training.mmi_graph import create_bigram_phone_lm, MmiTrainingGraphCompiler


def main():
    lang = Path("data/lang_k2mmi")
    lexicon = Lexicon(lang)
    device = torch.device("cpu")
    graph_compiler = MmiTrainingGraphCompiler(lexicon, device=device)

    phone_ids = lexicon.phone_symbols()
    P = create_bigram_phone_lm(phone_ids)
    
    dim = len(phone_ids) + 1
    T = 100
    nnet_output = torch.rand(1, T, dim)
    supervision = torch.Tensor([[0, 0, T]]).to(torch.int32)
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

    texts = ['你 好']
    num_graphs, _ = graph_compiler.compile(texts, P, replicate_den=False)

    # num_lats = k2.intersection_dense(num_graphs, dense_fsa_vec, output_beam=10.0)
    num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)

    print(num_tot_scores)

main()
    
