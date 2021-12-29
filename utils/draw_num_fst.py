#!/usr/bin/env python3
# encoding: utf-8
import sys
import torch
import k2
from pathlib import Path
from espnet.snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from espnet.snowfall.lexicon import Lexicon
from espnet.snowfall.training.mmi_graph import create_bigram_phone_lm

def main():

    # compiler
    lang = Path("data/lang_k2mmi")
    lexicon = Lexicon(lang)
    device = torch.device("cuda:0")
    graph_compiler = MmiTrainingGraphCompiler(lexicon=lexicon, device=device)
    
    # P
    phone_ids = lexicon.phone_symbols()
    P = create_bigram_phone_lm(phone_ids)
    P = P.to(device)

    # compile num graph
    ys = ["S O U R C E <space> C O L O N"]
    num_graphs, _ = graph_compiler.compile(ys, P, replicate_den=True)
    num = num_graphs[0]

    # draw
    num.draw("num.svg") 


main() 

