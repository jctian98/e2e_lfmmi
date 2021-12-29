import torch
import k2
import torch.nn.functional as F
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import List
from typing import Union
from k2 import Fsa, SymbolTable
from espnet.snowfall.lexicon import Lexicon
from snowfall.training.ctc_graph import CtcTrainingGraphCompiler
from espnet.snowfall.objectives.ctc import CTCLoss
from espnet.snowfall.common import get_phone_symbols, find_first_disambig_symbol, get_texts
from espnet.snowfall.training.ctc_graph import build_ctc_topo
from espnet.snowfall.decoding.graph import compile_HLG
from lhotse.utils import nullcontext
from espnet.snowfall.warpper.mmi_utils import build_word_mapping, convert_transcription, encode_supervision

"""
June 29th
self.phone_ids and self.phones are not identical
self.phone_ids is built from lexicon and have no esp/blk
self.phones is read from file and have esp/blk
Not clear would this leads to a bug
"""

class K2CTC(torch.nn.Module):

    def __init__(self, 
                 idim, 
                 lang, 
                 char_list, 
                 device, 
                 dropout, 
                 den_scale, 
                 eos_id, 
                 pad_id=-1,
                 use_segment=False):

        """
        idim: input dim, usually the transformer output dim
        lang: k2 lang directory
        word_mapping: mapping from attention vocab to MMI vocab
        device: torch.device object. device to build the loss module
        dropout: dropout rate for linear out layer
        den_scale: den_scale for MMI loss computation
        eos_id: end of sentence id
        pad_id: id of padding in ys_pad
        use_segment: If true, the supervision of MMI training would use "texts"
                     instead of ys_pad. Sensitive for Chinese
        """

        super().__init__()
        self.device = device
        
        # compiler
        self.lang = Path(lang)
        self.lexicon = Lexicon(self.lang)
        self.graph_compiler = CtcTrainingGraphCompiler(
                              L_inv=self.lexicon.L_inv,
                              phones=self.lexicon.phones,
                              words=self.lexicon.words
                              )

        # bigram LM
        self.phone_ids = self.lexicon.phone_symbols() # blank excluded
        self.words = self.lexicon.words
        self.phones = self.lexicon.phones 

        # linear
        self.idim = idim
        self.odim = len(self.phone_ids) + 1
        self.lo = torch.nn.Sequential(
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(self.idim, self.odim)
                                     )

        # others
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.use_segment=use_segment
        self.char_list = char_list
        self.oovid = int(open(self.lang / 'oov.int').read().strip())
        self.probs = None # for visualization
        self.HLG = None # Decoding graph. build by "decode_init"
        print("INFO from CTC module:")
        print(f"device: {device}")
        print(f"use segment info: {use_segment}")
        print(f"self.lo {self.lo}")
        print(f"number of phones {len(self.phone_ids)}")

    # softmax, log_softmax and argmax for decoding and visualization
    def log_softmax(self, hs_pad):
        return self.softmax(hs_pad).log()

    def softmax(self, hs_pad):
        # self.probs is required by visualization
        self.probs = F.softmax(self.lo(hs_pad), dim=2)
        return self.probs

    def argmax(self, hs_pad):
        return torch.argmax(self.lo(hs_pad), dim=2)

    def forward(self, hs_pad, hlens, ys_pad, texts):
        
        if self.use_segment:
            ys = texts
            if "<space>" in self.char_list:
                ys = [y.replace(" ", "<space> ") for y in ys]
        else:
            # split by every character: BPE or chinese chars
            ys = [[self.char_list[c] for c in y if c != self.pad_id] for y in ys_pad]
            ys = [" ".join(y).replace("<eos>", "") for y in ys]
 
        supervision, indices = encode_supervision(hlens)
        ys = [ys[i] for i in indices]
        
        nnet_output = self.lo(hs_pad)
        nnet_output = F.log_softmax(nnet_output, dim=-1)
        
        loss_fn = CTCLoss(self.graph_compiler)

        grad_context = nullcontext if self.training else torch.no_grad

        with grad_context():
            ctc_loss, ctc_frames, all_frames = loss_fn(
                nnet_output, ys, supervision)
        batch_size = hlens.size()[0]
        ctc_loss /= batch_size
        return - ctc_loss

    def decode(self, nnet_output, hlens, texts, is_english):
       
        # add linear function 
        nnet_output = self.lo(nnet_output)
        supervision, indices = encode_supervision(hlens)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

        # Show MMI loss before decoding
        self.P.set_scores_stochastic_(self.lm_scores)
        if self.training:
            assert self.P.is_cpu
            assert self.P.requires_grad is False
        """
        loss_fn = LFMMILoss(
            graph_compiler = self.graph_compiler,
            P = self.P,
            den_scale = self.den_scale
        )

        grad_context = nullcontext if self.training else torch.no_grad
       
        if not is_english: 
            texts_reorder = [
                      " ".join(list(text.replace(" ", "")))
                    for text in texts]

            texts_reorder = [texts_reorder[i] for i in indices]
        else:
            texts_reorder = [texts[i] for i in indices]
        with grad_context():
            mmi_loss, tot_frames, all_frames = loss_fn(
                nnet_output, texts_reorder, supervision)
        mmi_loss = - mmi_loss / len(texts)
        print("MMI Loss: ", mmi_loss)
        """
        assert nnet_output.device == self.HLG.device
        # 7.0 output beam is tunable
        lattices = k2.intersect_dense_pruned(self.HLG, dense_fsa_vec, 20.0, 7.0, 30,
                                             10000)
        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        
        assert best_paths.shape[0] == len(texts)
        hyps = get_texts(best_paths, indices)
        assert len(hyps) == len(texts)

        results = []
        batch_size =len(texts)
        for i in range(batch_size):
            hyp_words = [self.words.get(x) for x in hyps[i]]
            ref_words = texts[i].split(' ')
            
            if not is_english:
                hyp = "".join(hyp_words).replace(" ", "")
                ref = "".join(ref_words).replace(" ", "")
            else:
                hyp = " ".join(hyp_words)
                ref = " ".join(ref_words)
            print("#"*20)
            print(f"Reference: {ref}")
            print(f"Hypothesis: {hyp}")
            sys.stdout.flush()

            if not is_english:
                ref_char = list(ref)
                hyp_char = list(hyp)
            else:
                ref_char = ref.split()
                hyp_char = hyp.split()
            results.append((ref_char, hyp_char))

        return results

    def decode_init(self):
        # Build HLG.fst
        phone_ids = get_phone_symbols(self.phones) # will remove 0
        phone_ids_with_blank = [0] + phone_ids
        ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))
        if not os.path.exists(self.lang / 'HLG.pt'):
            logging.debug("Loading L_disambig.fst.txt")
            with open(self.lang / 'L_disambig.fst.txt') as f:
                L = k2.Fsa.from_openfst(f.read(), acceptor=False)
            logging.debug("Loading G.fst.txt")
            with open(self.lang / 'G.fst.txt') as f:
                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            first_phone_disambig_id = find_first_disambig_symbol(self.phones)
            first_word_disambig_id = find_first_disambig_symbol(self.words)
            print("first disambig symbol: ", first_phone_disambig_id, first_word_disambig_id, flush=True)
            HLG = compile_HLG(L=L,
                             G=G,
                             H=ctc_topo,
                             labels_disambig_id_start=first_phone_disambig_id,
                             aux_labels_disambig_id_start=first_word_disambig_id)
            torch.save(HLG.as_dict(), self.lang / 'HLG.pt')
        else:
            logging.debug("Loading pre-compiled HLG")
            d = torch.load(self.lang / 'HLG.pt')
            HLG = k2.Fsa.from_dict(d)

        HLG = HLG.to(self.device)
        HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
        HLG.requires_grad_(False)
        if not hasattr(HLG, 'lm_scores'):
            HLG.lm_scores = HLG.scores.clone()
        self.HLG = HLG
        print("Successful Initialize Decoding HLG")
        
    def dump_weight(self, rank):
        d = {}
        for k, v in self.named_parameters():
            print(f"Found parameter {k} with shape {v.size()}")
            d[k] = v
        save_path = self.lang / f"ctc_param.{rank}.pth"
        torch.save(d, save_path)
