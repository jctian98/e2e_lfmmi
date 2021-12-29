# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from functools import lru_cache
from typing import Iterable
from typing import List
from typing import Tuple
from pathlib import Path

import logging

import k2
import torch

from .ctc_graph import build_ctc_topo
from snowfall.common import get_phone_symbols
from snowfall.decoding.graph import compile_HLG


def find_first_disambig_symbol(symbols: k2.SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if k.startswith('#'))


class MmiMbrTrainingGraphCompiler(object):

    def __init__(self,
                 L_inv: k2.Fsa,
                 L_disambig: k2.Fsa,
                 G: k2.Fsa,
                 phones: k2.SymbolTable,
                 words: k2.SymbolTable,
                 device: torch.device,
                 oov: str = '<UNK>'):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
          L_disambig:
            L with disambig symbols. Its labels are phones and aux_labels
            are words.
          G:
            The language model.
          phones:
            The phone symbol table.
          words:
            The word symbol table.
          device:
            The target device that all FSAs should be moved to.
          oov:
            Out of vocabulary word.
        '''

        L_inv = L_inv.to(device)
        G = G.to(device)

        if L_inv.properties & k2.fsa_properties.ARC_SORTED != 0:
            L_inv = k2.arc_sort(L_inv)

        if G.properties & k2.fsa_properties.ARC_SORTED != 0:
            G = k2.arc_sort(G)

        assert L_inv.requires_grad is False
        assert G.requires_grad is False

        assert oov in words

        L = L_inv.invert()
        L = k2.arc_sort(L)

        self.L_inv = L_inv
        self.L = L
        self.phones = phones
        self.words = words
        self.device = device
        self.oov_id = self.words[oov]

        phone_symbols = get_phone_symbols(phones)
        phone_symbols_with_blank = [0] + phone_symbols

        ctc_topo = k2.arc_sort(
            build_ctc_topo(phone_symbols_with_blank).to(device))
        assert ctc_topo.requires_grad is False

        self.ctc_topo = ctc_topo
        self.ctc_topo_inv = k2.arc_sort(ctc_topo.invert())

        lang_dir = Path('data/lang_nosp')
        if not (lang_dir / 'HLG_uni.pt').exists():
            logging.info("Composing (ctc_topo, L_disambig, G)")
            first_phone_disambig_id = find_first_disambig_symbol(phones)
            first_word_disambig_id = find_first_disambig_symbol(words)
            # decoding_graph is the result of composing (ctc_topo, L_disambig, G)
            decoding_graph = compile_HLG(
                L=L_disambig.to('cpu'),
                G=G.to('cpu'),
                H=ctc_topo.to('cpu'),
                labels_disambig_id_start=first_phone_disambig_id,
                aux_labels_disambig_id_start=first_word_disambig_id)
            torch.save(decoding_graph.as_dict(),
                       lang_dir / 'HLG_uni.pt')
        else:
            logging.info("Loading pre-compiled HLG")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lang_dir / 'HLG_uni.pt'))

        assert hasattr(decoding_graph, 'phones')

        self.decoding_graph = decoding_graph.to(device)

    def compile(self, texts: Iterable[str],
                P: k2.Fsa) -> Tuple[k2.Fsa, k2.Fsa, k2.Fsa]:
        '''Create numerator and denominator graphs from transcripts
        and the bigram phone LM.

        Args:
          texts:
            A list of transcripts. Within a transcript, words are
            separated by spaces.
          P:
            The bigram phone LM created by :func:`create_bigram_phone_lm`.
        Returns:
          A tuple (num_graph, den_graph, decoding_graph), where

            - `num_graph` is the numerator graph. It is an FsaVec with
              shape `(len(texts), None, None)`.
              It is the result of compose(ctc_topo, P, L, transcript)

            - `den_graph` is the denominator graph. It is an FsaVec with the same
              shape of the `num_graph`.
              It is the result of compose(ctc_topo, P).

            - decoding_graph: It is the result of compose(ctc_topo, L_disambig, G)
              Note that it is a single Fsa, not an FsaVec.
        '''
        assert P.device == self.device
        P_with_self_loops = k2.add_epsilon_self_loops(P)

        ctc_topo_P = k2.intersect(self.ctc_topo_inv,
                                  P_with_self_loops,
                                  treat_epsilons_specially=False).invert()
        ctc_topo_P = k2.arc_sort(ctc_topo_P)

        num_graphs = self.build_num_graphs(texts)

        num_graphs_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            num_graphs)

        num_graphs_with_self_loops = k2.arc_sort(num_graphs_with_self_loops)

        num = k2.compose(ctc_topo_P,
                         num_graphs_with_self_loops,
                         treat_epsilons_specially=False,
                         inner_labels='phones')
        num = k2.arc_sort(num)

        ctc_topo_P_vec = k2.create_fsa_vec([ctc_topo_P.detach()])
        indexes = torch.zeros(len(texts),
                              dtype=torch.int32,
                              device=self.device)
        den = k2.index_fsa(ctc_topo_P_vec, indexes)

        return num, den, self.decoding_graph

    def build_num_graphs(self, texts: List[str]) -> k2.Fsa:
        '''Convert transcript to an Fsa with the help of lexicon
        and word symbol table.

        Args:
          texts:
            Each element is a transcript containing words separated by spaces.
            For instance, it may be 'HELLO SNOWFALL', which contains
            two words.

        Returns:
          Return an FST (FsaVec) corresponding to the transcript. Its `labels` are
          phone IDs and `aux_labels` are word IDs.
        '''
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split(' '):
                if word in self.words:
                    word_ids.append(self.words[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        fsa = k2.linear_fsa(word_ids_list, self.device)
        fsa = k2.add_epsilon_self_loops(fsa)
        num_graphs = k2.intersect(self.L_inv,
                                  fsa,
                                  treat_epsilons_specially=False).invert_()
        num_graphs = k2.arc_sort(num_graphs)
        return num_graphs
