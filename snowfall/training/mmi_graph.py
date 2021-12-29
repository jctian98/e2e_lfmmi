# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from typing import Iterable
from typing import List
from typing import Tuple
import numpy as np
import k2
import torch

from .ctc_graph import build_ctc_topo
from snowfall.common import get_phone_symbols
from ..lexicon import Lexicon


def create_bigram_phone_lm(phones: List[int]) -> k2.Fsa:
    '''Create a bigram phone LM.
    The resulting FSA (P) has a start-state and a state for
    each phone 1, 2, ....; and each of the above-mentioned states
    has a transition to the state for each phone and also to the final-state.

    Caution:
      blank is not a phone.

    Args:
      A list of phone IDs.

    Returns:
      An FSA representing the bigram phone LM.
    '''
    assert 0 not in phones
    final_state = len(phones) + 1
    rules = ''
    for i in range(1, final_state):
        rules += f'0 {i} {phones[i-1]} 0.0\n'

    for i in range(1, final_state):
        for j in range(1, final_state):
            rules += f'{i} {j} {phones[j-1]} 0.0\n'
        rules += f'{i} {final_state} -1 0.0\n'
    rules += f'{final_state}'
    return k2.Fsa.from_str(rules)


class MmiTrainingGraphCompiler(object):

    def __init__(
            self,
            lexicon: Lexicon,
            device: torch.device,
            oov: str = '<UNK>'
    ):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
        phones:
          The phone symbol table.
        words:
          The word symbol table.
        oov:
          Out of vocabulary word.
        '''
        self.lexicon = lexicon
        L_inv = self.lexicon.L_inv.to(device)

        if L_inv.properties & k2.fsa_properties.ARC_SORTED != 0:
            L_inv = k2.arc_sort(L_inv)

        assert L_inv.requires_grad is False

        assert oov in self.lexicon.words

        self.L_inv = L_inv
        self.oov_id = self.lexicon.words[oov]
        self.oov = oov
        self.device = device

        phone_symbols = get_phone_symbols(self.lexicon.phones)
        phone_symbols_with_blank = [0] + phone_symbols

        ctc_topo = build_ctc_topo(phone_symbols_with_blank).to(device)
        assert ctc_topo.requires_grad is False

        self.ctc_topo_inv = k2.arc_sort(ctc_topo.invert_())

    def compile(self,
                texts: Iterable[str],
                P: k2.Fsa,
                replicate_den: bool = True) -> Tuple[k2.Fsa, k2.Fsa]:
        '''Create numerator and denominator graphs from transcripts
        and the bigram phone LM.

        Args:
          texts:
            A list of transcripts. Within a transcript, words are
            separated by spaces.
          P:
            The bigram phone LM created by :func:`create_bigram_phone_lm`.
          replicate_den:
            If True, the returned den_graph is replicated to match the number
            of FSAs in the returned num_graph; if False, the returned den_graph
            contains only a single FSA
        Returns:
          A tuple (num_graph, den_graph), where

            - `num_graph` is the numerator graph. It is an FsaVec with
              shape `(len(texts), None, None)`.

            - `den_graph` is the denominator graph. It is an FsaVec with the same
              shape of the `num_graph` if replicate_den is True; otherwise, it
              is an FsaVec containing only a single FSA.
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
        # num_graphs_with_self_loops[0].draw("linear_lex_rmeps_addslp.svg")

        num_graphs_with_self_loops = k2.arc_sort(num_graphs_with_self_loops)

        num = k2.compose(ctc_topo_P,
                         num_graphs_with_self_loops,
                         treat_epsilons_specially=False)
        num = k2.arc_sort(num)
        # num[0].draw("num.svg")

        ctc_topo_P_vec = k2.create_fsa_vec([ctc_topo_P.detach()])
        if replicate_den:
            indexes = torch.zeros(len(texts),
                                  dtype=torch.int32,
                                  device=self.device)
            den = k2.index_fsa(ctc_topo_P_vec, indexes)
        else:
            den = ctc_topo_P_vec

        return num, den

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
                if word in self.lexicon.words:
                    word_ids.append(self.lexicon.words[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        fsa = k2.linear_fsa(word_ids_list, self.device)
        fsa = k2.add_epsilon_self_loops(fsa)
        assert fsa.device == self.device
        num_graphs = k2.intersect(self.L_inv,
                                  fsa,
                                  treat_epsilons_specially=False).invert_()
        num_graphs = k2.arc_sort(num_graphs)
        return num_graphs

    def compile_lookahead_numerators(self, word_fsa_vec, P):
        # Compile lexicon graph
        fsa = k2.add_epsilon_self_loops(word_fsa_vec)
        assert fsa.device == self.device
        num_graphs = k2.intersect(self.L_inv,
                                  fsa,
                                  treat_epsilons_specially=False).invert_()
        num_graphs = k2.arc_sort(num_graphs)

        # Compile ctc_topo_P
        assert P.device == self.device
        P_with_self_loops = k2.add_epsilon_self_loops(P)

        ctc_topo_P = k2.intersect(self.ctc_topo_inv,
                                  P_with_self_loops,
                                  treat_epsilons_specially=False).invert()

        ctc_topo_P = k2.arc_sort(ctc_topo_P)

        # Combine
        num_graphs_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            num_graphs)

        num_graphs_with_self_loops = k2.arc_sort(num_graphs_with_self_loops)

        num = k2.compose(ctc_topo_P,
                         num_graphs_with_self_loops,
                         treat_epsilons_specially=False)
        num = k2.arc_sort(num)
        return num

    """
    def build_word_fsa(self, prefix, candidate_intervals, drop_prefix_tail):
        # convert prefix_ids in BPE domain to word sequence.
        if '' in prefix:
            prefix.remove('')

        prefix_ids = [self.lexicon.words[word] if word in self.lexicon.words else self.oov_id 
                      for word in prefix]

        # a special token that does not start with '_' could also be proposed in first iteration
        # they requires 'drop_tail' but there is no tail to drop
        # in this case, disable the 'drop_tail' operation
        batch = len(candidate_intervals)  
        drop_prefix_tail = [0] * batch if prefix == [] else drop_prefix_tail
 
        # Prefix part 
        prefix_len = len(prefix_ids)
        start_state = np.arange(prefix_len)
        end_state = np.arange(prefix_len) + 1
        labels = np.array(prefix_ids)
        scores = np.zeros(prefix_len)
        prefix_part = np.stack([start_state, end_state, labels, scores], axis=1)
        
        # candidate part
        candidate_parts = []
        ending_parts = []
        for (start, end), drop_tail in zip(candidate_intervals, drop_prefix_tail): 
            num_candidate = end - start
            start_state = np.ones(num_candidate) * (prefix_len - drop_tail)
            end_state = np.ones(num_candidate) * (prefix_len + 1 - drop_tail)
            labels = np.arange(start, end)
            scores = np.zeros(num_candidate)
            candidate_part = np.stack([start_state, end_state, labels, scores], axis=1)
            candidate_parts.append(candidate_part)

            # end arc
            end_arc = np.array([[prefix_len + 1 - drop_tail, prefix_len + 2 - drop_tail, -1, 0]])
            ending_parts.append(end_arc)
         
       
        # assemble: do not need to arc_sort 
        num_vec = []
        for i, (candidate_part, drop_tail) in enumerate(zip(candidate_parts, drop_prefix_tail)):
            this_prefix_part = prefix_part[:-1] if drop_tail else prefix_part
            end_arc = ending_parts[i]
            num_mat = np.concatenate([this_prefix_part, candidate_part, end_arc], axis=0)
            num_mat = torch.from_numpy(num_mat).to(torch.int32)
            num_vec.append(num_mat)
       
        # convert to k2 FsaVec 
        num_vec = [k2.Fsa.from_dict({"arcs": num}) for num in num_vec]
        num_vec = k2.create_fsa_vec(num_vec)
        return num_vec    
    """
