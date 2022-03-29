# Author: tyriontian
# tyriontian@tencent.com
# also see: https://github.com/k2-fsa/k2/issues/874

import os
import sys
import re
import torch
import k2
import copy
import fcntl
import time

from pathlib import Path
from espnet.nets.scorer_interface import PartialScorerInterface


def is_disambig_symbol(symbol: str, pattern: re.Pattern = re.compile(r'^#\d+$')) -> bool:
    return pattern.match(symbol) is not None

def find_first_disambig_symbol(symbols: k2.SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if is_disambig_symbol(k))

"""
Several reminders:
1. make sure '#0' is the last symbol in words.txt so '<s>' and '</s>' will not be 
   changed to epsilon in `G.labels[G.labels >= first_word_disambig_id] = 0`
2. There might be a bug that `k2.create_fsa_vec` will make G.fst unsorted. make sure
   you sort the G.fst as a FsaVec rather than a Fsa.
3. To be compatible with the back-off paths in G.fst, self-loops are added to the 
   raw lattices.
4. This module works on both CPU and GPU. it depends on the type of `device` you feed.
   We find that CPU is even faster than GPU when the test scale is small
5. Each time we load the G.fst from disk we need to call `k2.arc_sort` to make sure 
   the `properties` in it is correct. However, this would leads to a spike in memory
   use and will slow the initialization. Directly change the G.properties is dangerous
   and cannot solve this problem.
6. Reading the G.pt exclusively, which is ensured by the system file lock.
   Each time only one process would do the reading. This is to avoid memory spike. 
"""
class WordNgram():
    def __init__(self, lang, device):
        self.lang = Path(lang)
        self.device = device
        self.is_cuda = device.type == "cuda"

        self.symbol_table = k2.SymbolTable.from_file(self.lang / 'words.txt')
        self.oovid = int(open(self.lang / 'oov.int').read().strip())


        self.load_G()
        return
        # rapid access on the disk may lead to error
        # try many times
        for i in range(10):
            try:
                self.load_G()
                break
            except:
                print(f"{i}-th trial to load G.fst but failed")              

    def load_G(self):
        
        if os.path.exists(self.lang / 'G.pt'):
            f = open(self.lang / 'G.pt', 'r')
            fcntl.flock(f, fcntl.LOCK_EX) # lock
            G_dict = torch.load(self.lang / 'G.pt')
            fcntl.flock(f, fcntl.LOCK_UN) # unlock
            G = k2.Fsa.from_dict(G_dict).to(self.device)
            G = k2.create_fsa_vec([G])
            G = k2.arc_sort(G)
            print("Successfully load the cached G.pt", flush=True)
            
        else:
            f = open(self.lang / 'G.fst.txt') 
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            del G.aux_labels

            first_word_disambig_id = find_first_disambig_symbol(self.symbol_table)
            G.labels[G.labels >= first_word_disambig_id] = 0 
            G = k2.arc_sort(G)
            
            torch.save(G.as_dict(), self.lang / 'G.pt')
            print("No cached G.pt found. Build a new G from G.fst.txt")
        
        self.G = G.to(self.device)

    def text2lat(self, text, gram_len=6):
        """
        Enumreate all possible paths that split text into word sequence. 
        Output will be a epsilon-free, acyclic lattice on self.device
        
        text: list of token 
        ngram_len: the maximum length of each word gram, a.k.a, characters
        """
        text = [tok.replace(" ", "") for tok in text] # exclude blank
        
        if gram_len < 1:
            raise ValueError("invalid ngram_len. it should be larger than 1")

        arcs = []
        for s in range(len(text) + 1):
            for r in range(1, gram_len + 1):
                if len(text) - s - r < 0:
                    continue
                
                w = "".join(text[s: s + r])
                if w in self.symbol_table:
                    wid = self.symbol_table[w] 
                elif r == 1:
                    wid = self.oovid
                else:
                    continue
                
                arc = [s, s+r, wid, 0.0]
                arcs.append(arc)
        arcs.append([len(text), len(text) + 1, -1, 0.0])
        arcs.append([len(text) + 1])
      
        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [' '.join(arc) for arc in arcs]
        arcs = '\n'.join(arcs)

        lat = k2.Fsa.from_str(arcs, True)
        lat = k2.arc_sort(lat)
        return lat.to(self.device)

    def score_lattice(self, lats, log_semiring=True):
        """
        Apply the scores on G.fst to the lattice.
        Both the input and the output are k2.FsaVec
        """
        assert lats.device == self.device

        lats = k2.add_epsilon_self_loops(lats)
        if self.is_cuda:
            b_to_a_map = torch.zeros(lats.shape[0],
                         device=self.device, dtype=torch.int32)
            scored_lattice = k2.intersect_device(
                             self.G, lats, b_to_a_map,
                             sorted_match_a=True 
                             )
            scored_lattice = k2.top_sort(
                             k2.connect(
                             k2.remove_epsilon_self_loops(
                             scored_lattice
                             )))
        else:
            scored_lattice = k2.intersect(self.G, lats, 
                             treat_epsilons_specially=False)
            scored_lattice = k2.top_sort(
                             k2.connect(
                             k2.remove_epsilon_self_loops(scored_lattice
                             )))
        return scored_lattice

    def score_texts(self, texts, log_semiring=True):
        lats = [self.text2lat(t) for t in texts]
        lats = k2.create_fsa_vec(lats)
     
        scored_lattice = self.score_lattice(lats, log_semiring)
        
        scores = scored_lattice._get_tot_scores(log_semiring, True)
        
        return scores

    def draw(self, fsavec, prefix=None):
        for i in range(fsavec.shape[0]):
            fsa = fsavec[i]
            fsa.draw(f"{prefix}_{i}.svg")


# Warpper to Espnet scorer interface
class WordNgramPartialScorer(PartialScorerInterface):
    def __init__(self, lang, device, token_list, ignore_tokens=["<eos>", "<blank>"], log_semiring=True, lower_char=True):
        self.WordNgram = WordNgram(lang, device)
        self.log_semiring = log_semiring
        
        self.token_list = copy.deepcopy(token_list)
        for tok in ignore_tokens:
            if tok in self.token_list:
                idx = self.token_list.index(tok)
                self.token_list[idx] = ""
 
        # oov should be a single character
        if "<unk>" in self.token_list:
            idx = self.token_list.index("<unk>")
            self.token_list[idx] = "兲"

        # ignore upper / lower case of english characters.
        # this should be consistent with your words.txt
        for tok in self.token_list:
            if tok.isalpha():
                idx = self.token_list.index(tok)
                if lower_char:
                    self.token_list[idx] = tok.lower()
                else:
                    self.token_list[idx] = tok.upper()

    def init_state(self, x):
        return 0.0

    def score_partial(self, y, next_tokens, state, x):
        prefix = [self.token_list[i] for i in y] 
        next_tokens = [self.token_list[i] for i in next_tokens]
        texts = [prefix + [tok] for tok in next_tokens]

        scores = self.WordNgram.score_texts(texts, 
                     log_semiring=self.log_semiring)
        
        return scores - state, scores
    
if __name__ == "__main__":
    device = torch.device("cpu") # cpu or cuda:0
    lang = sys.argv[1]
    word_ngram = WordNgram(lang, device)

    texts = [["这", "件", "事", u"\u2581"+"INTEREST", "ING", u"\u2581"+"ALLOW" , "ED"]]
    print(texts)
    for i in range(1):
        scores = word_ngram.score_texts(texts, log_semiring=True)
        print(scores)
