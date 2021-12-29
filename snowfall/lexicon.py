import re

from typing import List

import logging

import torch
from pathlib import Path

import k2


class Lexicon:
    def __init__(self, lang_dir: Path):
        self.lang_dir = lang_dir
        self.phones = k2.SymbolTable.from_file(self.lang_dir / 'phones.txt')
        self.words = k2.SymbolTable.from_file(self.lang_dir / 'words.txt')

        logging.info("Loading L.fst")
        if (self.lang_dir / 'Linv.pt').exists():
            L_inv = k2.Fsa.from_dict(torch.load(self.lang_dir / 'Linv.pt'))
        else:
            with open(self.lang_dir / 'L.fst.txt') as f:
                L = k2.Fsa.from_openfst(f.read(), acceptor=False)
                L_inv = k2.arc_sort(L.invert_())
                torch.save(L_inv.as_dict(), self.lang_dir / 'Linv.pt')
        self.L_inv = L_inv

    def phone_symbols(
            self,
            regex: str = re.compile(r'^#\d+$')
    ) -> List[int]:
        '''Return a list of phone IDs containing no disambiguation symbols.

        Caution:
          0 is not a phone ID so it is excluded from the return value.

        Args:
          regex:
            Symbols containing this pattern are disambiguation symbols.
        Returns:
          Return a list of symbol IDs excluding those from disambiguation symbols.
        '''
        symbols = self.phones.symbols
        ans = []
        for s in symbols:
            if not regex.match(s):
                ans.append(self.phones[s])
        if 0 in ans:
            ans.remove(0)
        ans.sort()
        return ans


