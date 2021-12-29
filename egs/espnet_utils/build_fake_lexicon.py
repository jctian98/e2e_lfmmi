import sys
import os

orig_lexicon = sys.argv[1]

for line in open(orig_lexicon, encoding="utf-8"):
    word = line.strip().split()[0]
    if word.startswith("<") or word == "SIL" or word == "sil":
        print(f"{word} {word}")
    else:
        out = [word] + list(word)
        out = " ".join(out)
        print(out)
        
