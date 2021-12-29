import numpy as np
import sys

def main():
    npy_file = sys.argv[1]
    symbol_table = sys.argv[2]
    
    print(f"Parsing file {npy_file}")

    log_probs = np.load(npy_file)
    probs = np.exp(log_probs) 

    syms = {}
    for line in open(symbol_table):
        ph, pid = line.split()
        syms[int(pid)] = ph

    max_probs = np.argmax(probs, axis=-1)
    max_syms = [syms[x] for x in max_probs]
    max_syms = " ".join(max_syms)
    print(max_syms)

main()
