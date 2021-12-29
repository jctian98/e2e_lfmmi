import sys

text = sys.argv[1]
vocab_f = sys.argv[2]

vocab = {} # use hush
for line in open(vocab_f):
    w = line.strip().split()[0]
    vocab[w] = 0

tot_tok = 0
oov_tok = 0

for line in open(text):
    elems = line.strip().split()
    uttid = elems[0]
    toks = elems[1:]

    # count oov
    is_oov = [int(not tok in vocab) for tok in toks]
    tot_tok += len(is_oov)
    oov_tok += sum(is_oov)

oov_ratio = float(oov_tok) / float(tot_tok)
print(f"Total tokens: {tot_tok}, Tot oov tokens: {oov_tok}, oov ratio: {oov_ratio}")
