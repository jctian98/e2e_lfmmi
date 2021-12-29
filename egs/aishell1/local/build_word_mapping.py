# convert the attention output vocabulary into lexicon vocabulary
import sys

att_vocab = sys.argv[1]
lex_vocab = sys.argv[2]
out_map = sys.argv[3]

# load lex_vocab
lex = {}
for line in open(lex_vocab, encoding='utf8'):
    tok, tid = line.split()
    lex[tok] = tid

writer = open(out_map, 'w', encoding='utf8')
for line in open(att_vocab, encoding='utf8'):
    tok, tid = line.split()
    if tok in lex.keys():
        info = "{} {}\n".format(tid, lex[tok])
        writer.write(info)
    else:
        print("CANNOT find ", tok)
