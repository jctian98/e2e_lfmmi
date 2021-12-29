import jieba
import sys

src_file = sys.argv[1]
dst_file = sys.argv[2]
dict_file = sys.argv[3]
jieba.set_dictionary(dict_file)

reader = open(src_file, 'r')
writer = open(dst_file, 'w')

word_dict = {}
for line in open(dict_file):
    w = line.strip().split()[0]
    word_dict[w] = 0 

oov_count = 0
for i, line in enumerate(reader):
    elems = line.strip().split()
    uttid, ctx = elems[0], elems[1:]
    ctx = " ".join(ctx)
    ctx = jieba.lcut(ctx, HMM=False)
    #for x in ctx:
    #    if x not in word_dict:
    #        print(i, x, flush=True)
    ctx = " ".join(ctx)
    writer.write(f"{uttid} {ctx}\n")
    
writer.close()
