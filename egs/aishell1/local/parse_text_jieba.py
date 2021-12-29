import jieba
import sys

in_f = sys.argv[1]
out_f = sys.argv[2]
word_dict = sys.argv[3]

jieba.load_userdict(word_dict)

writer = open(out_f, 'w', encoding="utf8")
for line in open(in_f, encoding="utf8"):
    elems = line.split()
    uttid = elems[0]
    trans = "".join(elems[1:])
    trans_seg = " ".join(elems[1:])
    trans_seg_jieba = list(jieba.cut(trans, cut_all=False))
    trans_seg_jieba = " ".join(trans_seg_jieba)
    if not trans_seg_jieba == trans_seg:
        writer.write(f"Initail: {trans_seg} | jieba: {trans_seg_jieba}\n")
writer.close()
