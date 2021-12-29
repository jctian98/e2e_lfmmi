import jieba
import sys
_ = jieba.lcut("aaa")


words_file = sys.argv[1]
dict_file = sys.argv[2]

reader = open(words_file, 'r')
writer = open(dict_file, 'w')
for line in reader:
    term = line.strip().split()[0]
    freq = jieba.dt.FREQ.get(term)
    freq = 1 if freq == None else freq
    writer.write(f"{term} {freq}\n")
writer.close()
