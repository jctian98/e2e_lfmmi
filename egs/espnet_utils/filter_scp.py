import sys

ref_f = sys.argv[1]
in_f = sys.argv[2]

# output is in the order of ref_f
ref = []
for line in open(ref_f, encoding='utf-8'):
    uttid = line.strip().split()[0]
    ref.append(uttid)

in_dic = {}
for line in open(in_f, encoding='utf-8'):
    elems = line.strip().split()
    uttid = elems[0]
    ctx = " ".join(elems[1:])
    in_dic[uttid] = ctx

for e in ref:
    if e in in_dic:
        print(f"{e} {in_dic[e]}")
