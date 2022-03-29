import sys

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

in_f, chn_f, eng_f = sys.argv[1:4]
chn_writer = open(chn_f, 'w', encoding="utf-8")
eng_writer = open(eng_f, 'w', encoding="utf-8")

for line in open(in_f, encoding="utf-8"):
    elems = line.strip().split()
    uttid = elems[-1]
    chn_buf, eng_buf = [], []
    for c in elems[:-1]:
        if is_all_chinese(c):
            chn_buf.append(c)
        else:
            eng_buf.append(c)
    
    chn_str = " ".join(chn_buf + [uttid]) + "\n"
    eng_str = " ".join(eng_buf + [uttid]) + "\n"

    chn_writer.write(chn_str)
    eng_writer.write(eng_str)

chn_writer.close()
eng_writer.close()
