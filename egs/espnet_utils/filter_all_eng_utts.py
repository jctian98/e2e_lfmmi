import sys

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

f_in = sys.argv[1]
f_out = sys.argv[2]

writer = open(f_out, 'w', encoding='utf-8')
for line in open(f_in, encoding='utf-8'):
    elems = line.strip().split()
    uttid = elems[0]
    
    if len(elems) <= 1:
        continue

    text = " ".join(elems[1:])
    if not is_all_chinese(text):
        continue

    out_line = " ".join([uttid, text, '\n'])
    writer.write(out_line)
