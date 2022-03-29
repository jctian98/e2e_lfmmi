import sys

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

in_f, out_f = sys.argv[1:]

lines = open(in_f, encoding="utf-8").readlines()
seperator = u'\u2581'
writer = open(out_f, 'w', encoding="utf-8")

for line in lines:
    line = line.strip().split()
    ans = []
    for p in line[1:]: # remove uttid
        if is_all_chinese(p):
            ans.append(p)
        else:
            ans.append(seperator + p)
    line = " ".join(ans) + '\n'
    writer.write(line)
writer.close()
