import sys

in_f = sys.argv[1]

for line in open(in_f, 'r', encoding="utf8"):
    elems = line.split()
    uttid = elems[0]
    for sp in ["0.9", "1.0", "1.1"]:
        uttid_sp = f"sp{sp}-{uttid}"
        line = f"{uttid_sp} " + " ".join(elems[1:])
        print(line)
