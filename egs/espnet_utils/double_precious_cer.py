import sys

in_f = sys.argv[1]

for line in open(in_f, encoding="utf-8"):
    if "Sum" in line and "|" in line and "Avg" not in line:
        line = line.strip().split()
        tot = line[4]
        err = line[10]
        cer = float(err) / float(tot) * 100
        print("CER: {:.3f}".format(cer))
