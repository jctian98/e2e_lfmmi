import sys
import os

in_f = sys.argv[1]

max_utt=360 # So batch size could be 1, 2, 3, 4, 5, 6, 8, 10, 12 etc.
count = 1

out_scp_base = os.path.basename(in_f)
out_scp_base = out_scp_base.replace(".", f".{count}.")
out_scp_dir = os.path.dirname(in_f)
out_scp = os.path.join(out_scp_dir, out_scp_base)
for i, line in enumerate(open(in_f, encoding='utf-8')):
    if i % max_utt == 0:
        scp_writer = open(out_scp, 'w', encoding="utf-8")
        out_scp = out_scp.replace(f".{count}.", f".{count+1}.") 
        count += 1
    scp_writer.write(line)
    
