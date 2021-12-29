import sys
import os

in_scp = sys.argv[1]
in_frame = sys.argv[2]
out_scp = sys.argv[3]
out_frame = sys.argv[4]

# read scp as dict
scp_dict = {}
for line in open(in_scp, encoding="utf-8"):
    uttid, add = line.strip().split()
    scp_dict[uttid] = add

# read utt2frames
frame_lst = []
for line in open(in_frame, encoding="utf-8"):
    uttid, length = line.strip().split()
    length = int(length)
    frame_lst.append([uttid, length])

frame_lst.sort(key=lambda x: x[1])

scp_writer = open(out_scp, 'w', encoding="utf-8")
frame_writer = open(out_frame, 'w', encoding='utf-8')

for e in frame_lst:
    uttid, length = e
    add = scp_dict[uttid]
    scp_writer.write(f"{uttid} {add}\n")
    frame_writer.write(f"{uttid} {length}\n")
scp_writer.close()
frame_writer.close()

"""
max_utt = 256
count = 1
out_scp_base = os.path.basename(out_scp)
out_scp_base = out_scp_base.replace(".", f".{count}.")
out_scp_dir = os.path.dirname(out_scp)
out_scp = os.path.join(out_scp_dir, out_scp_base)
for i, e in enumerate(frame_lst):
    if i % max_utt == 0:
       scp_writer = open(out_scp, 'w', encoding='utf-8')
       out_scp = out_scp.replace(f".{count}.", f".{count+1}.") 
       count += 1
    uttid, _ = e
    add = scp_dict[uttid]
    scp_writer.write(f"{uttid} {add}\n")
"""
