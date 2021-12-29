# author: tyriontian
# tyriontian@tencent.com

# this script is to change the root dir of some data files, like dump directory 
# in espnet format. By change the root, we can transplant the data into other 
# experiments

# e.g., python3 espnet_utils/change_root.py /mnt/ceph_asr_ts/tomasyu/jinchuan/las_mmi/ /apdcephfs/share_1149801/speech_user/tomasyu/jinchuan/lasmmi/ dump/dev_test/ ".json,.scp"
import sys
import os

org_pref = sys.argv[1]
dst_pref = sys.argv[2]
root = sys.argv[3]
suffix = sys.argv[4]

if org_pref[-1] != '/' or dst_pref[-1] != '/':
    raise ValueError("path should end with /")

suffix = suffix.strip().split(",")
print(f"Working under the directory: {root}")
print(f"Change the prefix in all files that end with {suffix}")
print(f"The initial prefix is {org_pref}; The destination prefix is {dst_pref}")

# BFS search: find all files to change root
queue = [root]
flist = []
while queue:
    cur_dir = queue.pop(0)
    for d in os.listdir(cur_dir):
        d = os.path.join(cur_dir, d)
        
        if os.path.isfile(d):
            for s in suffix:
                if d.endswith(s):
                    flist.append(d)
                    print(f"File to change: {d}")

        if os.path.isdir(d):
            queue.append(d)

# process these files one by one
for f in flist:
    handle = open(f, 'r+', encoding="utf-8")
    context = handle.readlines()
    handle.seek(0)
    handle.truncate(0)
    for line in context:
        line = line.replace(org_pref, dst_pref)
        handle.write(line)
    
