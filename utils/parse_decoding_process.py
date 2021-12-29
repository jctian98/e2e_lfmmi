import os
import numpy as np
import matplotlib.pyplot as plt


def plot_decoding_logs(graph_dir, char_list, recog_args, uttid, nbest_hyps):
    graph_subdir = os.path.join(graph_dir, uttid)
    os.makedirs(graph_subdir, exist_ok=True)

    for i, hyp in enumerate(nbest_hyps):
        hyp_chr = "".join([char_list[int(x)] for x in hyp["yseq"][1:]]).replace("<space>", " ")
        print(f"{i}-th hypothesis of {uttid}: {hyp_chr}")

        logs = hyp["logs"]
        step_logs = process_logs(logs, recog_args, accum=False)
        tot_logs = process_logs(logs, recog_args, accum=True)

        filename = f"{uttid}-{i}-step.png"
        filename = os.path.join(graph_subdir, filename)
        plot_dict(filename, step_logs, hyp_chr)

        filename = f"{uttid}-{i}-tot.png"
        filename = os.path.join(graph_subdir, filename)
        plot_dict(filename, tot_logs, hyp_chr) 
           
def plot_dict(filename, d, title):
    
    plt.clf()
    plt.cla() 
    lines = []
    for k, v in d.items():
        x = np.arange(len(v)) 
        line = plt.plot(x, v, label=k)
        lines.append(line)
    plt.legend()
    plt.title(title)
    plt.savefig(filename)

def process_logs(logs, args, accum=False):
    ans = {}
    
    for k, v in logs.items():
        
        if accum:
            v = [sum(v[:l+1]) for l in range(len(v))]
        
        v = np.array(v)
        if k == "att":
            v = v * (1 - args.ctc_weight)
        elif k == "ctc":
            v = v * args.ctc_weight
        elif k == "mmi":
            v = v * args.mmi_weight
        elif k == "lm":
            v = v * args.lm_weight

        ans[k] = v
    
    tot = np.zeros_like(ans["att"])
    for k, v in ans.items():
        tot += v
    ans["sum"] = tot

    return ans
        
