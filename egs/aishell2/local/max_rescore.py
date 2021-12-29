import sys
import json
import codecs
import copy

json_f = sys.argv[1]
json_f_out = sys.argv[2]
best_dict_f = sys.argv[3]

with codecs.open(json_f, "r", encoding="utf-8") as f:
        j = json.load(f)

best_dict = {}
for name in j["utts"]:
    hyp_lst = j["utts"][name]["output"]
    for idx, hyp in enumerate(hyp_lst):
        if hyp["text"] == hyp["rec_text"].replace("<eos>", "") and idx > 0:
            best_dict[name] = copy.deepcopy([hyp_lst[0]] + [hyp_lst[idx]]) 
            print(f"{name}: {idx}-th is the best")
            if hyp_lst[0]["mmi_tot_score"] - hyp_lst[idx]["mmi_tot_score"] <  - 1e-5:
                print("May be corrected by MMI")
            


            hyp_lst = [hyp]
    j["utts"][name]["output"] = hyp_lst[:1]

with open(json_f_out, "wb") as f:
    f.write(
        json.dumps(
            j, indent=4, ensure_ascii=False, sort_keys=True
        ).encode("utf_8")
    )

with open(best_dict_f, "wb") as f:
    f.write(
        json.dumps(
            best_dict, indent=4, ensure_ascii=False, sort_keys=True
        ).encode("utf_8")
    )
