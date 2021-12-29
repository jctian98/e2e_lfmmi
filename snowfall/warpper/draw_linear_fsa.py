import torch
import k2

fsa_lst = [[0, 1, 1, 0], 
           [1, 2, 2, 0], 
           [2, 3, 3, 0], 
           [2, 3, 4, 0], 
           [2, 3, 5, 0], 
           [3, 4, -1, 0]]
fsa_lst = torch.Tensor(fsa_lst).int()
print(fsa_lst)
num_vec = k2.Fsa.from_dict({"arcs": fsa_lst})
num_vec.aux_labels = ["a", "b", "c", "d", "e", "f"]
num_vec.draw("num.svg")
