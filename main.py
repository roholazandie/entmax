import torch

from torch.nn.functional import softmax

from entmax import sparsemax, entmax15, entmax_bisect

x = torch.tensor([2, 1.5, 0.5])

s1 = softmax(x, dim=0)

print(s1)
s2 = sparsemax(x, dim=0)

print(s2)
s3 = entmax15(x, dim=0)

print(s3)

s4 = entmax_bisect(x, alpha=3)
print(s4)


# inp = x#torch.Tensor([-float('Inf'), 3 , 2, -float('Inf')])
# ss = sparsemax(inp, dim=-1)
# print(ss)
#
# from sparsemax2 import Sparsemax
#
# inp[inp == -float("Inf")] = -1e10
# sparsemax2 = Sparsemax(dim=-1)
# ss2 = sparsemax2(inp)
# print(ss2)