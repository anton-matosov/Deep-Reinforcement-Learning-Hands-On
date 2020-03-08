#!/usr/bin/env python3

import torch
import torch.cuda

# print(torch.tensor(5))
# print(torch.tensor([1, 2, 3, 4]))

# z = torch.zeros(5, 3, 4)
# print(z)
# print(torch.ones_like(z))

# t = torch.ones(5, 3)
# print(t)

# t[1][2] = 5
# print(t)

t = torch.ones(2, 3, 4, device='cpu')
print(t)
print(t.transpose(0, 2))

print(torch.cuda.is_available())
# print(torch.cuda.is_initialized())
# print(torch.cuda.get_device_capability())

if torch.cuda.is_available():
  t.to('cuda:0')
else:
  t = t.to('cpu')
