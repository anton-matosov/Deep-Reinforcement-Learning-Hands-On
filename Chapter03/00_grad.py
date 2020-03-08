#!/usr/bin/env python3

import torch

v1 = torch.tensor([1., 1.], requires_grad=True)
print('v1', v1, v1.grad, v1.is_leaf, v1.requires_grad)

v2 = torch.tensor([2., 2.], requires_grad=True)
print('v2', v2, v2.grad, v2.is_leaf, v2.requires_grad)

v3 = torch.tensor([3., 3.], requires_grad=True)
print('v3', v3, v3.grad, v3.is_leaf, v3.requires_grad)

for i in range(5):
  print(f'###### {i} ######')
  v_23 = v2 + v3
  print('v_23', v_23, v_23.grad, v_23.is_leaf, v_23.requires_grad)

  v_sum = v1 + v2
  print('sum', v_sum, v_sum.grad, v_sum.is_leaf, v_sum.requires_grad)

  v_res = ((v_sum * 4.) + v3).sum()
  print('res', v_res, v_res.grad, v_res.is_leaf, v_res.requires_grad)

  v_res.backward()
  print('v1 grad', v1.grad)
  print('v2 grad', v2.grad)
  print('v3 grad', v3.grad)
  print('sum grad', v_sum.grad)
  print('res grad', v_res.grad)
