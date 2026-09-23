#!/usr/bin/env python3
import torch
from pathlib import Path
root=Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports')
for name in ('inputs.pt','map_tokens.pt','map_token_masks.pt','language.pt','global_logits.pt'):
 a=torch.load(root/'fwd_eq_A'/name,map_location='cpu'); b=torch.load(root/'fwd_eq_B'/name,map_location='cpu')
 if isinstance(a,dict):
  for k in sorted(a):
   x,y=a[k],b.get(k); print(name,k,tuple(x.shape), None if y is None else tuple(y.shape), y is not None and torch.equal(x,y), None if y is None or x.shape!=y.shape else float((x.float()-y.float()).abs().max()))
 else: print(name,tuple(a.shape),tuple(b.shape),torch.allclose(a,b,atol=1e-4,equal_nan=True),float((a.float()-b.float()).abs().nan_to_num().max()))
