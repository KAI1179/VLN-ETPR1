#!/usr/bin/env python3
"""Minimal fp32 forward probe for the 121c369 and current PriorGT trees."""
import argparse, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--tree', choices=('A', 'B'), required=True)
parser.add_argument('--out-dir', required=True, type=Path)
args = parser.parse_args()
args.out_dir = args.out_dir.resolve()
roots = {'A': '/home/xukai/code/ETP-R1-snapshot/.worktrees/etpr1-121c369',
         'B': '/home/xukai/code/ETP-R1-snapshot/ETP-R1'}
root = roots[args.tree]
os.chdir(root); sys.path.insert(0, root)
import numpy as np
import torch
from gym import spaces
from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.etp_prior_gt import policy as prior_policy

ckpt = '/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth'
namespace = 'gt.online121c369.r1p5.direction5.v1'
opts = ['TORCH_GPU_ID','0','MODEL.MAP_ENCODER.enabled','True']
if args.tree == 'B':
    opts += ['MODEL.policy_name','PriorGTTry5Policy','MODEL.MAP_ENCODER.architecture','try5','MODEL.MAP_ENCODER.source','prior_gt','MODEL.MAP_ENCODER.cache_namespace',namespace,'MODEL.MAP_ENCODER.refiner_ckpt','']
else:
    opts += ['MODEL.policy_name','PriorGTPolicy','MODEL.MAP_ENCODER.radius_m','1.5']
config = get_config('run_r2r/iter_train.yaml', opts)
config.defrost(); config.MODEL.pretrained_path = ckpt; config.freeze()
obs_spaces = {
    'rgb': spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
    'depth': spaces.Box(0, 1, (256, 256, 1), dtype=np.float32),
    'instruction': spaces.Box(0, 250000, (200,), dtype=np.int64),
}
for heading in range(30, 360, 30):
    obs_spaces[f'rgb_{heading}'] = spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8)
    obs_spaces[f'depth_{heading}'] = spaces.Box(0, 1, (256, 256, 1), dtype=np.float32)
obs = spaces.Dict(obs_spaces)
policy_cls = prior_policy.PriorGTPolicy if args.tree == 'A' else prior_policy.PriorGTTry5Policy
policy = policy_cls.from_config(config, obs, spaces.Discrete(4))
net = policy.net.cuda().eval()
state = torch.load(ckpt, map_location='cpu')['state_dict']
state = {(k[len('net.module.'):] if k.startswith('net.module.') else k): v for k,v in state.items()}
net.load_state_dict(state, strict=True)

ids = [53, 992, 444, 1352, 1685]
scenes = {53:'zsNo4HB9uLZ', 992:'TbHJrupSAjP', 444:'EU6Fwq7SyZv', 1352:'QUCTc6BB5sX', 1685:'zsNo4HB9uLZ'}
raw = []
for eid in ids:
    p = Path('/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps') / namespace / 'raster' / scenes[eid] / f'R2R_val_unseen_{eid}.npz'
    with np.load(p, allow_pickle=True) as x:
        raw.append({k: torch.from_numpy(x[k]).float() for k in ('grid','direction_vectors','start_direction_vector','start_position')})
grid = torch.stack([x['grid'] for x in raw]).cuda()
dirs = torch.stack([x['direction_vectors'] for x in raw]).cuda()
sdirs = torch.stack([x['start_direction_vector'] for x in raw]).cuda()
spos = torch.stack([x['start_position'] for x in raw]).cuda()
with torch.no_grad():
    if args.tree == 'A':
        mt, mm = net(mode='map_encoding', cognitive_crops=grid, direction_vectors=dirs, start_direction_vectors=sdirs, start_positions=spos)
    else:
        mt, mm = net(mode='map_encoding', cognitive_crops=grid, trajectory_keypoints=dirs, start_direction_vectors=sdirs, start_positions=spos)
    torch.manual_seed(0)
    txt_ids = torch.randint(0, 250000, (1,40), device='cuda')
    txt_masks = torch.ones(1,40, dtype=torch.bool, device='cuda')
    txt_task_encoding = torch.zeros(1,40, dtype=torch.long, device='cuda')
    txt = net(mode='language', txt_ids=txt_ids, txt_masks=txt_masks, txt_task_encoding=txt_task_encoding)
    torch.manual_seed(1)
    g=8; d=mt.shape[-1]
    nav = dict(gmap_vp_ids=[[None,'0','1','2','g0','g1','g2','g3']],
        gmap_img_fts=torch.randn(1,g,d,device='cuda'), gmap_pos_fts=torch.randn(1,g,7,device='cuda'),
        gmap_step_ids=torch.zeros(1,g,dtype=torch.long,device='cuda'), gmap_masks=torch.ones(1,g,dtype=torch.bool,device='cuda'),
        gmap_visited_masks=torch.zeros(1,g,dtype=torch.bool,device='cuda'),
        gmap_pair_dists=torch.randn(1,g,g,device='cuda').abs(),
        gmap_task_embeddings=torch.zeros(1,g,dtype=torch.long,device='cuda'))
    out = net(mode='navigation', txt_embeds=txt, txt_masks=txt_masks, map_tokens=mt[:1], map_token_masks=mm[:1], **nav)
args.out_dir.mkdir(parents=True, exist_ok=True)
torch.save({'grid':grid.cpu(),'direction_vectors':dirs.cpu(),'start_direction_vector':sdirs.cpu(),'start_position':spos.cpu(),'txt_ids':txt_ids.cpu(),'txt_masks':txt_masks.cpu(),'txt_task_encoding':txt_task_encoding.cpu(), **{k:v.cpu() for k,v in nav.items() if torch.is_tensor(v)}}, args.out_dir/'inputs.pt')
torch.save(mt.cpu(),args.out_dir/'map_tokens.pt'); torch.save(mm.cpu(),args.out_dir/'map_token_masks.pt'); torch.save(txt.cpu(),args.out_dir/'language.pt'); torch.save(out['global_logits'].cpu(),args.out_dir/'global_logits.pt')
print(args.tree, tuple(mt.shape), tuple(txt.shape), tuple(out['global_logits'].shape))
