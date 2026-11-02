# D9 attention analysis

See [stats.md](stats.md) for all prespecified A1–A4 tables and [stats.json](stats.json) for machine-readable values and scene counts.

Raw per-frame scalar summaries: frames_base.npz, frames_try5.npz, frames_kd.npz. Original attention dumps remain in the provenance directory.

Attention is descriptive, not causal. Map attention is upstream of graph–text interaction. No claim about training-time versus test-time assistance is made.

## Case assets

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_base_step0.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_base_step3.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_base_step4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_base_step14.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_try5_step0.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_try5_step0.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_try5_step0_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_try5_step3.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_try5_step3.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_try5_step3_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_try5_step4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_try5_step4.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_try5_step4_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_try5_step9.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_try5_step9.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_try5_step9_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_kd_step0.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_kd_step0.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_kd_step0_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_kd_step3.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_kd_step3.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_kd_step3_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_kd_step4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_kd_step4.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_kd_step4_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_kd_step9.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_kd_step9.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_map_kd_step9_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_aligned_step0.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_aligned_step3.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_aligned_step4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_aligned_step14.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/302/attn_txt_aligned_step9.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_base_step0.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_base_step7.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_base_step8.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_base_step10.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_try5_step0.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_try5_step0.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_try5_step0_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_try5_step7.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_try5_step7.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_try5_step7_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_try5_step8.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_try5_step8.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_try5_step8_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_try5_step14.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_try5_step14.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_try5_step14_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_kd_step0.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_kd_step0.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_kd_step0_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_kd_step7.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_kd_step7.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_kd_step7_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_kd_step8.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_kd_step8.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_kd_step8_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_kd_step14.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_kd_step14.npz`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_map_kd_step14_raw.png`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_aligned_step0.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_aligned_step7.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_aligned_step8.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_aligned_step10.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d9_case/1813/attn_txt_aligned_step14.json`

### Requested own-rollout frames not recorded

No absent frame is fabricated or substituted.

```json
[]
```


## Provenance

```json
{
  "dump_root": "/data/xukai/checkpoint/etp-r1-snapshot-data-logs/logs/d9_reference_layout",
  "draws": 2000,
  "seed": 0,
  "cpu_workers": 1,
  "script_commit": "514ca45a1b7c167b27c0598e0224b02761a7a2ca",
  "heads": {
    "base": {
      "graph_query_text": 12,
      "graph_map_attention": null
    },
    "try5": {
      "graph_query_text": 12,
      "graph_map_attention": 12
    },
    "kd": {
      "graph_query_text": 12,
      "graph_map_attention": 12
    }
  },
  "A4": "unpaired high-minus-low scene-cluster bootstrap; STOP query has no candidate; other uses agent union route in both groups",
  "registration": "assumed registration; upstream map attention before graph-text interaction",
  "patch_distance": "mean distance conditional on spatial-token mass (metadata excluded)",
  "special_flag": ">0.8; paired words use neither-flagged common frames",
  "head_spread": "NA if any head has zero word mass after float16 storage; summary n counts defined frames",
  "undefined": "NA, never zero"
}
```
