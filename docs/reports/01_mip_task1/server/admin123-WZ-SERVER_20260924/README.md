# Server run — admin123-WZ-SERVER, 2026-09-24

`task1_report.md` was produced by `tools/01_mip_task1/run_task1.sh` on the user's server and handed
back by upload (not by `collect.sh`, so no logs / summary.json here). The first A0–A4 block is an
interrupted first run; the second block (from 01:27) is the complete run.

Headline: A1–A4 PASS, A5 PARTIAL (mp3d at /data/xukai/mp3d, 90 scans; connectivity at
/data/xukai/VLN-GOAT/datasets/R2R/connectivity; no R2R_VLNCE_v1-3_preprocessed full split),
A6 PASS, **A7-1 and A7-2 PASS** (simulator + data + tools wired; scripted agent and the mini
harness on the fake endpoint both complete episode 0), A7-3 / C2 gated (no model chosen yet),
B1–B4 identical to the cloud-container results, C1 no provider keys (codex CLI 0.153.2 present,
no claude CLI), C3 8×RTX 4090.
