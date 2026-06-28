# VACE Upstream

- upstream repository: ali-vilab/VACE
- pinned upstream commit: 48eb44f1c4be87cc65a98bff985a26976841e9f3
- license: Apache-2.0
- retrieval date: 2026-06-29

## Project Policy

- Do not modify upstream core model logic in `third_party/VACE`.
- Data A v1 calls VACE through wrappers in `scripts/dataa_v1`.
- Wan2.1 is not copied into this repository; it is installed on the server as an offline Python dependency.
- VACE checkpoints, Wan weights, generated videos, mask NPZ files, upload staging, wheel caches and attempt outputs must not be committed.
