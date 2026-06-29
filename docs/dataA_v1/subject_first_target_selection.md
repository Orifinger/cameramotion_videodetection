# Data A v1 Subject-First Target Selection

This change rebuilds a new VACE execution plan by changing only which target
track is selected per source video. It does not rerun Qwen3-VL, SAM3, object
pairing, video packaging, or VACE inference.

Inputs:

- Existing SAM3 quality-pass track bank.
- Existing mask tube `.npz` files.
- Video metadata from ffprobe, cached once per `video_id`.
- Optional base execution plan for reusing case ids, operation, generator route,
  donor references, and quota order.

Outputs:

- `res/dataA_v1/catalogs/subject_first_target_catalog.json`
- `res/dataA_v1/audits/subject_first_selection_audit.json`
- `res/dataA_v1/audits/subject_first_selection_audit.csv`
- `res/dataA_v1/plans/frozen_subject_first_vace_execution_plan.json`

The historical `res/dataA_v1/plans/frozen_full_vace_execution_plan.json` is kept
unchanged for debug and smoke runs. The subject-first plan must be used with a
new run id so any currently running VACE job is not affected.

Build:

```bash
python scripts/dataa_v1/build_subject_first_execution_plan.py \
  --track-bank res/sam_track_bank/sam3_quality_tracks_enriched.json \
  --base-plan res/dataA_v1/plans/frozen_full_vace_execution_plan.json \
  --selection-config configs/dataa_v1/subject_selection_v1.json \
  --ffprobe-bin /input/tmp/ffmpeg/ffmpeg-7.0.2-amd64-static/ffprobe
```

Dry-run writes the catalog and audit files but skips writing the final plan:

```bash
python scripts/dataa_v1/build_subject_first_execution_plan.py \
  --track-bank res/sam_track_bank/sam3_quality_tracks_enriched.json \
  --base-plan res/dataA_v1/plans/frozen_full_vace_execution_plan.json \
  --dry-run
```

Run VACE with the new plan:

```bash
python scripts/dataa_v1/run_vace14b_batch.py \
  --config configs/dataa_v1/vace14b_subject_first.yaml \
  --track-bank res/sam_track_bank/sam3_quality_tracks_enriched.json \
  --checkpoint-dir /home/admin/wan2.1-VACE \
  --run-id dataa_v1_vace14b_subject_first_YYYYMMDD \
  --execute \
  --launch-workers
```

Manual offline sync:

Copy these files to the server repo at the same relative paths:

- `scripts/dataa_v1/subject_selection.py`
- `scripts/dataa_v1/build_subject_first_execution_plan.py`
- `configs/dataa_v1/subject_selection_v1.json`
- `configs/dataa_v1/vace14b_subject_first.yaml`
- `docs/dataA_v1/subject_first_target_selection.md`
- `tests/dataa_v1/test_subject_selection.py`

Then run the build command on the server where the track bank, mask tubes, and
videos are available.
