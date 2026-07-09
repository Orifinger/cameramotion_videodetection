# Camera-Compensated Residual Route

Date: 2026-07-09

This note replaces the weak "camera text + detection CoT" route as the main
camera-conditioned method candidate. It does not modify the old final plan.

## Why the Previous Route Is Weak

The old camera-conditioned SFT path only concatenates camera labels/captions
with the existing detection reasoning. That is risky because the model can
learn a text pattern instead of learning how camera motion changes local
artifact evidence. A rule sentence such as "this anomaly cannot be explained by
camera motion" is also a form of supervision pollution if it is applied
uniformly.

## Data Fact

The local DataA files show a stronger structure:

- Detection JSON contains 1076 real/fake pairs.
- Camera JSONL covers 1067 of those pairs.
- Within each covered pair, the real and fake videos have identical camera
  labels/captions.
- Within each detection pair, the real and fake samples share the same
  `<t>` and `<bbox>`.
- Fake samples label that shared region as an artifact; real samples describe
  the same region as clean/consistent.

This means camera motion is not a real/fake shortcut in DataA. The useful
signal is:

```text
same scene + same camera motion + same time/window region
  real: camera-consistent normal evidence
  fake: local edit artifact evidence
```

## Main Hypothesis

If global camera motion is compensated, local edit artifacts should leave a
stronger residual inside the annotated region than the matched real video.

```text
video frames
  -> estimate global camera motion
  -> warp adjacent frames
  -> compute local residual in <t>/<bbox>
  -> train/validate local artifact evidence
```

This connects camera motion to AIGC detection through a visual intermediate
signal, not through a template.

## Gate 0: Training-Free Probe

Run the residual probe before training:

```bash
python tools/dataa_camera_compensated_residual_probe.py \
  --detection-json /input/workflow_58770161/workspace/test/cameramotion_det/detection/dataa_vace_grounded_cot_instruct_tp8x2_sft_all.json \
  --camera-jsonl /input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_v2.jsonl \
  --out-summary /tmp/1res/dataa_residual_probe_summary_200.json \
  --out-csv /tmp/1res/dataa_residual_probe_rows_200.csv \
  --max-pairs 200
```

By default, the probe masks the target `<bbox>` with 10% padding when estimating
homography, then computes residual inside the target region. This prevents the
edited region from contaminating the global camera-motion estimate. Use
`--include-bbox-in-homography` only as a rough diagnostic ablation.

If server frame paths differ from the JSON paths, add:

```bash
--old-prefix /tmp/cameramotion_det \
--new-prefix /actual/frame/root
```

Treat the probe as valid only if enough pairs are successfully scored:

```text
ok_pairs >= 100 for the 200-pair pilot
or
ok_pairs / total_pairs >= 0.70 for larger runs
```

Continue only if the valid scored subset also satisfies one of the following:

```text
AUC(fake residual vs real residual) >= 0.60
or
P(fake_score > real_score) >= 0.60
```

If this gate fails, camera-compensated local evidence is not strong enough for
the current DataA setup, and the camera method should not be the main line.

## Training Route If Gate 0 Passes

### Stage 1: Camera Motion Pretext

Train the detector checkpoint to output compact camera motion labels from
frames.

Training data:

```text
CameraBench official cam_motion
DataA camera labels/captions
DataB camera pseudo labels/captions
```

Target format:

```xml
<camera_motion>pan-right; regular-speed; no-shaking</camera_motion>
```

This stage injects camera motion perception, but it is not claimed as the final
detector improvement by itself.

### Stage 2: Camera-Compensated Region Pretext

Use DataA pair structure. For each pair, use the same `<t>/<bbox>` for real and
fake. The model sees full frames plus residual/crop evidence and predicts:

```xml
<camera_motion>...</camera_motion>
<region_status>normal</region_status>
```

or

```xml
<camera_motion>...</camera_motion>
<region_status>artifact</region_status>
```

Labels come from the pair relation, not from generated long reasoning.

Build a smoke dataset with:

```bash
python tools/build_dataa_pair_region_pretext.py \
  --detection-json /input/workflow_58770161/workspace/test/cameramotion_det/detection/dataa_vace_grounded_cot_instruct_tp8x2_sft_all.json \
  --camera-jsonl /input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_v2.jsonl \
  --out /tmp/1res/dataa_pair_region_pretext_smoke.json \
  --task both \
  --max-pairs 200 \
  --pair-max-frames-per-video 8
```

The pair-selection task uses 8 frames per video by default, so each A/B record
has 16 images. It prioritizes frames inside the target time window.

### Stage 3: Pair Preference

Use preference training only with hard negatives derived from the pair:

```text
chosen: correct real/fake region status and final answer
rejected: swapped real/fake region status, wrong bbox/time, or wrong answer
```

Avoid long generated CoT pairs. The purpose is to prefer correct local evidence
under matched camera motion.

### Stage 4: Detection Retention

Return to final detection format and mix DataB replay heavily:

```text
DataB detection replay : DataA final detection : DataA region replay = 6 : 2 : 1
```

Final inference remains single-model:

```xml
<camera_motion>...</camera_motion>
<think>...<type>...</type> in <t>...</t> at <bbox>...</bbox></think>
<answer>Fake</answer>
```

No external gold camera context is used at inference time.

## Required Ablations

Use these to avoid overclaiming:

```text
DataB detector baseline
old camera-text SFT baseline
camera pretext only
paired-region without residual
paired-region with camera-compensated residual
ours without camera pretext
```

The main claim is valid only if residual/pair training improves DataA local
evidence metrics while keeping VIF-Bench close to the DataB detector.

## Minimal Claim

```text
Camera-compensated paired-region supervision improves local-edit AIGC video
detection by turning camera motion from a text condition into a visual
residual signal for grounded artifact evidence.
```

