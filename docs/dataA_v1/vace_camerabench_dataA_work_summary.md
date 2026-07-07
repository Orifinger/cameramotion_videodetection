# Data A v1 VACE 局部编辑数据构建工作汇总

更新时间：2026-07-05

本文记录当前在 `Orifinger/cameramotion_videodetection` 中围绕 CameraBench 训练集构建 Data A v1 的工作：用 Qwen3-VL + SAM3 从 CameraBench 视频中提取可编辑物体/区域，构造 target/donor 配对或 text-edit reserve，再用 VACE 做局部视频编辑，最终保留同一原视频对应的 `full_real.mp4` 与 `full_fake.mp4`，用于后续 AIGC 视频检测训练与评估。

## 1. 数据目标

Data A v1 的样本单位是完整视频级 Real/Fake pair：

```text
Real_A = CameraBench target 视频的原始完整视频规范化副本
Fake_A = 同一个 target 视频中，指定物体/区域生命周期片段经 VACE 局部编辑后再拼回的完整视频
donor B = 仅作为 reference condition，不是 Real/Fake 配对对象，也不能把 RGB 直接贴到 target
```

这里的核心目的是保持 CameraBench 原始视频的相机运动、场景结构和非编辑区域尽量不变，只在可追溯的局部 mask tube 内制造 AIGC 编辑痕迹。后续训练时，CameraBench 自带的 camera motion caption 和 labels 可以作为每个 Real/Fake pair 的运动语义上下文。

当前只使用 CameraBench 训练集。验证集/测试集是否加入，需要后续单独确定，避免污染评测。

## 2. CameraBench 元数据对接

CameraBench 论文和官方仓库提供的是相机运动理解 benchmark。我们当前关心的字段包括：

```text
video_id / relative_path / split
camera motion caption
camera motion labels
可能存在的 scene/video caption
原始视频路径
```

这些字段不参与 VACE 生成本身，但必须进入最终索引或 manifest，原因是：

- Real/Fake 必须能回溯到 CameraBench 原视频；
- AIGC fake 的局部编辑不能覆盖相机运动标签的来源；
- 后续检测任务可以按 camera motion caption/label 分层统计性能；
- 如果某类相机运动下 fake 特别容易或特别难检测，需要能按标签回查。

当前建议在最终汇总索引中保留：

```json
{
  "camera_bench": {
    "split": "train",
    "video_id": "...",
    "relative_path": "...",
    "camera_motion_caption": "...",
    "camera_motion_labels": ["..."],
    "source_metadata_path": "..."
  }
}
```

注意：本仓库当前主要实现了 Data A 生成链路，CameraBench 官方论文/仓库 URL 需要在最终发布说明中补齐并确认版本，避免引用错误。

## 3. 当前总体流程

当前 Data A v1 生成链路分成六段：

```text
CameraBench train videos
-> Qwen3-VL inventory v2
-> taxonomy / compatibility normalization
-> SAM3 mask extraction
-> pairing dataset materialization
-> frozen VACE execution plan
-> VACE local editing
-> full_real.mp4 / full_fake.mp4 + manifest + OSS
```

其中 Qwen3-VL 负责识别视频里有哪些可编辑物体/区域，SAM3 负责生成跨帧 mask tube，pairing 阶段负责确定 target、donor、operation 和 mask policy，VACE 只执行已经冻结的 plan。

不允许在 package/runtime 阶段临时重选 target、重选 donor 或脏 fallback。

## 4. Qwen3-VL inventory v2

入口脚本：

```text
scripts/run_qwen_object_proposals.py
```

当前推荐的 inventory 路线是先让 Qwen3-VL 对每个视频做对象清点，而不是直接让它只给少量 SAM3 prompt。inventory 的作用是：

- 尽量记录视频中所有明显人物、动物、车辆、物体、屏幕、海报、文字载体等；
- 标注前景/主体/大小/可编辑性；
- 给出 `sam3_prompt_phrase`；
- 归一化到 taxonomy label，方便后续 pair；
- 区分真实人物、卡通人物、3D 角色、正面/侧面/背面等，减少不匹配 donor。

主要产物：

```text
res/qwen_inventory_v2/qwen_inventory_entities.json
res/qwen_inventory_v2/qwen_sam3_candidates_inventory_v2.json
```

`qwen_inventory_entities.json` 是后续 pairing 当前使用的主要 inventory 文件。之前曾出现把 normalized 文件直接喂给 pairing 后 `missing_compatibility_group` 很多的问题，因此当前 pairing 命令应使用 `qwen_inventory_entities.json`。

## 5. Taxonomy 与 compatibility

配置文件：

```text
configs/dataa_v1/taxonomy_v2_seed.json
configs/dataa_v1/compatibility_matrix_v2.json
```

taxonomy 的目的不是做通用分类，而是服务于可控编辑配对。当前重点：

- 人物优先：真实人物、卡通人物、3D 角色、雕像/模型要区分；
- 交通工具不能只粗暴按 vehicle 配对，例如飞机和汽车不能互换；
- surface 类单独处理，例如 screen、poster/sign、book/paper/map；
- 不规则物体或小目标可以倾向 text edit、膨胀 mask 或矩形 mask；
- donor 不能来自同一个 target video；
- donor 复用要限制，避免一个 reference image 在大量 fake 中重复出现，形成检测捷径。

后续如果继续补数据，应该优先修 taxonomy/compatibility，而不是在 VACE 阶段靠随机 fallback。

## 6. SAM3 mask extraction

SAM3 候选构建入口：

```text
scripts/dataa_v1/build_sam3_inventory_candidates.py
```

SAM3 并行提取入口：

```text
scripts/launch_sam3_parallel.py
scripts/merge_sam3_parallel_results.py
```

当前 inventory v2 路线的典型产物：

```text
res/sam_track_bank/inventory_v2/parallel_runs/sam3_quality_tracks.json
```

mask npz 大文件位于服务器 `/tmp`，例如：

```text
/tmp/cambench_train/cam_train/object_discovery_sam/inventory_v2/track_masks/...
```

SAM3 `.npz` 是唯一 mask 真值。格式要求：

```text
frame_indices: int32 [N_visible]
masks: uint8 [N_visible, H, W]
```

中间不可见 gap 不复制最近可见 mask。VACE package 阶段会在 target track 的 first-visible 到 last-visible 生命周期内保留时间上下文，不可见帧写零 mask。

## 7. Pairing dataset

入口脚本：

```text
scripts/dataa_v1/build_pairing_dataset.py
```

当前 materialized dataset 默认放在：

```text
/tmp/camerabenchtrain/dataset
```

每个 paired case 的目录命名：

```text
/tmp/camerabenchtrain/dataset/pairs/<operation>/<target_video_id>__<case_id>/
```

典型内容：

```text
source_video.mp4
target_mask_raw.npz
target_mask_effective.npz
target_mask_vis.mp4
donor_mask_raw.npz
reference.png
reference_alpha.png
pair_meta.json
vace_case_spec.json
```

索引与审计：

```text
res/dataA_v1/dataset_v2/pairing_dataset_index.json
res/dataA_v1/audits/pairing_dataset_v2_audit.json
```

当前已知一次 pairing 结果约为：

```text
pairs=714
operations={
  object_swap: 277,
  person_appearance_swap: 406,
  surface_attribute_edit: 31
}
```

这个数量小于 CameraBench 训练视频总数，是因为已完成 14B target video 被保留并跳过，且 donor/target 复用、compatibility、mask 质量和视频占用都会过滤一部分样本。

## 8. Operation 策略

当前目标是让 fake 对检测任务有足够可见的 AIGC 痕迹，同时不破坏相机运动上下文。

优先级：

1. 如果视频里有足够明显、不是太小的人物，优先做 `person_appearance_swap`。
2. 没有人物或人物太小，再考虑语义兼容的 `object_swap`。
3. screen/poster/sign/book/map 等 surface 可以做 `surface_attribute_edit` 或 text degradation。
4. 配对质量差、找不到 donor、或 target/donor 语义不兼容时，不强行 reference swap，可进入 text-edit reserve。

不推荐：

- target 是路灯，donor 是台灯/椅子，还继续 object swap；
- 真实人物和卡通/3D 人物混配，除非 plan 明确记录该策略；
- 同一 reference image 被几十个 fake 重复使用；
- 人物 swap 使用矩形 mask。

## 9. Effective mask policy

mask policy 在 plan 或 pairing dataset 阶段冻结，package 阶段只执行。

当前常见 variant：

```text
sam3_shape     保留 SAM3 原始轮廓
dilated        在原始轮廓基础上膨胀
expanded_bbox  用扩张 bbox 形成矩形区域
```

建议策略：

- person swap：允许腐蚀/膨胀，不使用矩形 mask；
- 语义相近且形状相近的 object swap：优先保留 `sam3_shape`；
- 小目标或不规则目标：可用 `dilated` 或 `expanded_bbox` 增加编辑可见度；
- target/donor 跨细分类但仍兼容时，可倾向 `expanded_bbox`，降低原物体轮廓暴露。

manifest 需要记录：

```text
mask_variant_type
seed
dilation_radius_px
erosion_radius_px
bbox_expand_ratio
original/effective area stats
original/effective bbox tube
trigger reason
```

## 10. VACE execution plan

从 pairing dataset 冻结 VACE plan：

```text
scripts/dataa_v1/build_vace_plan_from_pairing_dataset.py
```

当前 1.3B plan：

```text
res/dataA_v1/plans/frozen_dataset_v2_vace13b_plan.json
```

text-edit reserve plan：

```text
scripts/dataa_v1/build_textedit_reserve_plan.py
res/dataA_v1/plans/frozen_dataset_v2_textedit_reserve_vace13b_plan.json
```

14B 历史 subject-first plan/run 仍保留，用于保留已经生成且质量可接受的样本：

```text
res/dataA_v1/plans/frozen_subject_first_vace14b_execution_plan.json
/tmp/cameramotion_det/dataA_v1/vace14b/dataa_v1_subject_first_vace14b_finalv1
```

当前原则：

- 14B 已完成且接受的结果保留，不重跑；
- 1.3B 质量不满意的部分，优先通过更好的 inventory/taxonomy/pairing 重做 plan；
- 旧 plan、continuation plan、dataset v2 plan 的 lineage 需要保留，不能混成不可回溯的一份文件。

## 11. VACE package 与生成

单 case package：

```text
scripts/dataa_v1/package_vace_case.py
```

批量执行入口：

```text
scripts/dataa_v1/run_vace14b_batch.py
scripts/dataa_v1/vace_persistent_worker.py
scripts/dataa_v1/vace_runtime.py
```

虽然入口脚本名含 `14b`，但 1.3B 和 14B 都使用同一个 runner。真正模型由 config、checkpoint 和 plan 中的 `sampling_meta.vace_model_plan` 决定。

当前 1.3B fast config：

```text
configs/dataa_v1/vace13b_subject_first_fast.yaml
```

关键设置：

```yaml
run:
  tmp_root: /tmp/cameramotion_det/dataA_v1/vace13b

vace:
  model_name: vace-1.3B
  profile: production_480
  size: 480p
  sample_steps: 16
  force_flash_attn_2: true
  offload_model: false
  t5_cpu: false

gpu:
  worker_groups: 16
  gpus_per_worker: 1
  workers_per_gpu: 1
```

运行根目录由 config 的 `run.tmp_root` 和命令行 `--run-id` 拼出：

```text
/tmp/cameramotion_det/dataA_v1/vace13b/<run-id>
```

## 12. 单 case 产物

每个 attempt 目录中应包含：

```text
preflight_report.json
source_clip.mp4
target_mask_raw.npz
target_mask_edit.npz
target_mask_gen.npz
target_mask_alpha.npz
target_mask_gen.mp4
source_vace_condition.mp4
donor_reference.png / donor_reference_alpha.png   # donor route only
vace_command.json
case_manifest.json
generated_raw.mp4
generation_result.json
full_real.mp4
full_fake.mp4
upload_receipt.json
```

`source_vace_condition.mp4` 是 VACE 实际输入，不应回退到原始 `source_clip.mp4`。

package 期硬校验：

```text
source_clip.mp4
target_mask_gen.mp4
source_vace_condition.mp4
```

三者必须 frame_count / fps / height / width 完全一致。

## 13. Full-video reassembly

逻辑文件：

```text
scripts/dataa_v1/full_video.py
```

VACE 生成的是局部生命周期片段，最终样本必须重新拼回完整视频：

```text
generated_raw.mp4
-> generated_trimmed.mp4
-> edited_segment.mp4
-> full_fake.mp4

原 target video
-> full_real.mp4
```

最终要求：

```text
full_real.mp4 与 full_fake.mp4:
  fps 完全一致
  frame_count 完全一致
  height/width 完全一致
```

已修复的问题：

- 部分样本生成后 `full_real` 与 `full_fake` 只差一帧；
- 当前新增 `repair_full_video_one_frame_mismatch.py`，可裁掉更长视频最后一帧并更新 manifest/run_state；
- 后续 reassembly 也会自动尝试修复 one-frame mismatch。

修复脚本：

```text
scripts/dataa_v1/repair_full_video_one_frame_mismatch.py
```

## 14. OSS 与 /tmp 规则

服务器上大文件优先写 `/tmp`：

```text
/tmp/cambench_train/cam_train/video
/tmp/cambench_train/cam_train/object_discovery_sam/...
/tmp/camerabenchtrain/dataset
/tmp/cameramotion_det/dataA_v1/vace13b/<run-id>
/tmp/cameramotion_det/dataA_v1/vace14b/<run-id>
```

原因：

- `/tmp` 空间更大、速度更快；
- `/input` 是个人 NAS，空间较小，不适合堆视频/npz/attempt；
- `/tmp` 会随容器/镜像关闭丢失，所以必须及时上传 OSS。

Git 中只保留：

```text
plan JSON
audit JSON
index JSON
config
代码
文档
```

不提交：

```text
模型权重
原始视频
生成视频
mask npz
attempt 目录
wheel/cache
```

## 15. 当前常用命令

Qwen inventory v2：

```bash
python scripts/run_qwen_object_proposals.py \
  --manifest data/cambench_videos.json \
  --out-root res/qwen_inventory_v2 \
  --prompt-profile inventory_v2 \
  --model-name Qwen3-VL-8B-Instruct \
  --api-base http://127.0.0.1:8000/v1 \
  --max-output-tokens 3000 \
  --max-concurrency 40 \
  --all-videos
```

构建 SAM3 candidates：

```bash
python scripts/dataa_v1/build_sam3_inventory_candidates.py \
  --inventory res/qwen_inventory_v2/qwen_inventory_entities.json \
  --out res/qwen_inventory_v2/qwen_sam3_candidates_inventory_v2.json
```

并行跑 SAM3：

```bash
python scripts/launch_sam3_parallel.py \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --workers-per-gpu 1 \
  --run-id sam3_inventory_v2_YYYYMMDD \
  --qwen-candidates res/qwen_inventory_v2/qwen_sam3_candidates_inventory_v2.json \
  --out-root res/sam_track_bank/inventory_v2 \
  --mask-root /tmp/cambench_train/cam_train/object_discovery_sam/inventory_v2/track_masks \
  --startup-wave-delay-sec 5
```

合并 SAM3：

```bash
python scripts/merge_sam3_parallel_results.py \
  --run-id sam3_inventory_v2_YYYYMMDD \
  --num-workers 16 \
  --qwen-candidates res/qwen_inventory_v2/qwen_sam3_candidates_inventory_v2.json \
  --out-root res/sam_track_bank/inventory_v2/parallel_runs
```

构建 pairing dataset：

```bash
python scripts/dataa_v1/build_pairing_dataset.py \
  --inventory res/qwen_inventory_v2/qwen_inventory_entities.json \
  --track-bank res/sam_track_bank/inventory_v2/parallel_runs/sam3_quality_tracks.json \
  --taxonomy configs/dataa_v1/taxonomy_v2_seed.json \
  --compatibility configs/dataa_v1/compatibility_matrix_v2.json \
  --dataset-root /tmp/camerabenchtrain/dataset \
  --out-index res/dataA_v1/dataset_v2/pairing_dataset_index.json \
  --out-audit res/dataA_v1/audits/pairing_dataset_v2_audit.json \
  --completed-run-root /tmp/cameramotion_det/dataA_v1/vace14b/dataa_v1_subject_first_vace14b_finalv1 \
  --ffmpeg-bin /input/tmp/ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
  --ffprobe-bin /input/tmp/ffmpeg/ffmpeg-7.0.2-amd64-static/ffprobe \
  --num-workers 64 \
  --execute
```

构建 1.3B VACE plan：

```bash
python scripts/dataa_v1/build_vace_plan_from_pairing_dataset.py \
  --pairing-index res/dataA_v1/dataset_v2/pairing_dataset_index.json \
  --out-plan res/dataA_v1/plans/frozen_dataset_v2_vace13b_plan.json \
  --model-name vace-1.3B \
  --profile production_480 \
  --size 480p
```

运行 1.3B VACE：

```bash
python scripts/dataa_v1/run_vace14b_batch.py \
  --execution-plan res/dataA_v1/plans/frozen_dataset_v2_vace13b_plan.json \
  --track-bank res/sam_track_bank/inventory_v2/parallel_runs/sam3_quality_tracks.json \
  --config configs/dataa_v1/vace13b_subject_first_fast.yaml \
  --checkpoint-dir /home/admin/wan2.1-VACE-1.3B \
  --run-id dataa_v1_dataset_v2_vace13b_v1 \
  --oss-prefix oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataA_v1/vace13b_dataset_v2 \
  --topology 16x1 \
  --workers-per-gpu 1 \
  --resume \
  --allow-reshard \
  --execute \
  --launch-workers
```

修复已生成但 full pair 差一帧的样本：

```bash
python scripts/dataa_v1/repair_full_video_one_frame_mismatch.py \
  --run-root /tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_dataset_v2_vace13b_v1 \
  --ffmpeg-bin /input/tmp/ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
  --ffprobe-bin /input/tmp/ffmpeg/ffmpeg-7.0.2-amd64-static/ffprobe \
  --oss-prefix oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataA_v1/vace13b_dataset_v2 \
  --execute \
  --execute-upload
```

## 16. 质量风险与后续重点

当前最重要的质量问题：

- 1.3B 生成快，但 reference-following 和局部质量不一定稳定；
- 早期 plan 存在明显人物却选小物体或奇怪物体的情况；
- object donor/target 语义不匹配会导致生成很差；
- donor/reference 重复复用会成为检测捷径；
- 小目标原始 SAM3 mask 太贴边时，生成后变化不明显；
- text-edit reserve 能补量，但需要单独标记 route，不能和 reference swap 混淆。

后续建议：

1. 保留已确认可用的 14B 结果。
2. 1.3B 不满意的部分先重做 Qwen inventory/taxonomy/pairing，再跑 VACE。
3. 有明显人物的视频优先 person swap，并区分真实/卡通/3D/视角。
4. donor 使用 ledger 限制，尽量不让同一 reference image 多次出现在 fake 中。
5. 对没有合适 donor 的视频，用 text-edit reserve 补量，并在 manifest 中明确 `generator_route`。
6. 最终 dataset index 统一合并 14B、1.3B、text-edit reserve，并带上 CameraBench camera motion caption/labels。

## 17. 最终数据索引建议

最终发布/训练入口建议不是直接扫目录，而是生成一个统一 index，例如：

```text
res/dataA_v1/final/dataA_v1_video_pairs_index.json
```

每条记录至少包含：

```json
{
  "case_id": "...",
  "target_video_id": "...",
  "operation": "person_appearance_swap",
  "generator": "VACE-1.3B",
  "generator_route": "dataset_v2_pairing",
  "real_path": ".../full_real.mp4",
  "fake_path": ".../full_fake.mp4",
  "source_video_path": "...",
  "mask_npz_path": ".../target_mask_gen.npz",
  "mask_policy": {},
  "target": {
    "track_id": "...",
    "taxonomy_label": "...",
    "bbox_tube": "...",
    "time_range_sec": [0.0, 0.0]
  },
  "donor": {
    "track_id": "...",
    "video_id": "...",
    "reference_image_path": "..."
  },
  "camera_bench": {
    "split": "train",
    "camera_motion_caption": "...",
    "camera_motion_labels": []
  },
  "lineage": {
    "execution_plan": "...",
    "case_manifest": "...",
    "run_id": "...",
    "oss_prefix": "..."
  }
}
```

这样后续训练检测模型时，可以同时按编辑类型、mask 类型、模型版本、相机运动标签、视觉域、人物/物体类型做分层分析。
