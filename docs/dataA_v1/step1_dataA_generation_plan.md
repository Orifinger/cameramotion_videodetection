# Data A v1 Step 1 生成方案

本文档整理当前 Data A v1 的 Step 1 生成流程：从已有 Qwen/SAM3 结果出发，构建 subject-first VACE execution plan，打包单 case，运行 VACE，生成完整视频级 Real/Fake，并保留可回溯信息。

当前目标不是重新做对象发现或重做采样，而是在已有资源基础上尽快产出可用于 AIGC 检测训练的 Data A 样本。

## 1. 数据定义

Data A v1 的样本单位是完整视频 pair：

```text
Real_A = target 视频 A 的完整规范化真实视频
Fake_A = 同一个 target 视频 A 中，目标对象生命周期片段被局部 AIGC 编辑后拼回的完整视频
donor B = 仅作为 reference condition，不能作为 Real pair，也不能直接贴到 target
```

关键约束：

- Fake 必须来自同一个 target video。
- 编辑区域必须来自 SAM3 mask tube，不允许用 bbox 代替 SAM3 mask 作为主输入。
- donor RGB 只能作为 VACE reference condition，不能进入 target compositing。
- 最终 `full_real.mp4` 和 `full_fake.mp4` 必须 fps、frame_count、height、width 一致。
- 生成失败、mask 不一致、路径缺失、语义不兼容时要 block 并记录原因，不做脏 fallback。

## 2. 当前输入资产

Step 1 不重跑以下上游步骤：

- Qwen3-VL 对象候选发现。
- 视频 domain/style 分类。
- SAM3 track/mask tube。
- 已完成 case 的生成结果。

当前主要输入文件：

```text
res/sam_track_bank/sam3_quality_tracks_enriched.json
res/sam_track_bank/sam3_quality_tracks_enriched_merged_v001.json
res/dataA_v1/plans/frozen_subject_first_vace14b_execution_plan.json
configs/dataa_v1/subject_selection_v1.json
configs/dataa_v1/vace13b_subject_first_fast.yaml
configs/dataa_v1/vace14b_subject_first.yaml
```

服务器上的原始大文件主要在：

```text
/tmp/cambench_train/cam_train/video
/tmp/cambench_train/cam_train/object_discovery_sam/track_masks_v1/...
```

运行产物主要在：

```text
/tmp/cameramotion_det/dataA_v1/vace14b/<run-id>
/tmp/cameramotion_det/dataA_v1/vace13b/<run-id>
```

`/tmp` 可正常读写，速度快、空间大，但会随镜像结束丢失，因此需要及时上传 OSS。`/input` 是 NAS，不适合长期保存大量视频中间产物。

## 3. 当前生成策略

### 3.1 Subject-first target selection

当前 subject-first 选择脚本：

```text
scripts/dataa_v1/build_subject_first_execution_plan.py
```

选择依据来自 `subject_selection_v1.json`：

- `primary_probability = 0.90`
- 替换类 operation 的面积门槛更高。
- surface 类 operation 的面积门槛较低。
- 每个视频尽量选一个主编辑对象，避免同一视频重复占用。

当前已知问题：

- 仍可能出现“画面中有明显人物，但 plan 使用了奇怪物体”的情况。
- 原因是当前逻辑仍会尊重 base plan 的 operation；只有在原 operation 不兼容或无法 repair 时才更强地切到 person。
- 后续应改成：同一视频里只要存在足够大的真人/人物 track，就优先 `person_appearance_swap`，再考虑 object/surface。

### 3.2 Continuation plan

已经跑过的样本保留，不重跑。对未完成样本，通过 continuation plan 重新规划：

```text
scripts/dataa_v1/build_continuation_execution_plan.py
```

它会：

- 扫描已有 run root，识别已完成 case/video。
- 已完成的 case/video 跳过。
- 未完成的 case 尝试重新选择 target / donor / operation。
- 同一视频冲突的 case 写入 reserve plan。
- 需要 Qwen/SAM3 rerun 的视频写入 rerun manifest。
- 可以通过 `--force-model vace-1.3B` 将新生成的 continuation case 统一冻结为 1.3B。

默认输出：

```text
res/dataA_v1/plans/frozen_subject_first_vace_continuation_plan.json
res/dataA_v1/plans/frozen_subject_first_vace14b_continuation_plan.json
res/dataA_v1/plans/frozen_subject_first_vace13b_continuation_plan.json
res/dataA_v1/plans/frozen_subject_first_vace_continuation_reserve_plan.json
res/dataA_v1/audits/subject_first_continuation_plan_audit.json
res/dataA_v1/audits/subject_first_qwen_sam3_rerun_manifest.json
```

## 4. Case 打包流程

单 case 打包入口：

```text
scripts/dataa_v1/package_vace_case.py
```

打包流程：

```text
execution plan case
-> track-bank 回查 target/donor track
-> path mapping / mask npz preflight
-> 选择 target 生命周期片段
-> 导出 source_clip.mp4
-> 构建 M_raw / M_edit / M_gen / M_alpha
-> 导出 target_mask_gen.npz
-> 导出 target_mask_gen.mp4
-> 反解码 mask video 并和 M_gen 做一致性校验
-> 导出 source_vace_condition.mp4
-> 导出 donor_reference.png / donor_reference_alpha.png
-> 构建 model_prompt / control_prompt
-> 写 vace_command.json
-> 写 case_manifest.json
```

### 4.1 Clip 生命周期

当前策略：

- clip 覆盖 target track 的 `first_visible_frame -> last_visible_frame`。
- 中间不可见 gap 保留时间上下文。
- 不可见帧 mask 为零，不复制最近可见 mask。
- 短片段按 canonical fps 映射并补到最近 `4n+1`。
- 长片段压到可用 frame budget，并记录 case 级 `generation_fps`。

### 4.2 Mask 四层

每个 case 保存四层 mask：

```text
M_raw   = SAM3 原始 mask 映射到 canonical 时间轴
M_edit  = 轻度清理后的二值编辑区域
M_gen   = 实际给 VACE 的二值生成区域
M_alpha = full-video compositing 使用的 soft alpha
```

mask policy 在 plan 的 `sampling_meta.mask_policy` 中冻结。package 阶段只执行，不再随机抽。

常见 mask variant：

```text
sam3_shape
dilated
expanded_bbox
```

person swap 不使用矩形 bbox mask，但可做腐蚀/膨胀。

### 4.3 VACE condition video

VACE 实际输入不是原始 `source_clip.mp4`，而是：

```text
source_vace_condition.mp4
```

manifest 中：

```json
"source_clip": {
  "source_clip_path": ".../source_clip.mp4",
  "source_vace_condition_path": ".../source_vace_condition.mp4",
  "vace_input_path": ".../source_vace_condition.mp4"
}
```

打包期硬校验：

```text
source_clip.mp4
target_mask_gen.mp4
source_vace_condition.mp4
```

三者必须 frame_count / fps / height / width 完全一致。

## 5. VACE 运行流程

批量运行入口：

```text
scripts/dataa_v1/run_vace14b_batch.py
```

虽然脚本名仍含 `14b`，但当前也用于 1.3B；真正模型由 config 和 plan 中的 `sampling_meta.vace_model_plan` 决定。

运行流程：

```text
execution plan
-> runtime preflight
-> deterministic sharding
-> persistent worker groups
-> 每个 worker 模型只加载一次
-> 连续消费 assigned shard
-> package case
-> VACE generate
-> full-video reassembly
-> 写 generation_result.json
-> 上传 OSS
```

### 5.1 当前 1.3B deadline 配置

配置文件：

```text
configs/dataa_v1/vace13b_subject_first_fast.yaml
```

当前关键参数：

```yaml
vace:
  model_name: vace-1.3B
  profile: production_480
  size: 480p
  sample_steps: 16
  use_prompt_extend: plain
  force_flash_attn_2: true
  offload_model: false
  t5_cpu: false

gpu:
  worker_groups: 16
  gpus_per_worker: 1
  workers_per_gpu: 1
```

说明：

- `sample_steps: 16` 是 deadline 优先配置，质量会低于 30/50 steps，但速度更快。
- `workers_per_gpu: 1` 避免每张卡两个 worker 抢显存和带宽。
- 如果观察到 GPU 利用率明显不足，可以临时命令行加 `--batch-size 2` 做 worker 级并发。
- `--batch-size` 是每 GPU 多 worker，并不是模型内部 batch。

### 5.2 FlashAttention

当前 runtime 会强制检查 FlashAttention 2：

- `force_flash_attn_2: true`
- 如果 FA2 不可用，会报错。
- 如果检测到 FA3，会强制关闭 FA3，走 FA2。
- `generation_result.json` 会记录 `attention_backend`。

## 6. Full-video reassembly

full-video 逻辑在：

```text
scripts/dataa_v1/full_video.py
```

生成后会立即执行：

```text
generated_raw.mp4
-> generated_trimmed.mp4，如果需要裁掉 padding
-> edited_segment_padded.mp4
-> edited_segment.mp4
-> full_real.mp4
-> full_fake.mp4
```

`full_fake.mp4` 的构造：

```text
prefix 原视频片段
+ edited_segment
+ suffix 原视频片段
```

其中 edited segment 使用：

```text
M_alpha * generated + (1 - M_alpha) * source_clip
```

full pair 校验：

```text
full_real fps == full_fake fps
full_real frame_count == full_fake frame_count
full_real height == full_fake height
full_real width == full_fake width
donor_rgb_used == false
```

失败时 block，不输出 accepted pair。

## 7. Run state / 续跑 / 进度

状态文件：

```text
<run-root>/coordinator/run_state.json
<run-root>/coordinator/case_status.jsonl
<run-root>/coordinator/batch_plan.json
<run-root>/coordinator/plan_lineage.json
```

当前已实现：

- `run_state.json` 原子写。
- 多 worker 更新状态时使用 lock。
- 如果 `run_state.json` 损坏，可从 `case_status.jsonl` 恢复。
- `--resume` 会跳过已有 terminal/uploaded case。
- `--allow-reshard` 允许更换 topology 后继续跑。
- 运行时每 30 秒打印进度：

```text
progress elapsed_min=... total=... seen=... done=... blocked=... active=... alive_workers=...
```

查看完成情况：

```bash
ROOT=/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_subject_first_vace13b_remaining_v1

python - <<'PY'
import json
from pathlib import Path
from collections import Counter

root = Path("/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_subject_first_vace13b_remaining_v1")
status = Counter()
blockers = Counter()

for p in root.rglob("case_manifest.json"):
    x = json.loads(p.read_text(encoding="utf-8"))
    status[x.get("stage_status", "<missing>")] += 1
    for b in (x.get("preflight") or {}).get("blockers", []):
        blockers[b] += 1

print("stage_status")
for k, v in status.most_common():
    print(k, v)

print("\nblockers")
for k, v in blockers.most_common():
    print(k, v)
PY
```

查看生成结果：

```bash
ROOT=/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_subject_first_vace13b_remaining_v1
find "$ROOT" -name generation_result.json | wc -l
find "$ROOT" -name full_fake.mp4 | wc -l
find "$ROOT" -name full_real.mp4 | wc -l
```

## 8. 当前推荐执行命令

### 8.1 拉最新代码

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
git pull origin main
git log -1 --oneline
```

### 8.2 构建 1.3B continuation plan

对剩余未完成样本统一转 1.3B：

```bash
python scripts/dataa_v1/build_continuation_execution_plan.py \
  --track-bank res/sam_track_bank/sam3_quality_tracks_enriched_merged_v001.json \
  --base-plan res/dataA_v1/plans/frozen_subject_first_vace14b_execution_plan.json \
  --run-root /tmp/cameramotion_det/dataA_v1/vace14b/dataa_v1_subject_first_vace14b_finalv1 \
  --selection-config configs/dataa_v1/subject_selection_v1.json \
  --ffprobe-bin /input/tmp/ffmpeg/ffmpeg-7.0.2-amd64-static/ffprobe \
  --num-workers 64 \
  --progress-every 100 \
  --force-model vace-1.3B
```

如果还要把已有 1.3B run 也作为已完成来源，追加：

```bash
  --run-root /tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_subject_first_vace13b_remaining_v1
```

### 8.3 运行 1.3B VACE

```bash
python scripts/dataa_v1/run_vace14b_batch.py \
  --execution-plan res/dataA_v1/plans/frozen_subject_first_vace13b_continuation_plan.json \
  --track-bank res/sam_track_bank/sam3_quality_tracks_enriched_merged_v001.json \
  --config configs/dataa_v1/vace13b_subject_first_fast.yaml \
  --checkpoint-dir /home/admin/wan2.1-VACE-1.3B \
  --run-id dataa_v1_subject_first_vace13b_remaining_v1 \
  --oss-prefix oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataA_v1/vace13b \
  --topology 16x1 \
  --resume \
  --allow-reshard \
  --execute \
  --launch-workers
```

如果 GPU 利用率长期很低，可以试：

```bash
  --batch-size 2
```

如果单卡负载已经高，不要开 `--batch-size 2`，否则可能更慢。

## 9. Dataset view 方案

当前建议新增一个整理视图目录：

```text
/tmp/camerabenchtrain/dataset/
```

结构建议：

```text
/tmp/camerabenchtrain/dataset/
├── person_appearance_swap/
│   └── <target_video_stem>__<case_id>/
├── object_swap/
│   └── <target_video_stem>__<case_id>/
├── object_attribute_edit/
│   └── <target_video_stem>__<case_id>/
├── surface_content_edit/
│   └── <target_video_stem>__<case_id>/
└── surface_attribute_edit/
    └── <target_video_stem>__<case_id>/
```

每个 pair 目录建议包含：

```text
full_real.mp4
full_fake.mp4
source_clip.mp4
generated_raw.mp4
generated_trimmed.mp4
edited_segment.mp4
target_mask_gen.mp4
source_vace_condition.mp4
donor_reference.png
donor_reference_alpha.png
case_manifest.json
generation_result.json
dataset_pair_meta.json
```

用途划分：

- VACE 生成主流程仍以 `execution_plan + track_bank + 原始 video/mask` 为真源。
- dataset view 用于训练、质检、人工查看、上传整理。
- 不建议把 dataset view 作为唯一真源，因为 replan / rerun / mask policy repair 仍需要 plan 和 track-bank。

实现建议：

- 优先 hardlink，失败再 copy。
- 不使用 symlink，避免上传或搬运时只上传链接。
- 缺失文件写入 `dataset_pair_meta.json` 的 `missing_files`，不静默 fallback。

## 10. 当前主要问题与后续修改

### 10.1 Plan 质量问题：person-first 不够强

当前现象：

- 视频里明明有明显人物，但 plan 仍选 object/surface。
- 某些 pair 目标奇怪，生成效果弱，训练价值低。

建议修改：

```text
只要同一视频中有不小的人物 track
-> 优先 person_appearance_swap
-> 优先真人配真人 donor，卡通/3D 人物配同类 donor
-> 找不到同类 donor 时才允许 any person donor，并记录 risk_tags
-> 找不到 person target/donor 时才回退 object/surface
```

需要修改：

```text
scripts/dataa_v1/build_subject_first_execution_plan.py
scripts/dataa_v1/build_continuation_execution_plan.py
configs/dataa_v1/subject_selection_v1.json
```

可选修改：

```text
scripts/run_qwen_object_proposals.py
```

让 Qwen rerun 明确区分：

```text
real_person
cartoon_person
3d_character
statue/mannequin/toy figure
```

### 10.2 速度问题

当前 deadline 配置：

```text
1.3B
480p
16 steps
16x1
workers_per_gpu=1
```

再提速选项：

- `sample_steps: 12`，质量再下降，但速度更快。
- 对低价值 case 进一步减少 frame_count。
- 优先跑 person / 大主体 case，surface 和小目标可放后。

不建议：

- 默认每卡 2 worker。如果 GPU 已经忙，会互抢资源。
- 开 `offload_model` 或 `t5_cpu`，这是省显存，不是加速。
- 在 deadline 前接入复杂推理加速框架，风险太高。

## 11. Git 与存储规则

不能提交到 Git：

```text
模型权重
视频
mask npz
wheel cache
attempt 产物
VACE 输出
dataset view 大文件
```

必须保留到 OSS：

```text
full_real.mp4
full_fake.mp4
case_manifest.json
generation_result.json
upload_receipt.json
必要 mask/video 条件文件
```

本地 `/tmp` 只作为高速工作区，运行结束或镜像关闭前必须上传 OSS。

## 12. 最小成功标准

一个 accepted Data A case 至少应有：

```text
case_manifest.json
generation_result.json
full_real.mp4
full_fake.mp4
source_clip.mp4
target_mask_gen.mp4
source_vace_condition.mp4
target_mask_alpha.npz
```

并满足：

```text
full_real/full_fake fps 一致
full_real/full_fake frame_count 一致
full_real/full_fake height/width 一致
mask video roundtrip 通过
donor_rgb_used == false
sampling_meta 可追溯到 plan
```

对于训练优先级：

```text
1. person_appearance_swap，尤其是真实世界人物、大主体人物
2. 大主体 object_swap
3. object_attribute_edit
4. surface_content_edit / surface_attribute_edit
5. 小目标或 pair 语义弱的样本放后
```

