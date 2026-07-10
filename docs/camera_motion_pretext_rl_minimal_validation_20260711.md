# 相机运动前置感知强化学习最小验证方案

- 日期：2026-07-11
- 状态：P0 工具链已完成，服务器训练待执行
- 目标：在不向检测模型提供外部 camera caption 的前提下，验证“让模型先从视频中学习相机运动能力”能否提高 AIGC 视频检测。

## 1. 结论先行

当前优先验证的方法是“相机运动前置感知强化学习”。相机信息不再作为 user prompt 中的输入条件，而是作为模型必须从视频中预测的前置感知任务。完成相机任务训练后，再在完全相同的检测任务上比较普通检测对照与相机前置感知模型。

这条路线与已经失败的 camera caption 注入实验有三个本质区别：

1. 推理时不需要 CameraBench 模型、camera label 或 camera caption。
2. 训练时 camera labels 是模型输出目标和奖励真值，不是可以被忽略的输入文本。
3. camera pretext 与 detection 分阶段训练，不把两个互不关联的 CoT 强行拼成一条自动标注解释。

CameraBench 已表明相机运动能力可以通过少量标注视频注入生成式 VLM；VideoVeritas 则表明感知前置任务经过 RL 可以迁移到 AIGC 视频检测，并且分阶段学习比把长短差异很大的任务直接混在同一批次更简单有效。

- CameraBench：[论文](https://arxiv.org/abs/2504.15376)，[代码](https://github.com/sy77777en/CameraBench)
- VideoVeritas：[论文](https://arxiv.org/abs/2602.08828)，[代码](https://github.com/EricTan7/VideoVeritas)
- 显式运动表征备选依据：[Efficient Motion-Aware Video MLLM](https://arxiv.org/abs/2503.13016)
- 运动专家用于生成视频检测的依据：[What Matters in Detecting AI-Generated Videos like Sora?](https://arxiv.org/abs/2406.19568)

## 2. 本轮只回答一个问题

> 在相同初始检测 checkpoint、相同 DataA 检测强化学习数据和相同检测评测条件下，提前通过可验证的 camera-motion pretext 学习相机运动，是否比只进行检测任务训练获得更好的 DataA/VIF-Bench 检测结果？

本轮不验证 camera caption 输入、自由文本 caption 质量、光流/深度新结构、完整论文规模扩增和 DataB 伪 camera labels。

## 3. 模型与数据

### 3.1 初始模型

两条训练分支都从同一模型开始：

`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115/`

该 checkpoint 已见过完整 DataB detection 数据，因此 DataB 不能作为无泄漏测试集。DataA test 和 VIF-Bench 承担本轮检测迁移验证。

### 3.2 DataA detection

- 完整 JSON：`/input/workflow_58770161/workspace/test/cameramotion_det/detection/dataa_vace_grounded_cot_instruct_tp8x2_sft_all.json`
- 已知 test split：`/input/workflow_58770161/workspace/test/cameramotion_det/tools/data/camera_motion_splits/dataA_test.json`
- train split：优先读取同目录的 `dataA_train.json`，文件是否存在需在服务器预检。若不存在，用完整 JSON 按 source/case group 排除 test 中所有同组记录后生成。

DataA 共 2152 条 detection 记录，每条通常包含 16 帧。split 必须以 source/case family 为单位，real/fake 对不能跨 train/test。

### 3.3 DataA camera labels

- 服务器：`/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_v2.jsonl`
- 本地镜像：`ourdata/dataA/dataa_cameramotion_labels_v2.jsonl`

本地统计共有 2134 条唯一帧目录记录，可以通过 detection 第一帧的父目录匹配。最小验证只从 DataA train 中选择 real 记录作为 camera pretext，避免同一 source 的 real/fake 重复，也避免 camera 任务学习局部编辑伪影。

camera target 使用规范化 label 集合，不使用 caption。`static` 在当前构建逻辑中描述场景静态性，并不等于相机静止，因此从第一轮 camera-motion reward 中排除；其余标签按固定 taxonomy 顺序输出。

### 3.4 DataB replay

- detection JSON：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`
- camera JSONL：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`

本轮 DataB 只给检测对照分支做训练步数匹配和能力 replay。DataB camera labels 是 CameraBench 模型预测结果，第一轮不加入 camera pretext。

## 4. 输出格式

### 4.1 Camera pretext

输入只包含视频帧与简短 camera-motion 问题，输出限定为：

```text
<camera_motion>["no-shaking", "complex-motion", "pan-right"]</camera_motion>
```

要求：

- label 必须来自固定 taxonomy；
- 顺序固定，比较时按集合计算；
- 每条视频只生成一个训练样本；
- 从 6 至 8 个短 prompt 模板中确定性抽取一个，不复制同一视频 25 次；
- 不生成 caption，不生成长 CoT。

### 4.2 Detection

保持当前 detection system prompt、user prompt 和输出结构，不提供任何 camera 文本。GRPO rollout 至少要求：

```text
<think>...</think>
<answer>Fake</answer>
```

第一轮奖励只使用可稳定计算的真假与格式。bbox/时间 IoU 奖励在主信号通过后再加入，避免同时调试太多变量。

## 5. 两条严格对照分支

### 5.1 普通检测强化学习对照

1. 从 checkpoint-2115 开始。
2. 第一阶段使用一批 DataB detection replay，样本数与相机分支 camera pretext 相同。
3. 第二阶段使用完全相同的 DataA train detection GRPO 数据。
4. 推理时使用普通 detection prompt。

### 5.2 相机运动前置感知强化学习

1. 从同一个 checkpoint-2115 开始。
2. 第一阶段使用 DataA train 的 real-only camera pretext 做 GRPO。
3. 第二阶段使用与普通对照完全相同的 DataA train detection GRPO 数据。
4. 推理时仍使用普通 detection prompt，不提供 camera label/caption。

两条分支保持相同的总 optimizer steps、batch、随机种子、LoRA/full 设置、第二阶段数据顺序、学习率、KL、生成数量、保存步数和评测 prompt。

第一阶段样本数不强行写死为 1000。以 DataA train 中可用的 real-only、无泄漏 camera 样本为准，预计约 700 至 800 条；普通对照抽取完全相同数量的 DataB replay。

## 6. GRPO 奖励

### 6.1 Camera 多标签奖励

`camera_set_f1` 解析 `<camera_motion>...</camera_motion>` 中的 JSON label 列表，与真值计算集合 F1：

`2 * TP / (2 * TP + FP + FN)`

额外规则：

- 完全匹配可获得小幅 exact bonus；
- taxonomy 外 label 计入 FP；
- 缺失标签计入 FN；
- 无法解析时为 0；
- format reward 只检查单一完整 camera tag 和合法 JSON。

### 6.2 Detection 奖励

- `detection_binary_acc`：严格解析 `<answer>Fake</answer>` 或 `<answer>Real</answer>`；
- `detection_format`：同时存在非空 `<think>` 和单一 `<answer>`；
- 第一轮不使用 LLM judge、caption 相似度或 bbox/时间奖励；
- 使用 reference/KL 限制，降低 detection CoT 退化成单一短答案的风险。

### 6.3 训练后端

当前 LlamaFactory 官方功能列表没有稳定公开的 Qwen3-VL GRPO 工作流。ms-swift 官方支持 Qwen3-VL、多模态 GRPO、LoRA/full、多机训练和 external reward plugin，因此本轮 RL 使用 ms-swift；已有 LlamaFactory 环境继续用于原 SFT，不修改。

VideoVeritas 公共仓库没有发布论文专用训练启动脚本，只提供 ms-swift fork、通用 reward 示例和推理脚本，因此需要本项目自己的小型 reward plugin 与启动脚本。

## 7. 最小训练规模

- 优先 LoRA GRPO，降低遗忘和 rollout 权重同步成本；
- camera completion 上限 128 tokens；
- detection completion 上限先用 1024 tokens；
- `num_generations` 先用 4；
- 至少保存第一阶段结束、第二阶段中点和第二阶段结束；
- 一套 16 卡顺序跑，或两套 16 卡并行跑两个分支；
- learning rate、beta 和 GPU 切分在 8 至 16 条样本 smoke test 通过后锁定，不在此处猜测。

## 8. 评测矩阵

### 8.1 Camera 能力

使用 DataA test 的 real-only camera 样本，prompt 模板与训练模板分离。报告 exact-set accuracy、micro-F1、macro-F1、per-label precision/recall/F1、非法格式率和 taxonomy 外 label 率。

### 8.2 Detection

两个分支都不提供 camera 信息，测试：

- DataA test：ACC、Balanced ACC、Fake Recall、Fake F1、已有 Evidence@0.3；
- VIF-Bench：平均 ACC、Recall、F1；
- 按 `no-motion`、`minor-motion`、`complex-motion` 分桶的 DataA detection 指标。

DataB 不作为本轮泛化结论，因为祖先 checkpoint 已见过完整 DataB。

## 9. 通过标准

1. 相机分支相对 checkpoint-2115 的 camera macro-F1 至少提高 10 个百分点。
2. 相机分支相对普通检测对照，在 DataA Balanced ACC 或 Fake F1 至少提高 1.5 个百分点。
3. VIF-Bench ACC/F1 下降不超过 0.5 个百分点；提升约 1 个百分点则为强信号。
4. 提升不能只来自 `no-motion`，`minor-motion`/`complex-motion` 至少一个分桶应有明确改善。
5. 所有检测推理都不接收 camera caption。

判定：

- camera 提升且 detection 提升：通过，扩展正式实验；
- camera 提升但 detection 不变：能力已注入但未迁移，转向显式 motion-token adapter；
- camera 未提升：先检查数据、reward 或训练链路，不能直接否定方法；
- detection 提升但 camera 未提升：不能作为 camera 方法结果。

## 10. 处理代码审计

### 10.1 已有代码可复用

| 现有文件 | 可复用内容 | 不能直接使用的原因 |
|---|---|---|
| `tools/build_dataa_camera_context_ablation.py` | 路径标准化、camera 匹配、case conflict key | 生成 camera 输入消融，不是输出目标 |
| `tools/build_camera_context_sft_sets.py` | group split、统计和 JSON 写出 | 仍构建 no/gold/shuffled 输入 |
| `eval/infer_dataa_camera_ablation.sh` | 16 卡并行推理 | 只适合已有 SFT 条件消融 |
| `eval/eval_dataa_camera_ablation.sh` | 调用 DataA detection evaluator | 不计算 camera 多标签和 motion bucket |
| VideoVeritas `plugin.py` | ms-swift ORM、binary/format reward 示例 | 没有 camera reward，也没有论文训练脚本 |

### 10.2 第一批必须新增

| 优先级 | 建议文件 | 作用 |
|---|---|---|
| P0 | `tools/build_camera_pretext_grpo_sets.py` | 对齐 split/detection/camera，生成 camera GRPO、DataB replay、公共 DataA detection GRPO 和 camera eval |
| P0 | `rl/camera_detection_rewards.py` | 注册 camera set-F1、camera format、binary detection、detection format |
| P0 | `tests/test_camera_detection_rewards.py` | 覆盖合法、缺失、多余、乱序、非法 JSON 和 Fake/Real 格式 |
| P0 | `eval/eval_camera_motion_predictions.py` | 计算 exact/micro/macro/per-label camera 指标 |
| P0 | `eval/summarize_detection_by_camera_motion.py` | 对齐 detection 预测与 camera JSONL，输出 motion bucket 指标 |

### 10.3 数据 dry-run 后新增

| 优先级 | 建议文件 | 作用 |
|---|---|---|
| P1 | `train/run_camera_pretext_grpo.sh` | 相机分支第一阶段 |
| P1 | `train/run_detection_replay_grpo.sh` | 普通对照第一阶段 |
| P1 | `train/run_common_dataa_detection_grpo.sh` | 两个分支运行相同 DataA detection 第二阶段 |
| P1 | `train/smoke_camera_grpo.sh` | 用 8 至 16 条样本验证图片、额外字段和 reward plugin |

### 10.4 第一闸门通过前不写

- 光流/压缩 motion-vector 提取与缓存；
- Qwen3-VL motion-token adapter；
- 多 LoRA 动态路由；
- bbox/时间联合 GRPO reward；
- DataB pseudo-camera 扩展；
- 大规模正式无泄漏重训脚本。

## 11. 数据构建审计

P0 构建器必须输出：

- detection/camera 数量和匹配率；
- DataA train/test source group 交集，必须为 0；
- camera pretext 是否全部来自 train/real；
- camera label、motion bucket 和 Real/Fake 分布；
- 每个输出文件的数量、SHA256 和随机种子；
- 无法匹配 camera 的路径样例；
- 被排除的 `static` label 数量。

任何 source group 泄漏、camera eval 进入训练、real/fake pair 跨 split 或空奖励字段都必须报错退出，不能只打印 warning。

## 12. 当前下一步

P0 数据构建器、reward plugin、两个 evaluator 和 reward 单元测试已经实现。本地全量 dry-run 与端到端评测冒烟测试已经通过；这只证明数据和指标链路可执行，不等于方法实验通过。服务器确认 ms-swift 版本、DataA train split 路径和 16 卡 rollout 方式后，再写 P1 训练启动脚本。

## 13. P0 代码运行方法

### 13.1 进入项目并测试 reward

```bash
ROOT=/input/workflow_58770161/workspace/test/cameramotion_det
cd "${ROOT}"

python -m unittest discover \
  -s tests \
  -p 'test_camera_detection_rewards.py' \
  -v
```

预期为 11 项测试全部 `OK`。

### 13.2 检查既有 DataA split

```bash
SPLIT_DIR=${ROOT}/tools/data/camera_motion_splits
test -f "${SPLIT_DIR}/dataA_test.json" && echo "OK: DataA test"
test -f "${SPLIT_DIR}/dataA_train.json" && echo "OK: DataA train" || echo "MISSING: DataA train"
```

如果 `dataA_train.json` 存在，执行：

```bash
OUT=/tmp/1res/camera_pretext_grpo_gate/data

python tools/build_camera_pretext_grpo_sets.py \
  --dataa-detection-json "${ROOT}/detection/dataa_vace_grounded_cot_instruct_tp8x2_sft_all.json" \
  --dataa-camera-jsonl "${ROOT}/camera/camerajson/dataa_cameramotion_labels_v2.jsonl" \
  --datab-detection-json /input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json \
  --dataa-train-json "${SPLIT_DIR}/dataA_train.json" \
  --dataa-test-json "${SPLIT_DIR}/dataA_test.json" \
  --out-dir "${OUT}" \
  --seed 20260711 \
  --check-images
```

如果 `dataA_train.json` 不存在，只删除命令中的 `--dataa-train-json ...` 一行。构建器会从完整 DataA 中按 source/case group 排除 test 的全部同源记录来生成 train，并继续执行零交集硬检查。

查看审计结果：

```bash
python -m json.tool "${OUT}/camera_pretext_grpo_sets_summary.json"
```

必须满足：`dataa_train_test_group_overlap=[]`、`camera_train_eval_group_overlap=[]`、`camera_train_all_real=true` 和 `first_stage_counts_matched=true`。

### 13.3 服务器训练环境预检

```bash
swift --version
python -c 'import swift; print(swift.__file__)'
swift rlhf --help
```

将这三条输出和 `camera_pretext_grpo_sets_summary.json` 发回后，再锁定与服务器 ms-swift 版本完全匹配的 P1 启动参数。当前不要凭不同版本文档直接启动 16 卡 GRPO。

### 13.4 Camera 预测完成后的评测

```bash
python eval/eval_camera_motion_predictions.py \
  --gt-json "${OUT}/camera_pretext_eval.json" \
  --pred-json /path/to/camera_prediction_json_or_dir \
  --out-dir /tmp/1res/camera_pretext_grpo_gate/camera_eval
```

### 13.5 DataA 检测结果按相机运动分桶

先用既有 `eval_dataa.py` 得到 `*_eval_items.csv`，再运行：

```bash
python eval/summarize_detection_by_camera_motion.py \
  --eval-items-csv /path/to/dataa_eval_items.csv \
  --camera-jsonl "${ROOT}/camera/camerajson/dataa_cameramotion_labels_v2.jsonl" \
  --out-json /path/to/dataa_detection_by_camera_motion.json
```

该脚本分别报告 `no-motion`、`minor-motion` 和 `complex-motion` 的 ACC、Balanced ACC、Fake Recall 与 Fake F1。
