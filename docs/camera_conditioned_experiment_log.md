# 相机条件化 AIGC 视频检测实验记录

本文件是当前论文项目的统一实验记录。目标是让新会话不依赖代号或聊天记忆，也能看懂每个实验具体测了什么、使用了哪些数据、得到什么结果，以及下一步为什么这样安排。

## 当前实验索引

| 日期 | 中文实验名称 | 状态 | 这个实验测什么 | 当前结论 |
|---|---|---|---|---|
| 2026-07-08 | 完整 DataB 检测模型的 VIF-Bench 基线 | 已完成 | 仅使用自动标注检测数据训练后，在通用全生成视频测试集上的检测能力 | 旧记录为 ACC 83.96、F1 84.72；需用当前完全一致的提示词复测一次 |
| 2026-07-08 | DataA 与 DataB 混合检测续训 | 未通过 | 加入局部编辑 DataA 检测数据后，能否同时提高 DataA 并保持 VIF-Bench | DataA 没有形成可靠提升，VIF-Bench 明显下降 |
| 2026-07-08 | DataA 与 DataB 相机条件化混合续训 | 未通过 | 在检测训练中加入相机条件后，是否优于普通混合续训 | 只有局部波动，没有稳定优于普通续训，VIF-Bench 仍明显下降 |
| 2026-07-09 | 相机补偿局部残差探针 | 已停止 | 传统全局运动补偿后的局部残差能否区分 DataA 真/假视频 | 整体接近随机，不继续作为主方法 |
| 2026-07-09 | DataA 成对局部编辑选择与位置交换控制 | 已停止 | MLLM 能否从同一真实/编辑视频对中选择局部编辑版本 | 存在严重 A/B 位置偏置，不能作为可靠验证信号 |
| 2026-07-09 | 不训练模型、直接追加相机描述的检测消融 | 未通过 | 给旧检测模型直接追加正确相机描述，是否无需训练即可提升检测 | 正确相机没有优于错误相机或无相机，直接提示注入失败 |
| 2026-07-10 | 普通 DataB 续训与相机文本 DataB 续训 | 未通过 | 经过匹配提示格式训练后，模型是否真正利用正确相机描述改善检测 | 正确相机描述不优于错误描述，并低于不提供相机描述；当前文本条件注入路线未学会利用相机内容 |
| 2026-07-11 | 相机运动前置感知强化学习最小验证 | 降为辅助消融 | 不向检测模型提供外部 camera 文本，先奖励模型从视频预测相机运动，再检验该能力是否迁移到检测 | 单独 camera pretext 没有直接约束局部证据；保留代码，只在局部反事实 Gate 通过后作为增量消融 |
| 2026-07-11 | 相机匹配局部反事实三门验收 | Gate 1 synthetic-rejected DPO 未通过并停止 | 控制相同内容和全局相机运动，只改变局部生成区域，先验证局部信号，再验证配对学习能否迁移到普通检测 | 半程与最终 LoRA-DPO 均未提升选择、定位或位置平衡，且训练偏好目标已正常收敛；排除后半程退化，不再追加 DPO/GRPO 试参 |
| 2026-07-12 | 相机补偿局部感知轨迹最小验证 | 直接探针与融合复核均未通过，路线停止 | 在相同密集原视频帧和局部 mask 监督下，显式相机补偿是否稳定优于未补偿局部轨迹 | 直接 aligned 检测显著退化；`global+aligned` 又低于 global-only 和 `global+unaligned`，五项融合验收全失败，不再追加 anchor/RAFT/融合试参 |
| 2026-07-12 | 相机分层同源配对独立判别最小验证 | 未通过，当前配对排序配方停止 | Real/Fake 分别独立计算 verdict 分数时，增加同源配对排序是否优于等数据、等步数的普通二分类续训 | Pair margin 被优化但 AUC 仅增 0.46 点、pair accuracy 仅增 1.56 点、复杂运动 AUC 仅增 0.27 点，bootstrap 跨 0；不进入 VIF 与 camera pretext |

## 1. 完整 DataB 检测模型的 VIF-Bench 基线

### 这个实验测什么

验证 Qwen3-VL-8B 在完整 DataB 自动标注检测数据上完成检测 SFT 后，对 VIF-Bench 中完整生成视频的真假检测能力。

### 模型与数据

- 初始模型：Qwen3-VL-8B-Instruct。
- 检测训练数据：`v4vif_2766busterall_trainall.json`，共 6766 条。
- 服务器数据路径：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`。
- 训练：约 5 个 epoch，全参数语言模型微调，冻结视觉塔和多模态投影层。
- 结果 checkpoint：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115/`。
- 测试集：VIF-Bench。

### 结果

| 指标 | 旧记录 |
|---|---:|
| VIF-Bench 平均 ACC | 83.96% |
| VIF-Bench F1 | 84.72% |

### 结论与限制

该模型是后续实验的 detection SFT 初始模型。旧 VIF-Bench 推理代码中的 user prompt 后缀与当前严格匹配训练数据的后缀有一个很小的措辞区别，因此正式比较继续训练造成的能力变化前，需要用当前提示词复测该 checkpoint。

## 2. DataA 与 DataB 混合检测续训

### 这个实验测什么

验证把局部编辑生成的 DataA 检测样本与完整生成视频 DataB 混合训练，能否提高 DataA 局部编辑检测，同时保留 VIF-Bench 通用检测能力。

### 结果

| 模型/条件 | VIF 平均 ACC | VIF F1 | DataA ACC | DataA Fake Recall | DataA Evidence@0.3 |
|---|---:|---:|---:|---:|---:|
| 完整 DataB 检测模型 | 83.96% | 84.72% | 51.56% | 34.27% | 15.26% |
| DataA+DataB 混合检测续训 | 73.95% | 76.23% | 48.91% | 45.17% | 23.36% |

### 结论

加入 DataA 后，局部证据命中和 Fake Recall 有局部改善，但 DataA 总体准确率没有提高，VIF-Bench 下降约 10 个百分点。该方案没有形成可接受的检测收益与通用能力平衡。

## 3. DataA 与 DataB 相机条件化混合续训

### 这个实验测什么

验证在 DataA 与 DataB 混合检测训练中加入相机运动条件，能否比不使用相机条件的混合续训更好地判断真假并定位局部伪影。

### 结果

| 模型/条件 | VIF 平均 ACC | VIF F1 | DataA ACC | DataA Fake Recall | DataA Evidence@0.3 |
|---|---:|---:|---:|---:|---:|
| DataA+DataB 普通混合续训 | 73.95% | 76.23% | 48.91% | 45.17% | 23.36% |
| DataA+DataB 相机条件化混合续训 | 75.13% | 74.94% | 52.80% | 43.93% | 19.63% |

### 结论

相机条件化版本在 DataA ACC 和 VIF ACC 上有小幅变化，但 F1、Fake Recall 和证据定位没有一致改善。它没有证明模型真正利用相机信息，而且两种混合续训都损害了 VIF-Bench 保留能力。

## 4. 相机补偿局部残差探针

### 这个实验测什么

先估计全局相机运动，再比较真实视频和局部编辑视频在目标区域中的补偿后残差，检验传统视觉残差信号是否能直接区分局部生成编辑。

### 主要结果

| 数据范围 | 有效样本/总样本 | 成对准确率 | AUC | 结论 |
|---|---:|---:|---:|---|
| 随机 200 对 | 168/200 | 51.79% | 0.514 | 接近随机 |
| `dataA_v1` 家族 | 待补充 | 52.40% | 0.515 | 接近随机 |
| `dataset_v2` 家族 | 待补充 | 48.50% | 0.489 | 接近随机 |
| `textedit_reserve` 家族 | 138/167 | 50.00% | 0.500 | 接近随机 |

### 结论

基于 ORB、RANSAC 和单应性补偿的手工残差没有形成可用区分能力。该路线已停止，不再继续按 DataA 家族重复测试。

## 5. DataA 成对局部编辑选择与位置交换控制

### 这个实验测什么

把同一个案例的真实视频与局部编辑视频分别放在 A/B 两个位置，让模型选择哪一个经过编辑；随后交换 A/B 位置，检查模型是真的识别编辑，还是只偏好固定位置。

### 结果

| 模型 | 原顺序准确率 | 预测 A 比例 | 交换后准确率 | 交换后预测 A 比例 |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 62.50% | 78.50% | 53.50% | 96.50% |
| 完整 DataB 检测模型 | 55.00% | 84.00% | 待补充 | 待补充 |
| Skyra 复现模型 | 52.50% | 96.50% | 待补充 | 待补充 |

Qwen3-VL-8B-Instruct 的位置交换实验中，预测翻转率只有 25%，75% 的样本仍选择相同位置。

### 结论

结果主要受到 A/B 位置偏置影响，不能作为局部编辑识别能力的可靠证据，也不适合作为后续训练标签。

## 6. 不训练模型、直接追加相机描述的检测消融

### 这个实验测什么

保持旧检测模型权重不变，只在 user prompt 中分别加入正确相机描述、其他视频的错误相机描述、明确缺失相机描述或完全不加相机描述，检查直接提示注入是否足以提高检测。

### DataB 1000 条诊断样本结果

| 推理输入 | ACC | Balanced ACC |
|---|---:|---:|
| 不提供相机描述 | 97.50% | 97.51% |
| 正确相机描述 | 96.20% | 96.22% |
| 错误相机描述 | 96.10% | 96.14% |
| 明确缺失相机描述 | 96.60% | 96.59% |

### DataA 测试集结果

| 推理输入 | ACC | Balanced ACC | Fake Recall |
|---|---:|---:|---:|
| 不提供相机描述 | 51.56% | 51.56% | 34.27% |
| 正确相机描述 | 50.93% | 50.93% | 35.20% |
| 错误相机描述 | 51.09% | 51.09% | 37.38% |
| 明确缺失相机描述 | 50.47% | 50.47% | 31.15% |

### 结论

正确相机描述没有优于不提供或提供错误相机描述。该结果只否定“无需训练、直接追加相机文本”的方案，不否定经过匹配格式训练后模型利用相机信息的可能性。

## 7. 普通 DataB 续训与相机文本 DataB 续训

### 这个实验测什么

从同一个完整 DataB 检测 checkpoint 出发，在同一批 DataB 样本上继续训练一轮：一个模型不接收相机文本，另一个模型接收当前视频匹配的相机标签和描述。随后比较正确、错误、缺失和不提供相机描述时的检测结果，判断模型是否真正利用相机内容。

### 当前模型定义

| 中文名称 | 代码名 | 初始 checkpoint | 继续训练数据 | 训练时相机输入 | 结果 checkpoint |
|---|---|---|---|---|---|
| 普通 DataB 续训模型 | M0 | 完整 DataB 检测 checkpoint-2115 | 约 4739 条 camera-covered DataB 训练记录 | 不提供 | `/tmp/1res/datab_camera_train_no_camera/checkpoint-297` |
| 相机文本 DataB 续训模型 | M1 | 完整 DataB 检测 checkpoint-2115 | 与普通续训模型相同 | 提供匹配 labels/caption | `/tmp/1res/datab_camera_train_gold_camera/checkpoint-297` |

### 数据文件

- 完整 DataB 检测数据：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`。
- DataB 相机描述：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`。
- 生成的训练与诊断测试文件目录：`/input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory/data`。
- 诊断测试文件：
  - `datab_camera_eval_no_camera.json`
  - `datab_camera_eval_gold_camera.json`
  - `datab_camera_eval_shuffled_camera.json`
  - `datab_camera_eval_null_camera.json`

### 已完成：VIF-Bench 缺失相机输入测试

#### 这个测试具体测什么

VIF-Bench 当前没有逐视频相机描述，因此两个续训模型推理时都没有获得相机文本。普通续训模型的输入与训练一致；相机文本续训模型的输入与训练不一致。该结果只衡量通用检测能力保留和相机输入缺失时的鲁棒性，不衡量正确相机信息带来的收益。

#### 结果来源

- 普通续训模型：`res/vifbench_datab_camera_m0/m0_vifbench_merged.json`。
- 相机文本续训模型缺失相机输入：`res/vifbench_datab_camera_m1_no_camera/m1_no_camera_vifbench_merged.json`。

| 测试条件 | 平均 ACC | Recall | F1 |
|---|---:|---:|---:|
| 普通 DataB 续训模型，不提供相机描述 | 79.00% | 89.11% | 80.30% |
| 相机文本 DataB 续训模型，不提供相机描述 | 78.26% | 87.53% | 79.37% |
| 相机文本续训模型相对普通续训模型 | -0.74 | -1.58 | -0.93 |

#### 结论

相机文本续训模型在缺失训练时相机输入的情况下没有完全失效，但比普通续训模型低约 1 个百分点。这个结果不能判断相机信息是否有帮助。两个续训模型都低于旧 detection checkpoint 的历史 VIF 结果，说明继续在较窄的 camera-covered DataB 子集上训练可能造成能力遗忘；旧 checkpoint 仍需使用当前提示词严格复测。

### 已完成：DataB 正确/错误/缺失相机快速验证

#### 这个测试具体测什么

在约 1000 条 DataB 诊断样本上，检查相机文本续训模型得到当前视频的正确相机描述时，是否比得到其他视频的错误描述、明确缺失描述或完全没有描述时更准确。

#### 最小测试矩阵

| 模型 | 推理输入 | 作用 |
|---|---|---|
| 普通 DataB 续训模型 | 不提供相机描述 | 训练步数匹配的普通对照 |
| 相机文本 DataB 续训模型 | 正确相机描述 | 要验证的条件 |
| 相机文本 DataB 续训模型 | 错误相机描述 | 检查是否真正使用相机内容 |
| 相机文本 DataB 续训模型 | 明确缺失相机描述 | 检查缺失条件 |
| 相机文本 DataB 续训模型 | 完全不提供相机描述 | 检查提示格式与缺失输入影响 |

#### 快速验收标准

- 正确相机描述的 ACC/Balanced ACC 应高于普通续训模型不提供相机描述。
- 正确相机描述应高于错误相机描述。
- 正确相机描述应高于明确缺失和完全不提供相机描述。
- 约 0.2 至 0.5 个百分点的波动视为结论不足；约 1 个百分点以上且方向一致，才视为值得进行无泄漏正式重训的信号。

#### 结果来源与指标

- 普通续训模型、不提供相机描述：`/tmp/1res/datab_camera_quick_gate/ordinary/no_camera/eval/dataa_no_camera_eval_summary.json`。
- 相机文本续训模型、不提供相机描述：`/tmp/1res/datab_camera_quick_gate/camera_conditioned/no_camera/eval/dataa_no_camera_eval_summary.json`。
- 相机文本续训模型、正确相机描述：`/tmp/1res/datab_camera_quick_gate/camera_conditioned/gold_camera/eval/dataa_gold_camera_eval_summary.json`。
- 相机文本续训模型、错误相机描述：`/tmp/1res/datab_camera_quick_gate/camera_conditioned/shuffled_camera/eval/dataa_shuffled_camera_eval_summary.json`。
- 相机文本续训模型、明确缺失相机描述：`/tmp/1res/datab_camera_quick_gate/camera_conditioned/null_camera/eval/dataa_null_camera_eval_summary.json`。

五个条件均匹配 1001 条记录。该 DataB 诊断集没有成对样本，因此输出中的 `Pair accuracy: 0.00% (0 pairs)` 不适用于本实验。

| 模型与推理输入 | ACC | Balanced ACC | Fake Recall | Fake F1 |
|---|---:|---:|---:|---:|
| 普通续训模型，不提供相机描述 | 97.20% | 97.20% | 97.80% | 97.22% |
| 相机文本续训模型，不提供相机描述 | 97.30% | 97.30% | 97.80% | 97.31% |
| 相机文本续训模型，正确相机描述 | 96.90% | 96.90% | 97.80% | 97.02% |
| 相机文本续训模型，错误相机描述 | 96.90% | 96.90% | 98.20% | 96.94% |
| 相机文本续训模型，明确缺失相机描述 | 96.60% | 96.60% | 97.00% | 96.71% |

正确相机描述相对相机文本模型的无相机输入，ACC 下降 0.40 个百分点、Fake F1 下降 0.29 个百分点；相对错误相机描述，ACC 完全相同。相机文本模型无相机输入相对普通续训模型仅高 0.10 个百分点，属于可忽略波动。

#### 重要限制

这约 1000 条诊断样本曾参与完整 DataB detection checkpoint 的 5-epoch 训练。虽然后续普通/相机文本续训阶段排除了它们，但两个模型继承的初始 checkpoint 已见过这些样本。因此本测试只能作为低成本信号检查，不能作为论文中的无泄漏泛化结果。

#### 验收结论

**未通过。** 正确相机描述没有高于错误相机描述、无相机输入和普通续训对照，不满足预设的任何关键验收条件。高达约 97% 的绝对准确率还受到继承 checkpoint 已见过诊断样本的影响，不能解释为泛化能力；但同一批样本上的条件消融仍足以说明当前模型没有表现出对正确相机文本内容的选择性利用。

### 快速验证后的决策

- 根据预设决策，停止当前“把 camera caption 作为 user prompt 条件进行一次 SFT”的路线，不为该方案投入无泄漏干净重训。
- 该结论不否定使用相机运动能力预训练、结构化运动表征或模型层融合的其他方案，只否定当前直接文本条件注入的实现。

## 8. 相机运动前置感知强化学习最小验证

### 这个实验测什么

从同一个完整 DataB detection checkpoint 出发，一条分支先进行普通 detection replay，另一条分支先通过可计算的多标签奖励学习从视频预测 camera motion；随后两条分支使用完全相同的 DataA detection GRPO 数据。最终检测推理都不提供 camera label/caption，比较 DataA test 与 VIF-Bench，判断内部相机运动能力是否真正迁移到 AIGC 视频检测。

### 初始模型与数据

- 初始 checkpoint：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115/`。
- Camera pretext：只使用 DataA train 中能够匹配官方 camera labels 的 real 记录；DataA test 完全排除。
- 普通对照第一阶段：使用与 camera pretext 数量相同的 DataB detection replay。
- 公共第二阶段：两条分支使用完全相同的 DataA train detection GRPO 数据。
- 检测测试：DataA test 与 VIF-Bench；DataB 不作为无泄漏测试。
- 完整方案：`docs/camera_motion_pretext_rl_minimal_validation_20260711.md`。

### 单一变化因素

第一阶段学习的是 detection replay 还是 camera-motion pretext。第二阶段训练、模型初始化、总步数、超参数和评测 prompt 必须保持一致。camera labels 始终是模型从视频预测的目标，不作为检测输入。

### 验收标准

- camera macro-F1 相对初始 checkpoint 提高至少 10 个百分点；
- 相机分支相对普通检测对照的 DataA Balanced ACC 或 Fake F1 提高至少 1.5 个百分点；
- VIF-Bench ACC/F1 下降不超过 0.5 个百分点；
- 提升不能只来自 `no-motion` 分桶；
- 检测推理不提供任何 camera caption。

### 已知限制与当前状态

该路线不再单独作为论文主线。Camera classification 可以提高相机理解，但没有直接约束模型在同相机运动条件下寻找局部生成差异；同时 DataB camera 分布可能成为真假捷径。现有 P0 工具链保留为后续 camera 增量消融。

2026-07-11 已完成 P0：数据构建器、ms-swift reward plugin、camera 多标签 evaluator、detection motion-bucket 汇总器以及 11 项 reward 单元测试。本地按完整 DataA/DataB 镜像 dry-run 得到 DataA train/test 1506/646 条、camera train/eval 745/322 条、DataB replay 745 条，两个 source-group 交集均为 0，camera train 全部来自 real。完美预测端到端冒烟测试的格式正确率、exact-set accuracy 和 micro-F1 均为 100%。这些是实现验证，不是模型效果结果。

本地审计输出为临时 dry-run 文件，服务器使用既有显式 split 后数量可能变化；以服务器生成的 `camera_pretext_grpo_sets_summary.json` 为准。

### 下一步

先执行第 9 节的真实 mask 数据 Gate。只有局部反事实学习本身通过后，才比较 pair-only 与 camera+pair，决定 camera 能否作为核心贡献。

## 9. 相机匹配局部反事实三门验收

### 这个实验测什么

DataA 的同一 case 中，Real 与 Fake 来自相同源视频、相同全局相机运动和相同编码流程，只有 VACE mask 内发生局部生成。该实验先验证这种局部差异是否真实存在，再训练模型比较 A/B 双顺序并定位编辑区域，最后检查能力是否迁移到不提供 camera、bbox 或配对视频的普通 DataA/VIF-Bench 检测。

### 初始模型与数据

- 后续训练初始 checkpoint：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115/`。
- DataA detection：`/input/workflow_58770161/workspace/test/cameramotion_det/detection/dataa_vace_grounded_cot_instruct_tp8x2_sft_all.json`。
- DataA camera：`/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_v2.jsonl`。
- DataA split：`/input/workflow_58770161/workspace/test/cameramotion_det/tools/data/camera_motion_splits/dataA_train.json` 与 `dataA_test.json`；若 train 文件不存在，从完整 JSON 按 case 排除 test。
- 正式局部区域：VACE grounded-CoT input index 中的 `mask_npz` / `M_gen`；具体服务器路径待 Gate 0 预检补充。
- DataB detection：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`。
- DataB camera：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`，只用于 replay 分层配平，不进入 prompt。
- 完整运行说明：`docs/camera_matched_counterfactual_validation_gates_20260711.md`。

### 实验条件与单一变化因素

- 数据有效性门（Gate 0）：不训练模型，只比较同源 real/fake 在真实 edit mask 内外的像素差异。
- 局部配对门（Gate 1）：每个 held-out case 同时评测 real-first 和 fake-first；user prompt 不提供 bbox/time。训练对照为等步数 detection replay。
- 检测迁移门（Gate 2）：比较等步数检测对照、pair-only、camera+pair 三个分支。三个分支使用相同初始 checkpoint、后续检测数据、评测 prompt 和推理设置。
- Camera 的有效增量是 `camera+pair` 相对 `pair-only`，不能用 `camera+pair` 相对旧 checkpoint 的差异代替。

### 数据处理约束

- DataA train/test 以 case 为单位，real/fake 不得跨 split。
- DataA 训练始终保留完整 real/fake pair；mask 外区域不继承视频级 Fake 局部标签。
- A/B 顺序双写，且训练/评测 prompt 中不出现 GT bbox 或时间范围。
- 自动 CoT bbox fallback 只能做诊断，`formal_gate_eligible=false` 时不能宣布 Gate 0 通过。
- DataB replay 在 `motion dynamics × speed × steadiness` 粗签名内等量抽取 Real/Fake；camera 文本不进入检测输入。
- Real/Fake 图像帧的尺寸、采样、编码和像素上限必须一致。

### 验收标准

- Gate 0：有效 pair、真实 mask 覆盖与 camera 标签覆盖均不低于 90%；在有 camera 标签的 pair 中一致率不低于 98%；mask 内/外差异中位数比值不低于 2.0；mask 外平均绝对差异中位数不高于 0.03；至少 70% pair 的 mask 内差异更大。
- Gate 1：选择准确率不低于 70%；swap consistency 不低于 85%；预测 A 比例在 45% 至 55%；mean bbox IoU 不低于 0.30；选择与 bbox 格式正确率均不低于 95%。
- Pair 迁移：相对等步数 detection replay，DataA Balanced ACC 或 Fake F1 至少提高 3 点；VIF-Bench ACC/F1 各下降不超过 1 点；`minor-motion` 或 `complex-motion` 至少一个指标提高 1 点。
- Camera 贡献：`camera+pair` 相对 `pair-only` 的 DataA Balanced ACC 或 Fake F1 再提高至少 1 点，同时满足 VIF 保留与移动相机分桶条件。

### 已知泄漏与分布限制

- 初始 checkpoint 已见过完整 DataB，因此 DataB 只做 retention/replay，不是无泄漏泛化测试。
- DataA 目前仅有 VACE 局部生成，Gate 通过后仍需第二生成器的小型 held-out local-edit 测试，才能支持局部跨生成器结论。
- 当前 CoT bbox 是自动证据框，不等于真实生成 mask；正式 Gate 只接受 VACE `M_gen`。
- VIF-Bench 主要是全生成视频，不能替代局部编辑测试，但用于检查通用能力是否遗忘。

### 当前状态与下一步

#### 2026-07-11 Gate 0 首次 200 对结果

结果来源：

- 构建 summary：`D:/1codex/camera/cameramotion_videodetection/counterfactual_gate/counterfactual_gate/data/dataa_counterfactual_gate_sets_summary.json`。
- Gate 0 原始 summary：`D:/1codex/camera/cameramotion_videodetection/counterfactual_gate/counterfactual_gate/gate0_200/dataa_counterfactual_signal_gate_summary.json`。
- 逐 pair 明细：`D:/1codex/camera/cameramotion_videodetection/counterfactual_gate/counterfactual_gate/gate0_200/dataa_counterfactual_signal_gate_items.csv`。

| 指标 | 结果 | 原验收线 |
|---|---:|---:|
| 完整 DataA pair | 1076 | - |
| train / held-out test case | 755 / 321 | case 交集为 0 |
| 真实 VACE mask 覆盖 | 1076/1076，100% | ≥90% |
| 200 对有效计算率 | 100% | ≥90% |
| 200 对 camera 标签覆盖率 | 191/200，95.5% | 修正后 ≥90% |
| 有标签 pair 的 camera 一致率 | 191/191，100% | ≥98% |
| mask 内平均绝对差异中位数 | 0.12118 | - |
| mask 外平均绝对差异中位数 | 0.01101 | ≤0.03 |
| mask 内/外差异比中位数 | 9.3429 | ≥2.0 |
| mask 内/外差异比 P10 | 2.8330 | - |
| mask 内差异高于 mask 外的 pair 比例 | 100% | ≥70% |

这次实际测试的是：同一 DataA case 的 Real/Fake 在真实 `M_gen` 内是否发生显著变化，同时 mask 外是否基本保持一致。局部像素信号以较大余量通过，说明 DataA 可以用于同源局部反事实学习；它尚未证明模型能学会该信号，也没有证明 camera 能提升检测。

结论标记：`结论不足`。局部信号本身通过，但首次脚本按排序截取前 200 对，并把 9 个 `camera_labels=[]`、`motion_bucket=unknown` 的 case 错误计入 camera 不一致。原始 summary 的 `status=failed` 保留；2026-07-11 更正为“camera 标签覆盖 95.5%，有标签 pair 一致率 100%”，原因是缺失标签不等于 Real/Fake 标签冲突。

下一步只做低成本复核：拉取修正版后用 `--seed 20260711` 随机抽取 200 对重跑；通过后去掉 `--max-pairs` 跑完整 train Gate 0。两次均通过后再进入配对学习，不直接启动 GRPO。

#### 2026-07-11 Gate 0 随机与全量复核结果

结果来源：

- 随机 200 对 summary：`/tmp/1res/counterfactual_gate/gate0_200_random/dataa_counterfactual_signal_gate_summary.json`。
- 随机 200 对明细：`/tmp/1res/counterfactual_gate/gate0_200_random/dataa_counterfactual_signal_gate_items.csv`。
- 完整 train summary：`/tmp/1res/counterfactual_gate/gate0_train_full/dataa_counterfactual_signal_gate_summary.json`。
- 完整 train 明细：`/tmp/1res/counterfactual_gate/gate0_train_full/dataa_counterfactual_signal_gate_items.csv`。

| 指标 | 固定 seed 随机 200 对 | 完整 train 755 对 | 验收线 |
|---|---:|---:|---:|
| 有效 pair 率 | 100% | 100% | ≥90% |
| 真实 VACE mask 覆盖率 | 100% | 100% | ≥90% |
| camera 标签覆盖率 | 99.5% | 98.81% | ≥90% |
| 有标签 pair 的 camera 一致率 | 100% | 100% | ≥98% |
| mask 内平均绝对差异中位数 | 0.12281 | 0.12467 | - |
| mask 外平均绝对差异中位数 | 0.01212 | 0.01270 | ≤0.03 |
| mask 内/外差异比中位数 | 8.5753 | 8.3310 | ≥2.0 |
| mask 内/外差异比 P10 | 2.4153 | 2.3811 | - |
| mask 内差异高于 mask 外的 pair 比例 | 100% | 99.87% | ≥70% |

两次复核的核心指标接近，说明首次结果不是排序抽样偶然性。完整训练集只有 1/755 个 pair 的 mask 内差异未高于 mask 外，且总体中位数比值仍为 8.33；局部反事实信号稳定存在。

结论标记：`通过`。Gate 0 只建立“DataA 中存在可学习的、真实 mask 对齐的局部差异”，不建立“MLLM 已能利用该差异”或“camera 已能提高检测”。下一步进入 Gate 1 的最小配对学习，对 held-out 321 个 case 做 A/B 双顺序选择、swap consistency 与 bbox IoU 验收；暂不直接运行 GRPO。

#### 2026-07-11 Gate 1 局部配对 DPO

这个实验测试：在不向 prompt 注入 camera 标签、GT bbox 或 GT 时间的情况下，局部同源配对偏好是否能让模型消除 A/B 位置偏置，并从视频中选择局部编辑版本、定位真实 VACE mask。

- 状态：训练前 baseline 与两步 DPO smoke 待执行。
- 初始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 训练数据：`/tmp/1res/counterfactual_gate/data/dataa_counterfactual_dpo_local_only.json`，755 个 train case、3020 条偏好记录。
- held-out 评测：`/tmp/1res/counterfactual_gate/data/dataa_counterfactual_eval_local_only.json`，321 个 test case、642 条双顺序记录。
- 单一训练因素：在相同初始 checkpoint 上加入 local-only LoRA-DPO；训练前 checkpoint 是第一对照，Gate 1 通过后再补等步数 detection replay 迁移对照。
- 主要设置：LlamaFactory、LoRA rank 16、sigmoid DPO、`pref_beta=0.1`、学习率 `5e-6`、1 epoch、16 卡全局 batch 16、ZeRO-2、视觉像素上限 262144。
- 训练输出：`/tmp/1res/gate1_pair_dpo_local_only`；合并模型：`/tmp/1res/gate1_pair_dpo_local_only_merged`。
- 验收标准：选择准确率 ≥70%、swap consistency ≥85%、预测 A 比例 45% 至 55%、mean bbox IoU ≥0.30、两类格式正确率 ≥95%。
- 泄漏约束：train/test 按 case 隔离；每个 train case 两种顺序同时存在；评测 prompt 不包含 GT bbox；camera 标签不进入 local-only prompt。
- 已知限制：Gate 1 只验证配对任务可学习性，不等于普通二分类迁移，不建立 camera 独立贡献。
- 完整执行说明：`docs/gate1_pair_dpo_execution_20260711.md`。
- 下一步：先跑初始 checkpoint 的 642 条 baseline，再跑 64 条数据、2 optimizer step 的 DPO smoke；smoke 通过后才运行完整 1 epoch。

##### 2026-07-11 Gate 1 训练前基线与最终 DPO 结果

结果来源：

- 训练前基线：`/tmp/1res/gate1_pair_eval/Qwen3-VL-8B-gate1-pair-baseline/eval/dataa_counterfactual_pair_gate_summary.json`。
- 最终 DPO：`/tmp/1res/gate1_pair_eval/Qwen3-VL-8B-gate1-pair-dpo/eval/dataa_counterfactual_pair_gate_summary.json`。
- DPO adapter：`/tmp/1res/gate1_pair_dpo_local_only`，包含 `checkpoint-95` 与 `checkpoint-189`。
- 最终合并模型：`/tmp/1res/gate1_pair_dpo_local_only_merged`。

| 指标 | 训练前基线 | 最终 DPO | DPO - 基线 | 验收线 |
|---|---:|---:|---:|---:|
| 选择格式正确率 | 96.57% | 95.17% | -1.40 点 | ≥95% |
| bbox 格式正确率 | 100% | 100% | 0 | ≥95% |
| pair 选择准确率 | 69.31% | 68.85% | -0.47 点 | ≥70% |
| 预测 A 比例 | 39.52% | 39.93% | +0.42 点 | 45%-55% |
| mean bbox IoU | 0.4533 | 0.4482 | -0.0052 | ≥0.30 |
| bbox IoU@0.3 | 68.85% | 68.07% | -0.78 点 | - |
| swap consistency | 55.14% | 55.76% | +0.62 点 | ≥85% |
| 双顺序均正确 | 48.29% | 48.60% | +0.31 点 | - |

各 motion 分桶没有一致改善：`complex-motion` 选择准确率下降 0.70 点，`minor-motion` 下降 1.06 点，`no-motion` 提高 0.82 点。最终模型仍明显偏向预测 B，且 swap consistency 只有约 56%。

这次实际测试的是：固定初始 detection checkpoint、真实 mask 对齐的 local-only 双顺序偏好数据和 LoRA-DPO，能否让模型学会局部编辑选择与定位。它没有测试 camera 增量，因为 camera 标签没有进入 prompt；也没有进入普通 Real/Fake 检测迁移。

结论标记：`未通过`。最终 DPO 没有超过训练前基线，不能继续把该模型用于 Gate 2，也不能据此启动 camera+pair 或 GRPO。下一步先审计 `trainer_log.jsonl` 中 chosen/rejected reward margin 与 reward accuracy，并低成本合并、评测 `checkpoint-95`；若半程同样无提升，则停止当前 synthetic-rejected DPO 配方，重新选择更直接的配对监督或模型错误驱动的偏好数据。

##### 2026-07-11 DPO 训练动态审计

结果来源：

- `/tmp/1res/gate1_pair_dpo_local_only/all_results.json`。
- `/tmp/1res/gate1_pair_dpo_local_only/trainer_log.jsonl`。

| 训练位置 | loss | batch preference accuracy |
|---|---:|---:|
| step 5 / 189 | 0.6922 | 31.25% |
| step 25 / 189 | 0.5998 | 83.75% |
| step 60 / 189 | 0.3445 | 82.50% |
| step 95 / 189，0.50 epoch | 0.3715 | 82.50% |
| step 150 / 189 | 0.3470 | 78.75% |
| step 185 / 189 | 0.3403 | 82.50% |
| 全程汇总 | 0.4103 | - |

完整训练耗时 1449.5 秒，约 24 分 10 秒，189 个 optimizer step 正常完成。日志只输出聚合 `accuracy`，没有单独输出 chosen/rejected reward 与 margin。

训练目标本身确实被优化：loss 明显下降，训练 batch 上 chosen 相对 rejected 的偏好准确率稳定到约 80%-85%。因此最终 Gate 1 无提升不能归因于 adapter 未加载、DPO 没有反向传播或训练没有完成。更符合结果的解释是：当前人工 rejected 很容易在 teacher-forced 序列概率上被排序，但这种排序没有迁移为 held-out 样本上的贪心 A/B 选择、顺序一致性或更好定位。

结论标记维持 `未通过`。为排除后半程退化，只再评测现有 `checkpoint-95`，不追加训练；若半程仍不优于基线，则停止当前 synthetic-rejected DPO 配方，不通过增加 epoch、提高学习率或直接切 GRPO 继续试参。

##### 2026-07-11 半程 checkpoint-95 复核

结果来源：`/tmp/1res/gate1_pair_eval/Qwen3-VL-8B-gate1-pair-dpo-ckpt95/eval/dataa_counterfactual_pair_gate_summary.json`。

| 指标 | 训练前基线 | DPO checkpoint-95 | 最终 DPO checkpoint-189 |
|---|---:|---:|---:|
| 选择格式正确率 | 96.57% | 95.64% | 95.17% |
| pair 选择准确率 | 69.31% | 68.85% | 68.85% |
| 预测 A 比例 | 39.52% | 39.74% | 39.93% |
| mean bbox IoU | 0.4533 | 0.4520 | 0.4482 |
| bbox IoU@0.3 | 68.85% | 68.07% | 68.07% |
| swap consistency | 55.14% | 55.76% | 55.76% |
| 双顺序均正确 | 48.29% | 48.60% | 48.60% |

半程与最终模型在关键决策指标上几乎相同，均没有超过训练前基线；只有不足 1 个百分点的随机级波动。结合训练 loss 与偏好 accuracy 已正常收敛，可排除“后半程退化”和“再减少 epoch 即可恢复”的解释。

结论标记：`未通过`，当前 synthetic-rejected LoRA-DPO 配方正式停止。它说明当前人工构造的错误视频/错误 bbox 序列偏好目标与 held-out 贪心生成决策不一致，不说明 Gate 0 的局部信号不存在。下一项最低成本验证应使用每个顺序唯一正确答案的双顺序直接监督，在同一 held-out Gate 1 上判断模型是否能直接学会该任务；该验证通过前不构造 hard-negative DPO、不进入检测迁移，也不启动 GRPO。


## 10. 相机补偿局部感知轨迹最小验证

### 这个实验测什么

这个实验测试：在输入帧数量、局部 mask 监督和轻量分类头完全一致时，用稠密光流估计并补偿全局相机运动，是否能让局部 DINOv2 patch 轨迹比未补偿版本更可靠地识别局部生成编辑。

### 状态与日期

- 日期：2026-07-12。
- 状态：数据、提取和监督审计均 `通过`；三探针对照为 `未通过`，当前“aligned 局部轨迹直接替代 unaligned 局部轨迹做视频判别”路线停止。
- 完整执行说明：`docs/camera_aligned_local_trajectory_gate_20260712.md`。

### 模型与数据

- 冻结视觉特征：`facebook/dinov2-small`，服务器默认路径 `/home/admin/dinov2-small`；实际内部模型目录也可能为 `/.aistudio/aistudio-modelhub/zeta/f94249_32800136/hugging_face/facebook__dinov2-small`。
- 冻结光流：TorchVision RAFT-Large，权重 `/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth`。
- 备用正式后端：SEA-RAFT，权重 `/home/admin/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/model.safetensors`；第一轮不启用。
- 统一 case/证据数据：`res/dataA_v1/autolabel/dataa_vace_grounded_cot_v4_records_40step_v3.jsonl`。
- 统一 camera 数据：`camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`。
- held-out test 身份：`tools/data/camera_motion_splits/dataA_test.json` 中既有 case ids；重新生成视频不能改变原 test 身份。
- 数据规模：1080 个 Real/Fake cases，由 198 个既有 VACE-14B、714 个新 VACE-1.3B dataset 40-step 和 168 个新 VACE-1.3B textedit 40-step 组成。

### 三个实验条件

1. 全局 ReStraV 风格轨迹：冻结 DINOv2 CLS 特征和 21 维轨迹统计。
2. 局部轨迹但不补偿相机：密集原视频帧、固定坐标 patch 轨迹和总光流。
3. 相机补偿局部轨迹：相同帧和监督，独立估计每个视频的全局运动，对齐 patch 后计算局部轨迹，并使用去除全局相机场后的残余光流。

唯一 camera 增量比较是条件 3 对条件 2；条件 1 只作为现有全局轨迹方法参照。Real/Fake 都单独估计相机运动，配对 Real 的变换不能作为 Fake 输入。GT mask 不参与相机拟合，只参与 train patch 标签和 test 定位评测。

### 主要设置

- 从原视频最高按 8 FPS 采样，不直接把均匀抽取的 16 张图片当连续光流帧。
- 每个窗口 16 帧、步长 8 帧，只在短窗口内累计变换。
- 前后向 RAFT 一致性过滤；主导单应性拟合失败时退化为仿射或平移。
- VACE mask 通过 case manifest 的 canonical-to-source frame mapping 对齐原视频时间，不使用旧版均匀位置近似。
- 三组条件共用 ReStraV 风格的 `64→32→1` 轻量 MLP，只在 DataA train 训练；阈值从 train 内 case-level validation 选择，最终 321 个 test cases 不参与训练和阈值选择。

### 验收标准

提取阶段必须满足：feature 覆盖率至少 95%、正 mask case 率至少 90%、相机拟合 inlier rate 中位数至少 50%、同源 Real/Fake 独立相机估计的角点差异中位数不超过图像对角线 2%，并且 feature 无 NaN/Inf。

方法阶段必须同时满足：相机补偿相对未补偿的整体视频 AUC 提升至少 3 个百分点、`complex-motion` AUC 提升至少 3 个百分点、`no-motion` AUC 下降不超过 2 个百分点，并且 case bootstrap 的 AUC 差值 95% 置信区间下界高于 0。

### 已知风险与不建立的结论

- 新 40-step 数据与旧 DataA 生成结果不同，旧 Gate 0/Gate 1 数值不作为当前数据结果。
- DataA 是局部同源编辑，第一闸门通过也不等于 VIF-Bench 或 DataB 泛化通过。
- camera label 只用于分桶和审计；当前方法使用从视频估计的几何运动，不把 caption 注入 prompt。
- 第一轮不测试 SEA-RAFT 对 RAFT 的替换收益，不测试 Qwen、DataB、SFT、DPO 或 GRPO。
- 未通过前不扩展训练；通过后下一步才是 DataB 弱监督迁移和 Qwen 局部证据注入。

### 2026-07-12 数据/权重预检与两轮 GPU smoke

结果来源：

- manifest summary：`/tmp/1res/camera_flow_probe_40step_v3/data/dataa_camera_flow_probe_manifest_40step_v3_summary.json`；
- 权重预检：`/tmp/1res/camera_flow_probe_40step_v3/weight_preflight.json`；
- smoke 提取审计：`/tmp/1res/camera_flow_probe_40step_v3/smoke/extraction_audit.json`；
- smoke 可视化：`/tmp/1res/camera_flow_probe_40step_v3/smoke/visualizations/`。

严格 manifest 和三个离线权重全部通过。最终数据为 1080 cases，来源数量分别为 198、714、168；train/test 为 759/321；camera 标签无缺失，旧 test case 均存在于新记录中。

| 首轮 smoke 指标 | 结果 | 验收线 |
|---|---:|---:|
| 有效 feature cases | 6/6，100% | ≥95% |
| 可映射出正 mask patch 的 cases | 6/6，100% | ≥90% |
| 相机拟合 inlier rate 中位数 | 94.25% | ≥50% |
| 相机拟合 inlier rate P10 | 58.25% | - |
| Real/Fake 相机角点差异中位数/图像对角线 | 0.0701% | ≤2% |
| Real/Fake 相机角点差异 P90/图像对角线 | 0.2337% | - |
| 非有限 feature 文件 | 0 | 0 |

6 个 case 的 GPU 前向全部成功，首个包含模型初始化耗时 24.0 秒，随后单 case 为 2.6 至 4.7 秒。三个 motion bucket 的拟合质量均通过，说明 TorchVision RAFT、DINOv2、本地权重加载、原视频采样和精确 mask 时间映射在这些样本上可以执行。

但审计中的 `source_counts` 只有 `vace14b_reused: 6`。这次没有实际读取新生成的 714 个 dataset 40-step 或 168 个 textedit 40-step case，因此不能建立“完整 40step_v3 数据链路已经通过”。它也没有训练三组 MLP，不能说明相机补偿已经改善检测。

结论标记：`结论不足`。数据/权重预检本身为 `通过`；首轮 smoke 仅建立旧 VACE-14B 上的工程与几何可执行性。已于 2026-07-12 修正 smoke 选择规则为每个“最终视频来源 × motion bucket”取 1 个 train case，修正原因是原规则只按 motion 排序抽样，意外被 case id 更靠前的 VACE-14B 占满。

#### 来源×运动联合分层复核

修正后共选择 9 个 train cases，覆盖三个最终视频来源与三个 motion bucket 的全部 9 个组合；每个来源 3 个、每个 motion bucket 3 个。结果仍来自同一个 `extraction_audit.json`，该文件已被本轮复核结果覆盖。

| 分层 smoke 指标 | 结果 | 验收线 |
|---|---:|---:|
| 来源×运动组合覆盖 | 9/9 | 9/9 |
| 有效 feature cases | 9/9，100% | ≥95% |
| 可映射出正 mask patch 的 cases | 9/9，100% | ≥90% |
| 相机拟合 inlier rate 中位数 | 90.12% | ≥50% |
| 相机拟合 inlier rate P10 | 66.17% | - |
| Real/Fake 相机角点差异中位数/图像对角线 | 0.1093% | ≤2% |
| Real/Fake 相机角点差异 P90/图像对角线 | 0.6227% | - |
| 缺失或非法 feature cases | 0 | 0 |

三个 motion bucket 的相机拟合内点率中位数均约为 89% 至 91%；其中 complex-motion 的 Real/Fake 相机角点差异 P90 为 1.3244%。P90 没有单独的正式验收线，这里只将其作为尾部质量参考；正式 2% 门槛作用于整体中位数。自动化分层 smoke 标记为 `通过`，说明最终 40step_v3 三类视频来源都能稳定进入当前提取链路。

这个结果不建立“相机补偿提高 AIGC 检测”的结论，因为三组 MLP 尚未训练和比较；方法效果仍标记为 `结论不足`。此外，可视化文件虽然已成功生成，但本轮提供的结果中没有人工检查记录。

2026-07-12 更正：上一轮记录的“分层 smoke 待复核”更新为“自动审计通过”；更正依据是修正后的 9 个来源×运动组合已全部运行且所有自动门槛通过。历史首轮 6-case 结果保留，不静默覆盖。

#### 三类视频来源的 Real/Fake 人工可视化复核

结果来源：`D:\1codex\camera\cameramotion_videodetection\vis\vis\` 中三个来源各一组 Real/Fake 面板，共 6 张 JPG。

| 来源与样本 | 人工检查结果 |
|---|---|
| 旧 VACE-14B：`dataA_v1_00006` | 补偿后的下一帧与当前帧方向一致；GT mask 覆盖实际编辑主体；未见全画面反向平移或 mask 坐标错位 |
| 新 VACE-1.3B dataset：`dataA_v1_dataset_v2_000001` | 狗主体的 mask 与编辑区域对齐，Real/Fake 独立补偿结果均稳定；草地纹理产生广泛 photometric residual，但未表现为坐标翻转 |
| 新 VACE-1.3B textedit：`dataA_v1_textedit_reserve_000002` | 中央被编辑球员与 mask 对齐，补偿方向正确；其他真实运动球员仍保留明显残余光流 |

人工可视化标记为 `通过`。它确认当前几何方向和 mask 映射可以进入全量提取，但也显示相机补偿不会自动消除独立物体运动。因此后续不能把残余光流幅值直接作为 Fake 分数，必须在相同监督和分类器下比较“局部未补偿”与“局部已补偿”两组特征。

2026-07-12 更正：索引状态从“人工可视化待核”更新为“分层 smoke 通过”；更正依据是三类来源的 Real/Fake 可视化已人工核对，未发现方向、全局错位或 mask 映射错误。

### 2026-07-12 全量特征提取审计

结果来源：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_flow_probe_40step_v3/full_extraction_audit.json`。

| 全量提取指标 | 结果 | 验收线 |
|---|---:|---:|
| 有效 feature cases | 1080/1080，100% | ≥95% |
| 有正 mask patch 的 cases | 1072/1080，99.26% | ≥90% |
| 相机拟合 inlier rate 中位数 | 93.10% | ≥50% |
| 相机拟合 inlier rate P10 | 59.76% | - |
| Real/Fake 相机角点差异中位数/图像对角线 | 0.1180% | ≤2% |
| Real/Fake 相机角点差异 P90/图像对角线 | 0.8082% | - |
| 缺失或非法 feature cases | 0 | 0 |

三个来源均完整覆盖 198、714、168 个 case。complex-motion、minor-motion 和 no-motion 的相机内点率中位数分别为 89.36%、97.09% 和 98.59%；对应 Real/Fake 相机角点差异中位数分别为 0.1804%、0.1050% 和 0.0176%。正式提取门标记为 `通过`，说明当前几何与特征链路可以覆盖完整 40step_v3 数据。

该结果仍不建立检测收益。另有 8 个 case 没有任何正的 aligned mask patch；总体比例满足门槛，但在明确它们属于 train/test、真实小区域低于 patch 阈值，还是 mask 时间映射问题之前，不进入三组 MLP 训练。

#### 无正 aligned patch 的 8 个 case 诊断

增强审计结果仍来自 NAS 的 `full_extraction_audit.json`；大型逐 case 特征目录为 `/tmp/1res/camera_flow_probe_40step_v3/full/features`，大小 849M。

| 诊断维度 | 结果 |
|---|---:|
| train/test | 6 / 2 |
| 旧 VACE-14B / dataset 40-step / textedit 40-step | 4 / 3 / 1 |
| complex-motion / minor-motion | 7 / 1 |
| unaligned 最大 mask patch 覆盖率 | 8/8 均为 1.0 |
| unaligned 正 patch 数 | 每 case 43 至 308 |
| aligned 最大 mask patch 覆盖率 | 8/8 均为 0.0 |

诊断标记为 `结论不足`。它排除了“编辑区域太小、10% patch 阈值过高”：原坐标中的 mask 完整且正 patch 很多，但变换到窗口 anchor 后整体消失。由于问题集中于强运动样本、而 smoke 的几何方向人工检查正常，当前更像是首帧 anchor 有效视野下的整块出界或少数累计变换失效，尚不能直接判定为全局方向写反。

同时发现旧 audit 的 1072/1080 只统计 `fake_label_aligned`，而探针训练实际使用 `fake_label_aligned AND fake_valid_aligned`。因此该数值不能作为最终有效监督覆盖率。2026-07-12 更正：保留原始 1072/1080 结果作为历史输出，但将“全量提取通过”收紧为“文件与相机几何提取通过，局部有效监督待复核”；更正原因是审计口径未与训练选择条件一致。代码已改为报告有效正 patch、aligned 有效 case 率和有效 patch 比例。

#### 与探针训练一致的最终监督审计

| 修正口径指标 | 结果 | 验收线 |
|---|---:|---:|
| 有有效正 aligned patch 的 cases | 1068/1080，98.89% | ≥90% |
| train 有效正监督 | 749/759，98.68% | ≥95%（探针共同训练覆盖率） |
| test 有效正监督 | 319/321，99.38% | ≥95%（共同定位覆盖率） |
| 至少有一个 aligned 有效 patch 的 cases | 1079/1080，99.91% | ≥95% |
| aligned 有效 patch 比例中位数 | 92.80% | - |
| aligned 有效 patch 比例 P10 | 59.74% | - |

修正口径后的监督审计标记为 `通过`。无有效正 aligned patch 的 case 从 8 个增至 12 个，其中 10 个 train、2 个 test。新增 4 个 case 虽有 raw aligned 正标签，但正标签全部位于 aligned 无效边界；只有 `dataA_v1_dataset_v2_000250` 整个 case 没有 aligned 有效 patch。12 个 case 占 1.11%，问题高度集中于 complex-motion，整体有效域分布仍健康，因此不修改 anchor、不重算全量特征。

为避免选择偏差，正式探针采用以下固定规则：

1. 全局轨迹、`local_unaligned` 与 `local_aligned` 三个探针统一使用同时具有两种局部有效正监督的 749 个 train cases。
2. 主视频级 Real/Fake AUC、分 motion AUC、bootstrap 差值和定位指标统一在共同 319 个 held-out test cases 上计算，并报告覆盖率 319/321。
3. 同时自动计算完整 321 个 held-out test 的视频指标作为敏感性结果，但不参与主 gate 判定；用于检查剔除两例是否改变结论方向。
4. camera 唯一增量仍只看 aligned 对 unaligned；全局轨迹只作为已有方法参照。

2026-07-12 更正：索引状态从“局部有效监督待复核”更新为“全量提取与有效监督审计通过”；更正依据是与探针训练一致的有效 patch 统计仍达到 98.89%。最终采用统一剔除协议：三个探针共同训练 749 cases、主测试共同 319 cases，并补完整 321-test 敏感性结果；原因是仅 12/1080 为几何支持例外，统一透明剔除比返工 anchor 更符合当前时限和实验可解释性。

### 2026-07-12 三探针对照结果

结果来源：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_flow_probe_40step_v3/probe/camera_aligned_local_probe_summary.json`。

主结果使用共同 749 个 train cases 和共同 319 个 held-out test cases；完整 321-test 敏感性结果方向一致。

| 主测试视频指标 | 全局轨迹 | 局部未补偿 | 局部已补偿 | 已补偿减未补偿 |
|---|---:|---:|---:|---:|
| 整体 AUC | 64.59% | 56.90% | 54.83% | -2.07 点 |
| complex-motion AUC | 67.81% | 58.62% | 55.23% | -3.39 点 |
| minor-motion AUC | 59.44% | 58.31% | 56.00% | -2.31 点 |
| no-motion AUC | 64.28% | 53.02% | 53.40% | +0.38 点 |
| pair accuracy | 75.55% | 75.86% | 72.10% | -3.76 点 |

完整 321-test 的局部未补偿/已补偿整体 AUC 为 56.86%/54.75%，差值 -2.11 点，与主结果一致。aligned-minus-unaligned 的 case bootstrap AUC 差值均值为 -2.07 点，95% CI 为 `[-3.14, -1.06]` 点，完整位于 0 以下；这说明检测退化不是抽样波动。

| 共同 319-test 定位指标 | 局部未补偿 | 局部已补偿 | 差值 |
|---|---:|---:|---:|
| patch AUC | 68.04% | 70.09% | +2.05 点 |
| patch IoU | 22.74% | 25.31% | +2.58 点 |
| pointing game | 50.47% | 44.20% | -6.27 点 |

结论标记：`未通过`。相机补偿改善了一部分 patch 级空间排序和 IoU，但直接用已补偿局部 patch 分数聚合视频判别时，整体与复杂运动 AUC 显著下降，并出现 Fake recall 86.83%、Real recall 18.81% 的强 Fake 偏置。因此该结果不能支持“相机补偿局部轨迹提高 AIGC 视频检测”，也不进入 DataB、Qwen、SFT 或 RL 扩展。

随后执行了一个严格限时的低成本复核：保留更强的全局轨迹判别，仅把 aligned/unaligned 局部分数作为辅助，分别做相同容量的验证集拟合分数融合。它只使用现有 849M 特征，不重跑 RAFT/DINO；预先约定若 `global+aligned` 不能稳定优于 `global+unaligned` 和 global-only，则停止当前几何相机补偿主路线。

正式特征已由用户确认上传至 `oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_flow/camera_flow_probe_40step_v3/full/features/`。上传对象数量和远端总字节数未单独提供，不作猜测。

2026-07-12 融合首次执行未形成结果：服务器部署目录不是 Git 仓库，`git pull` 失败，随后仍运行旧版 `train_probe.py`；旧 summary 没有 `fusion_gate`，读取时报 `KeyError: 'fusion_gate'`。该事件标记为环境部署问题，不是融合实验失败，不记录融合指标。

#### 低容量分数融合复核结果

更新服务器文件后重新执行成功，结果仍保存在同一 `camera_aligned_local_probe_summary.json` 的 `fusion_gate` 字段。

| 主 319-test 视频 AUC | 结果 | 相对 global-only |
|---|---:|---:|
| global-only | 64.59% | - |
| global + unaligned | 64.77% | +0.19 点 |
| global + aligned | 64.40% | -0.18 点 |

| complex-motion AUC | 结果 | 相对 global-only |
|---|---:|---:|
| global-only | 67.81% | - |
| global + unaligned | 68.35% | +0.53 点 |
| global + aligned | 67.81% | +0.00 点 |

验证集选择的局部非负权重为 unaligned `0.35`、aligned `0.27`。主测试上 `global+aligned` 相对 `global+unaligned` 的整体/复杂运动 AUC 差值为 -0.37/-0.53 点；bootstrap 均值 -0.38 点，95% CI 为 `[-0.89, +0.12]` 点。`global+aligned` 相对 global-only 的 AUC 差值 -0.18 点；bootstrap 95% CI 为 `[-0.93, +0.50]` 点。完整 321-test 敏感性结果方向一致：global、global+unaligned、global+aligned AUC 分别为 64.54%、64.70%、64.30%。

融合结论标记：`未通过`。`global+aligned` 没有优于 global-only 或 unaligned 融合，complex-motion 也没有增量，五项预设验收全部失败。该结果说明 aligned 局部特征的 patch AUC/IoU 小幅提升不能转化为视频检测的独立增量。

2026-07-12 最终收敛：当前几何相机补偿局部轨迹路线停止。停止范围包括继续调整融合权重、top-patch 聚合、anchor、RAFT/SEA-RAFT 和局部 MLP；原因是直接检测显著负增量，保留全局判别后的低容量融合也无正增量。已有特征、审计和定位结果保留为消融或负结果，不扩展到 DataB、Qwen、SFT、DPO 或 GRPO。

### 立即下一步

暂停当前几何相机补偿路线，不继续追加试参。下一步重新选择能够把 camera 能力接入 AIGC 检测的主方法时，必须以新的低成本 gate 重新立项；本轮只保留“camera compensation 改善部分 patch 排序但不改善视频检测”的经验事实。


## 11. 相机分层同源配对独立判别最小验证

### 这个实验测什么

验证同一 DataA case 的 Real/Fake 视频分别独立计算 `Real/Fake` verdict 分数时，在普通二分类损失上增加 `Fake 分数高于 Real 分数` 的配对 margin，是否能比等数据量、等训练步数的普通二分类续训更好地识别局部编辑视频。

### 状态与日期

- 日期：2026-07-12。
- 状态：`未通过，当前配对排序配方停止`。
- 完整执行说明：`docs/caspr_gate1_execution_20260712.md`。

### 模型与数据

- 初始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- DataA detection：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`。
- DataA camera labels：`/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`。
- DataA 训练：从固定 train 身份中按视频来源与 `no-motion/minor-motion/complex-motion` 联合分层选择 256 个完整 Real/Fake pairs。
- DataA 评估：固定 321 个旧 test 身份。由于它们已被多次用于项目诊断，本实验只称其为开发集，不称为未经使用的最终测试集。
- DataB replay：从 `v4vif_2766busterall_trainall.json` 选择 512 条 Real/Fake 平衡样本。
- 泄漏限制：初始 checkpoint 已看过完整 DataB，因此 DataB 不能作为 held-out 测试；本轮只用 VIF-Bench 检查通用检测保留。

### 唯一改变因素

- 普通独立判别续训对照：DataA pair 中两个视频分别计算二分类损失，DataB 计算相同二分类 replay loss。
- 相机分层同源配对排序方法：数据、prompt、batch、步数和二分类损失完全相同，仅在 DataA pair step 增加权重 0.2、margin 0.5 的 `Fake > Real` score ranking loss。
- Real/Fake 不放进同一个 A/B prompt；两条序列独立编码，pair 关系只进入 loss。
- detection prompt 不注入 camera caption、GT bbox 或 GT mask；当前也不使用 RAFT/DINO 特征。

### 主要设置

- Qwen3-VL-8B 的语言层 LoRA rank 32、alpha 64、dropout 0.05；基础权重和视觉塔保持冻结。
- 16 卡、每卡一个 pair 或一个 replay 样本、梯度累积 1、学习率 `2e-5`。
- 总计 64 optimizer steps：32 个 DataA pair steps 和 32 个 DataB replay steps 交替执行。
- 只训练位于 CoT 之前的短 verdict 分数，不让 GT CoT 内容参与真假 score。

### 验收标准

- 相对普通对照，DataA 开发集整体视频 AUC 提升至少 3 个百分点。
- Pair accuracy 提升至少 5 个百分点。
- `complex-motion` AUC 提升至少 3 个百分点。
- 任一 VACE 视频来源的 AUC 下降不超过 2 个百分点。
- DataA 门通过后，VIF-Bench 相对对照下降不超过 1.5 个百分点。

### 已知限制与下一步

该门只验证配对排序能否接入普通检测，不建立 camera pretext 的独立贡献，也不评价解释文本改善。先执行数据构建和单卡两步 smoke；DataA 与 VIF 保留同时通过后，才运行正确 camera labels、打乱 camera labels 和无 camera pretext 的等算力对照。若配对排序门未通过，则停止当前配方，不启动 camera pretext、DPO 或 GRPO。

### 2026-07-12 数据构建与单卡 smoke 结果

结果来源：`/tmp/1res/caspr_gate1/data/caspr_gate1_data_summary.json`、`/tmp/1res/caspr_gate1/smoke/pair_rank/all_results.json` 和 `trainer_log.jsonl`。

| 检查项 | 结果 |
|---|---:|
| 完整 DataA Real/Fake pairs | 1080 |
| 固定 dev pairs | 321 |
| 首次显式 train JSON 覆盖 | 746 |
| 预期 train 补集 | 759 |
| 选中 Gate 训练 pairs | 256 |
| DataB replay | 512，Real/Fake 各 256 |
| train/dev 交集 | 0 |
| `Real/Fake` candidate token | 均为单 token，ID 8800/36965 |
| 两步 smoke | 正常完成，pair 与 replay 均有有限 loss 和非零梯度 |

Smoke 的 DataA pair step 为 loss 1.4200、binary loss 1.2252、pair loss 0.9741、梯度范数 16.72；DataB replay step 为 loss 0.00861、梯度范数 0.429。该结果只建立工程链路可训练，不建立检测收益。首次构建读取旧 `dataA_train.json` 后只有 746 个 eligible train pairs；与最终 1080 减固定 321 dev 应有 759 train 不一致。

2026-07-12 更正：正式训练默认不再读取旧 `dataA_train.json`，统一用 1080 个完整 pair 减去固定 321 dev 得到 759 train，再按来源与 motion bucket 选择 256 pairs。更正原因是旧 train JSON 未覆盖 13 个新 40step_v3 case；首次 smoke 不作为训练结果，无需保留其旧抽样，重新运行 `STAGE=build` 即可。

### 2026-07-12 正式 64 步对照结果

结果来源：`/tmp/1res/caspr_gate1/eval/caspr_gate1_dataa_summary.json`。控制组与方法组均完整覆盖 321 个开发 pairs、642 个独立视频。

| 指标 | 初始 detection checkpoint | 普通独立判别续训对照 | 同源配对排序方法 | 方法减对照 | 验收线 |
|---|---:|---:|---:|---:|---:|
| 整体视频 AUC | 59.49% | 60.46% | 60.92% | +0.46 点 | 至少 +3 点 |
| Balanced accuracy@0 | 51.87% | 58.26% | 57.48% | -0.78 点 | - |
| Real recall@0 | 98.13% | 63.24% | 61.99% | -1.25 点 | - |
| Fake recall@0 | 5.61% | 53.27% | 52.96% | -0.31 点 | - |
| Pair accuracy | 64.17% | 60.75% | 62.31% | +1.56 点 | 至少 +5 点 |
| 平均 Fake-Real margin | 0.743 | 0.269 | 0.336 | +0.067 | - |
| Complex-motion AUC | 待补充 | 60.53% | 60.80% | +0.27 点 | 至少 +3 点 |

三个视频来源的 AUC 差值均为很小的正值：dataset 40-step `+0.54` 点、textedit 40-step `+0.25` 点、VACE-14B `+0.94` 点，因此“任一来源下降不超过 2 点”是唯一通过的方法效果检查。整体 AUC 差值的 1000 次 case bootstrap 均值为 `+0.45` 点，95% CI 为 `[-0.11, +1.05]` 点，跨越 0。

这次实际测试的是：在相同初始 checkpoint、相同 256 个 DataA train pairs、512 条 DataB replay、verdict prompt、LoRA 容量和 64 optimizer steps 下，额外的同源 Fake-vs-Real score margin 是否带来检测增益。方法组平均 pair margin 高于普通对照，说明 pair loss 已经作用于模型；但整体、配对和复杂运动三项增益均远低于验收线，且 pair accuracy 仍低于初始 checkpoint，不能解释为训练不足或损失未生效。

结论标记：`未通过`。当前权重 0.2、margin 0.5 的独立 verdict 配对排序配方正式停止，不追加 loss 权重、margin、epoch 或 LoRA rank 试参。由于第一道 DataA 门失败，不合并模型、不运行 VIF-Bench 保留测试，也不据此启动 camera pretext、DPO 或 GRPO。该结果不否定 camera motion 可能与检测有关，只否定“依靠当前短 verdict 同源排序把 camera 分层监督接入检测”是一个值得在当前期限继续扩展的机制。

## 记录维护说明

- 新实验开始时先在本文件新增中文实验定义和验收标准。
- 用户提供结果后，在对应小节补充指标、结论和下一步，不创建含义重复的新代号章节。
- 未知值保留为 `待补充`，不根据上下文猜测。
- `docs/final_experiment_plan_20260708.md` 是受保护文件，不在本记录维护过程中修改。
