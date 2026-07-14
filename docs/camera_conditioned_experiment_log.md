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
| 2026-07-12 | 正确相机能力学习与检测迁移闭环验证 | 阶段一未通过，当前 camera-label SFT 前置路线停止 | 先确认模型从视频学到正确相机运动，再检验该能力能否在无相机文本推理时迁移到局部编辑检测 | 四轮 correct 的 bucket balanced accuracy 仅 33.25%–35.98%，预测 266–283/321 为 complex-motion；确认多数类塌缩，不进入阶段二 |
| 2026-07-13 | DataA 平衡二元相机问答与视觉依赖门 | 通过；两个模型起点均完成 | 把每个相机 primitive 拆成平衡 Yes/No 问题，验证通用起点和检测起点能否从原视频真正学到相机运动，而不是背标签先验 | 通用/检测起点最终 macro AP 分别为 83.52%/81.70%，均通过视觉控制；没有证据支持检测 SFT 造成灾难性相机能力遗忘，主线继续使用检测起点并进入检测保留诊断 |
| 2026-07-13 | 相机二元问答适配器的原检测提示词保留诊断 | 未通过；确认 Yes/No 接口接管 | 只训练 camera VQA 的 LoRA 挂回检测模型后，在无 camera 文本的原检测任务中是否明显损伤 DataA 检测和解释格式 | 原模型格式有效率 99.84%，camera 模型 642/642 均无法解析为 Real/Fake；原始回复是字面 `Yes/No`，确认 camera 单任务 LoRA 覆盖检测输出契约，后续必须联合混入 detection replay |
| 2026-07-13 | VIF-Bench 相机适配器外部分布检测保留诊断 | 结论不足；Base 基线完成，Camera 未执行 | 同一 camera-only LoRA 是否在无 camera 文本的 VIF-Bench 原检测协议中损伤全生成视频检测能力 | 严格同提示词 Base 基线为 Balanced ACC 79.18%、Fake F1 80.47%；Camera 因半成品合并模型加载失败而无预测，且 DataA 已确认接口接管，不再为该顺序配方补跑全量 VIF |
| 2026-07-13 | 同 16 帧二元相机辅助与检测回放联合训练三分支验证 | 代码已就绪，待服务器执行 | 在相同 16 帧输入和检测回放下，正确二元相机监督是否比逐条翻转监督及仅检测对照学到视觉相关能力，同时保留检测接口 | 尚无模型结果；已完成 40step_v3 case split、全量平衡二元 VQA、1:1 检测 replay、三分支等量训练和视觉/RL 前置验收代码 |
| 2026-07-13 | DataB 自动解释的 DeepfakeJudge-7B 可靠性门 | 代码已就绪，待服务器执行 | 专用开源深伪解释 Judge 在 DataB 上是否真正依据有序帧、bbox、时间和类别评价自动 CoT，而不是只评价语言流畅度 | 先做 200 条分层样本及视觉错配控制；通过后才进入人工校准和全量筛选 |
| 2026-07-14 | 完整 DataB 检测模型的 Skyra 风格 GRPO 奖励动力学诊断 | 已完成；诊断链路通过，原奖励不宜直接延长 | 从既有检测 checkpoint 出发，比较论文式奖励及其问题消融如何改变正确率、真假偏置、证据计数和组内奖励方差 | 100 步训练稳定且无 Fake 单边塌缩；总奖励增长约 56% 来自证据计数项，后 10 步零方差组升至 65.63%，原配方出现计数驱动和有效 GRPO 信号衰减 |

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

## 12. 正确相机能力学习与检测迁移闭环验证

### 这个实验测什么

先验证检测模型能否直接从视频帧学会正确的全局相机运动标签，再检验该能力在检测推理不提供相机文本时，能否相对无相机前置学习和错配相机前置学习稳定提高局部编辑 AIGC 视频检测。

### 状态与日期

- 日期：2026-07-12。
- 状态：`阶段一未通过；当前 camera-label SFT 前置路线停止，不进入检测迁移`。
- 执行说明：`docs/camera_pretext_transfer_validation_20260712.md`。

### 模型与数据

- 初始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- DataA detection：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`。
- DataA camera labels：`/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`。
- 固定 DataA 开发身份：`/input/workflow_58770161/workspace/test/cameramotion_det/tools/data/camera_motion_splits/dataA_test.json`，预计 321 个 case；这些身份已被项目多次用于诊断，因此只称开发集，不称真正 held-out 最终测试。
- 相机训练身份：最终完整 DataA case 减固定开发身份，只使用 real 视频；准确可用条数待数据构建审计后补充，camera label 缺失项不会猜测或补标。
- 阶段二 DataA train pairs：`/tmp/1res/caspr_gate1/data/dataa_train_pairs_256.jsonl`。
- 阶段二 DataB replay：`/tmp/1res/caspr_gate1/data/datab_replay_512.jsonl`。
- 阶段二 DataA dev pairs：`/tmp/1res/caspr_gate1/data/dataa_dev_pairs.jsonl`。
- 无相机前置学习分支既有分数：`/tmp/1res/caspr_gate1/scores/pair_rank`。

### 唯一改变因素与对照

- 阶段一正确相机分支：每个 real 视频使用一个统一 canonical prompt，SFT 目标为规范 camera label JSON list。
- 阶段一错配相机分支：视频、prompt、步数、优化器和每条目标标签数完全相同；在抖动、运动强度、速度、方向和 tracking 语义组内执行固定合法标签置换，且不允许原目标保留。由于重复最多的完整 label set 超过样本半数，“完整 set 总体分布不变且逐条不同”的严格 derangement 数学上不可行；本对照明确牺牲标签名边缘频率相等性，换取零正确目标。
- 阶段一基础模型对照：不训练直接在相同开发 prompt 上生成 camera labels。
- 阶段二三分支使用完全相同的 DataA pair-rank 与 DataB replay 训练；唯一继承差异为无相机前置学习、正确相机前置学习或错配相机前置学习。
- 阶段二训练和推理 prompt 均不加入 camera caption、camera labels、bbox、mask 或光流特征。
- 正确相机与错配相机分支总更新步数相同，是判断正确相机监督内容是否有效的严格对照；无相机前置学习分支少 48 个相机 SFT steps，只作为现有方法基线，其差值不能单独归因于相机标签。

### 主要设置

- 阶段一采用 LoRA rank 32、alpha 64、dropout 0.05，学习率 `1e-5`，16 GPU，每卡 1 个视频，先运行 48 optimizer steps，并保存 step 24/48。
- 同一视频不复制成 CameraBench 式多个 prompt；主训练和主评测只使用一个统一 canonical prompt。
- 阶段一通过后，额外用一个未训练的同义改写 prompt 做鲁棒性诊断，但不以该诊断替代主指标。
- 阶段二从选定相机 LoRA 继续训练，不合并完整模型；沿用 64 步 pair-rank 配方、学习率 `2e-5`、32 个 DataA pair steps 与 32 个 DataB replay steps。
- 本轮是低成本验证，产物位于 `/tmp/1res/camera_pretext_transfer_gate`，无需放 NAS 或上传 OSS。

### 验收标准

- 阶段一：正确相机分支的支持标签 macro-F1 同时比基础模型和错配相机分支至少高 10 个百分点；格式有效率至少 95%；coarse motion bucket accuracy 至少 50%；预测覆盖率至少 99%。
- 阶段二：正确相机分支相对无相机前置学习和错配相机前置学习均需满足整体 AUC 至少 `+2` 点、pair accuracy 至少 `+3` 点、complex-motion AUC 至少 `+2` 点，且至少两个视频来源 AUC 为正增益。
- 只有阶段二 DataA 门通过才运行 VIF-Bench；相对无相机前置学习分支允许最多下降 1.5 个百分点。

### 已知限制与立即下一步

- 初始 detection checkpoint 已看过完整 DataB；DataB replay 只用于能力保留，不能作为 held-out 证明。
- 固定 DataA 开发身份已参与多轮方案诊断，最终论文仍需要独立保留集或明确称为开发消融。
- 阶段一的 camera labels 来自 CameraBench 标签体系，只建立相机能力是否可学；只有阶段二 correct 同时超过 no-pretext 与 shuffled 才建立相机监督的检测迁移证据。
- 已完成数据构建、smoke 和 step 24/48 学习曲线。48 步未通过时不执行阶段二；只有 correct 的语义学习曲线仍上升且与 shuffled 差距扩大，才按预案补到总计 96 步。96 步仍未通过后，不再追加 GRPO、DPO、训练轮数或 prompt-side camera 文本。

### 2026-07-13 阶段一 24/48 步结果

结果来源：

- `/tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_gate_step_24.json`
- `/tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_gate_step_48.json`

321 个固定开发 case 均完成预测。基础模型无法遵循新 camera 输出格式，其指标只作为零训练参照；判断正确 camera 监督内容是否有效，主要比较等算力 correct 与 shuffled 分支。

| checkpoint | 分支 | 格式有效率 | Exact set | Motion bucket ACC | Micro-F1 | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| step 24 | 正确相机标签 | 96.57% | 1.87% | 21.50% | 32.03% | 17.78% |
| step 24 | 错误语义置换标签 | 96.26% | 0.00% | 20.25% | 25.66% | 18.08% |
| step 48 | 正确相机标签 | 91.90% | 1.25% | 28.66% | 38.71% | 20.92% |
| step 48 | 错误语义置换标签 | 86.92% | 0.00% | 24.61% | 22.26% | 16.33% |

step 48 的 correct 相对 shuffled：格式有效率 `+4.98` 点、motion bucket accuracy `+4.05` 点、micro-F1 `+16.45` 点、macro-F1 `+4.58` 点。correct 从 step 24 到 48：motion bucket accuracy `+7.17` 点、micro-F1 `+6.68` 点、macro-F1 `+3.13` 点；同期 shuffled 的 micro-F1 和 macro-F1 分别下降 `3.40` 和 `1.75` 点。

结论标记：`结论不足，当前阶段一未通过`。它没有达到预设的 95% 格式、correct-vs-shuffled macro-F1 `+10` 点和 50% motion bucket accuracy，不能进入阶段二。但 correct 的三项语义指标仍同步上升，且与 shuffled 的差距从 step 24 到 48 扩大，不属于“正确监督与错误监督同样无效”的形态。按照实验开始前约定的学习曲线规则，只允许补一轮到总计 96 步；96 步仍未通过就停止该路线。

同时记录一个训练实现更正：首轮 SFT 只监督 `<camera_motion>...</camera_motion>` 内容，没有把 chat template 的 assistant 结束标记纳入 loss，这与 step 24→48 格式有效率下降一致。2026-07-13 起，correct 与 shuffled 续训均以相同方式监督目标之后的 assistant 结束标记；这是格式终止监督修复，不改变 camera 标签内容。为保持历史可追溯，以上 24/48 原始结果不覆盖。

2026-07-13 方案更正：不再把“累计 96 步的分段续训”作为最终阶段一判断，改为 correct/shuffled 均从原始 detection checkpoint 使用修正后的结束标记监督，连续训练固定最多 192 steps，并在 48/96/144/192 保存。更正原因是按约 750 条训练视频、16 GPU、每卡 batch 1 计算，48 steps 已约等于一个 epoch；旧结果说明一轮未达标但 correct 仍未平台，固定观察到四轮比把两轮当作理论学习上限更合理。选择规则预先固定为“最早通过全部检查的 checkpoint”，四轮均不过即停止，避免继续追加 epoch 或事后挑最高点。旧的 96 步分段续训说明保留为历史过程，但不再执行。

### 2026-07-13 干净四轮学习曲线结果

结果来源：`/tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_clean_4epoch_curve.json`。321 个开发 case 在所有 checkpoint 上均完整匹配。

| 累计 step / 约 epoch | 自动状态 | Correct 格式 | Correct Exact set | Correct bucket ACC | Correct Micro-F1 | Correct Macro-F1 | Correct-Shuffled Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 48 / 1 | 未通过 | 96.88% | 0.31% | 60.75% | 42.81% | 20.80% | +8.92 点 |
| 96 / 2 | 自动通过 | 99.07% | 2.49% | 60.75% | 49.68% | 21.42% | +12.67 点 |
| 144 / 3 | 自动通过 | 99.38% | 3.12% | 60.44% | 49.13% | 21.93% | +13.57 点 |
| 192 / 4 | 自动通过 | 99.38% | 3.12% | 60.44% | 50.48% | 22.00% | +13.52 点 |

按原自动规则，最早通过 checkpoint 为 correct step 96。结束标记监督修复后格式率恢复到 99% 左右，correct 相对错误语义置换训练的 micro/macro-F1 差距也明显扩大，说明训练目标内容产生了差异。

结论标记：`结论不足，暂不进入阶段二`。Correct 的 bucket accuracy 在 step 48/96 精确为 `195/321=60.7477%`，step 144/192 也只差一个样本；该恒定值强烈提示模型可能始终输出开发集多数 motion bucket。当前自动门只要求普通 bucket accuracy ≥50%，没有检查 balanced accuracy、预测 bucket 覆盖或混淆矩阵，因此“coarse bucket not collapsed”检查不充分。此外，固定语义置换会改变标签名边缘先验，correct 优于 shuffled-training 不能单独证明预测依赖视频内容。

2026-07-13 审计更正：保留原自动 `passed` 输出作为历史结果，但在补充以下两项后才决定阶段一是否真正通过：一是报告 bucket balanced accuracy、gold/pred bucket 分布和混淆矩阵；二是对同一个 correct step-96 模型比较正确视频帧与最大化 bucket 错配的帧置换输入。该复核只复用现有 checkpoint 和预测，不重新训练；原因是排除标签先验和多数类塌缩，而不是追加试参。

### 2026-07-13 多数类审计最终结果

更正后结果仍来自同一文件：`/tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_clean_4epoch_curve.json`。Gold motion bucket 分布为 complex-motion `213/321`、no-motion `61/321`、minor-motion `47/321`。

| 累计 step | Correct 普通 bucket ACC | Correct balanced ACC | Pred complex | Pred minor | Pred no-motion | 自动更正状态 |
|---|---:|---:|---:|---:|---:|---|
| 48 | 60.75% | 33.25% | 283 | 3 | 25 | 未通过 |
| 96 | 60.75% | 34.19% | 279 | 14 | 25 | 未通过 |
| 144 | 60.44% | 35.98% | 266 | 18 | 35 | 未通过 |
| 192 | 60.44% | 34.97% | 273 | 24 | 22 | 未通过 |

四个 checkpoint 虽然都至少预测过三个 bucket，但分布始终高度集中于 complex-motion；balanced accuracy 只在三分类随机水平 `33.33%` 附近，且从第三轮到第四轮没有继续改善。Correct 相对 shuffled-training 的多标签 F1 优势主要说明模型学到了不同标签先验/输出分布，不能证明从视频内容获得了可迁移的相机运动表征。

结论标记：`未通过`。原普通 accuracy 门的 `passed` 已被多数类审计推翻；历史输出保留，不静默覆盖。当前 camera-label SFT 前置路线正式停止，不运行打乱帧额外 GPU 推理、不执行阶段二 pair-rank 检测迁移，也不追加 epoch、prompt 复制、DPO 或 GRPO。打乱帧工具代码保留，但因为绝对 balanced accuracy 门已经失败，无需继续消耗算力证明视觉依赖。

### 2026-07-13 根因分析与结论边界

完整分析见 `docs/camera_pretext_failure_analysis_20260713.md`。对 CameraBench 论文、官方发布模型元数据和本地获批训练 JSON 的复核表明，本轮并未复现 CameraBench 的训练任务：CameraBench 将每个 primitive 分解成大规模、显式正负的 binary VQA，并用候选答案概率/AP 与配对 Q-Acc 评测；本轮只给每个视频一条完整稀疏标签集，用低学习率 rank-32 LoRA 做 token-level SFT。CameraBench 本地 processed files 有 38,672 条 balanced VQA、35,050 条 prompt-augmented captions 和 157,552 条 raw VQA，而本轮每轮只有约 750 条目标。官方主结果使用 full LM、8 FPS、LR `2e-5`；其 LoRA 对照也是 rank 64、LR `2e-4`。

因此本轮失败的合理解释是：完整标签集生成缺少逐标签负监督，类别不平衡促使模型学习高频标签先验；低容量、低学习率 LoRA、16 张离散图片以及 detection-specialized 起点进一步放大问题。干净四轮 train/inference prompt 与输入处理一致，结束标记也已修复，所以最终塌缩不再归因于明显的提示词或 adapter 工程错误。

结论边界更正为：停止的是“每视频一条完整 camera-label list 的低学习率 LoRA 前置学习”，不是整个 camera motion 方向。该结果不能证明 MLLM 学不会 camera，也不能证明 camera 对 AIGC 检测无用；下一步若继续，必须先用 balanced binary VQA、正确视频对 shuffled/no-video、candidate-level AP/Q-Acc 完成相机能力复现，再讨论联合检测迁移。

2026-07-13 算术更正：此前把 `60.7477%=195/321` 描述为开发集多数类比例，这是错误的；它实际是 bucket 判对数量。Gold 分布为 complex-motion `213/321=66.36%`、no-motion `61/321`、minor-motion `47/321`。更正不改变多数类塌缩结论，因为 correct-96 将 `279/321` 条预测为 complex，balanced accuracy 只有 34.19%；更正原因是区分 gold 先验比例与模型实际判对率。

## 13. DataA 平衡二元相机问答与视觉依赖门

### 这个实验测什么

把 DataA 的每个全局相机运动 primitive 分解成独立且正负平衡的 Yes/No 视频问答，先判断相机能力在当前数据上是否可学，再用对立标签视频置换和无视频输入排除只背问题、标签频率或回答格式的情况。这是相机辅助检测之前的能力门，不是最终检测方法，也不直接报告 AIGC 真伪检测收益。

### 状态与日期

- 日期：2026-07-13。
- 状态：`通过（通用模型起点与检测模型起点均通过）`。
- 执行说明：`docs/dataa_camera_binary_vqa_unattended_20260713.md`。

### 模型与数据

- 通用模型起点：`/home/admin/Qwen3-VL-8B-Instruct`。
- 检测模型起点：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 最终 DataA case manifest：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_flow_probe_40step_v3/dataa_camera_flow_probe_manifest_40step_v3.jsonl`。
- 训练身份：manifest 中固定 `dataset_split=train` 的 DataA real 视频。
- 开发身份：manifest 中固定 `dataset_split=test` 的 DataA real 视频，预计 321 个 case；这些身份已经参加过项目内多轮诊断，只称开发集，不称论文最终 held-out test。
- 构建后的平衡问答、三种开发输入和审计摘要：各运行的 `/tmp/1res/dataa_camera_binary_vqa/<运行名>/data/`。

### 唯一改变因素与对照

- 两套机器使用完全相同的 case split、问题、样本顺序、LoRA、学习率、FPS、最大像素、训练轮数和评测代码，唯一跨机器因素是通用 Qwen3-VL 起点或 DataB detection SFT 起点。
- 每个支持的 camera primitive 都构造等量 Yes/No 训练与开发样本；每个 primitive 只使用一个固定语义问题，不把同一视频复制成 25 个 prompt。
- 正确视频条件：问题与原视频 camera label 匹配。
- 对立标签视频置换条件：保留问题和 gold answer，但将同一 primitive 的 Yes/No 配对视频互换；若模型依赖视频，性能应明显下降。
- 无视频条件：保留同一问题和 gold answer，移除视频；平衡数据使纯文本标签先验不能稳定超过随机水平。
- 推理不接收 GT camera caption 或 camera label 文本；输出分数由首个回答位置的 `P(Yes)` 与 `P(No)` 得到。

### 主要设置

- 输入使用 DataA 原始 real MP4，按 8 FPS 采样，`video_max_pixels=16384`。
- 16 GPU、每卡 1 个视频；每个 rank 使用 4 个 CPU threads，避免 16 个解码进程过度抢占 96 个物理核。
- LoRA rank 64、alpha 128、dropout 0.05，学习率 `2e-4`，cosine scheduler，warmup 3%。
- 最多 5 epochs；训练墙钟上限默认 16200 秒，并保证至少完成 1 epoch。保存第一轮 adapter 和实际训练终点 adapter，避免用短跑失败直接否定可学性。
- 指标：candidate-level AP、ROC-AUC、balanced accuracy、逐 primitive macro 指标和 paired question accuracy；基础模型、第一轮和最终轮均有正确视频结果，最终轮额外评测两种视觉控制。

### 验收标准

- 预测覆盖率至少 99%，至少 20 个 primitive 同时具有足够 train/dev 正负支持，实际训练至少 1 epoch。
- 最终正确视频 macro AP 至少 65%，总体 balanced accuracy 至少 60%，paired question accuracy 至少 35%。
- 最终正确视频 macro AP 相对同一起点未训练模型至少提高 8 个百分点。
- 最终正确视频 balanced accuracy 相对无视频至少提高 8 个百分点，相对对立标签视频置换至少提高 15 个百分点。
- 上述阈值是进入下一阶段的工程门，不作为论文最终显著性结论；明日还要横向比较两个起点的逐标签曲线和训练终点。

### 泄漏、分布差异与结论边界

- 训练/开发按 case identity 严格互斥；构建脚本保存 hash、每类正负支持和交集审计。
- DataA 源自 CameraBench train videos，且固定开发身份已被前序实验反复查看，因此该门只能回答“当前监督和模型能否学到相机视觉能力”，不能冒充 CameraBench 官方测试或论文最终泛化结果。
- 本轮只使用 real 视频和 camera labels，不包含 fake 视频、AIGC verdict 或局部 bbox，因此通过也不证明 camera 能提高检测；它只允许下一步构造联合 `detection + camera auxiliary` 的最小门。
- 如果通用起点通过而检测起点失败，支持“专项 detection SFT 使 camera 能力难以恢复”的诊断；两者都通过说明可直接从检测起点继续；两者都失败则先审计标签映射、视频采样和优化目标，不启动联合检测大实验。

### 存储与立即下一步

- DataA 原视频继续位于 `/tmp`，本轮不复制视频。
- 逐样本分数、数据摘要、训练日志和 gate summary 属于持久化小结果，脚本退出时复制到 `/input/workflow_58770161/workspace/test/cameramotion_det/res/dataa_camera_binary_vqa/<运行名>/`。
- 第一轮和最终 LoRA adapter 属于可能复用的大结果，保存在 `/tmp/1res/dataa_camera_binary_vqa/<运行名>/`，脚本退出时自动上传到 `oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataa_camera_binary_vqa/<运行名>/`。
- 明日先比较两个 `gate_summary.json`；只有至少检测起点通过，或通用起点明显通过且检测起点失败的原因可修复，才设计无 GT camera 文本的联合检测辅助训练。今晚不并行投机运行联合检测、DPO、GRPO 或完整 VIF-Bench。

### 2026-07-13 检测模型起点正式结果

这次实际测试的是：从完整 DataB 检测 SFT checkpoint 出发，只用 DataA real 视频的平衡二元相机问答训练 LoRA，模型能否在 case-level 隔离的 321 个开发视频上学习相机 primitive，并在移除视频或替换为相反标签视频时出现符合因果预期的性能下降。

结果来源：

- NAS 小结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataa_camera_binary_vqa/detection_checkpoint_start/`。
- OSS 运行目录：`oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataa_camera_binary_vqa/detection_checkpoint_start/`。
- 本地结果包：`E:/newgaibeishi/detection_checkpoint_start_results.tar.gz`。
- 核心汇总：`eval/gate_summary.json`；逐阶段评测：`eval/base.json`、`eval/epoch1.json`、`eval/final.json`；训练曲线：`train/trainer_log.jsonl`。

#### 数据与训练完成性

| 项目 | 结果 |
|---|---:|
| DataA train / dev cases | 759 / 321 |
| train/dev case 交集 | 0 |
| 支持的 camera primitives | 32；`very-unsteady` 因无正例排除 |
| 平衡训练记录 | 5652；Yes/No 各 2826 |
| 每个开发条件记录 | 2002 |
| 训练步数 | 1770/1770，5.0 epochs |
| LoRA / 学习率 | rank 64，alpha 128，`2e-4` |
| 训练耗时 | 2863.8 秒；总模型设置加训练 2911.9 秒 |
| 16 GPU 计算阶段平均利用率 | 74.13% |

训练 loss 从首步 `0.3493` 下降到末步 `0.0580`，全程无非有限 loss 或 OOM；最终停止原因为 `planned_steps_completed`。GPU 计算阶段不足完整两小时，因此没有完整两小时窗口，但 56 次采样的平均利用率为 74.13%，不构成服务器低利用率风险。

#### 主指标与视觉控制

| 模型/输入条件 | Balanced ACC | Overall AP | Macro AP | ROC-AUC | Paired question ACC |
|---|---:|---:|---:|---:|---:|
| 未训练检测起点 + 正确视频 | 58.94% | 61.45% | 68.32% | 63.57% | 27.17% |
| 训练 1 epoch + 正确视频 | 50.05% | 57.58% | 60.76% | 59.74% | 0.50% |
| 训练 5 epochs + 正确视频 | 73.33% | 80.65% | 81.70% | 81.09% | 53.25% |
| 训练 5 epochs + 对立标签视频 | 26.67% | 34.16% | 36.87% | 18.91% | 6.59% |
| 训练 5 epochs + 无视频 | 50.00% | 50.00% | 50.00% | 50.00% | 0.00% |

预设检查全部通过：最终 macro AP 相对未训练起点提高 `13.38` 点；正确视频 Balanced ACC 相对无视频提高 `23.33` 点、相对对立标签视频提高 `46.65` 点。对立标签视频把同一 primitive 的 Yes/No 配对视频互换后，最终模型的 Yes/No 均值分数和混淆方向几乎精确反转；无视频时两类分数分布相同并回到随机水平。这比单纯“训练后测试更高”更强地说明模型实际依赖视频内容，而不是只背固定问题或回答先验。

逐 primitive 结果也不是少数大类驱动：32 类中 31 类最终 AP 不低于 65%，27 类相对未训练起点提高。当前最弱类别是 `truck-left`，AP 59.27%；`tilt-down`、`fast-speed`、`pan-tracking`、`tail-tracking` 和 `truck-left` 五类未超过各自起点，其中若干开发正负对只有 5 至 22 对，逐类差值仍受小样本波动影响。

第一轮 checkpoint 几乎全部回答 `No`：2002 条中只有 9 条预测为 Yes，paired question accuracy 仅 0.50%。但其 AP/ROC-AUC 仍高于随机，说明早期模型已经存在较弱排序信号，只是零阈值严重失准；随后继续训练到 5 轮才形成清晰的正负间隔。因此旧版 24/48 步或一轮即停的能力门不足以否定相机能力可学，本轮保存第一轮并继续到固定终点的设计是必要的。

结论标记：`通过`。该结果建立“DataB detection SFT 起点可以通过平衡二元 VQA 从原视频学到具有视觉依赖性的 camera-motion 能力”，并否定此前完整标签集 SFT 失败可外推为“当前模型学不会 camera motion”的解释。它不建立 AIGC 检测提升、检测能力保留或外部 CameraBench 泛化，因为训练和开发均来自 DataA/CameraBench train 分布，且当前开发身份已参与项目内多次诊断。

立即下一步：先完成通用 Qwen3-VL 起点的同协议结果并做横向比较；随后对现有 detection-start camera adapter 做无 camera 文本的 DataA/VIF-Bench 检测保留诊断。只有确认保留代价后，才固定一个联合 `detection replay + binary camera auxiliary` 训练配方和等步数 detection-only 对照；不把 camera caption 作为检测推理输入，也不在本结果上直接启动 DPO/GRPO。

2026-07-13 调度更正：通用起点同协议实验继续在原服务器独立运行，但不再阻塞主线。空闲的第二套 16 GPU 先执行第 14 节的 DataA 原提示词检测保留诊断；更正原因是检测起点已经以较大余量通过相机能力门，当前更关键的不确定性转为 camera-only LoRA 对既有检测输出的直接影响。

### 2026-07-13 通用模型起点正式结果与双起点比较

这次实际测试的是：在与检测模型起点完全相同的数据、样本顺序、训练参数和评测控制下，通用 Qwen3-VL-8B-Instruct 能否学习同一组平衡二元相机问答；该结果用于判断既有 detection SFT 是否明显损害相机能力，不用于代替 AIGC 检测实验。

结果来源：

- NAS 小结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataa_camera_binary_vqa/generic_instruct_start/`。
- OSS 运行目录：`oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataa_camera_binary_vqa/generic_instruct_start/`。
- 本地结果包：`E:/newgaibeishi/generic_instruct_start_results.tar.gz`。
- 核心汇总：`eval/gate_summary.json`；逐阶段评测：`eval/base.json`、`eval/epoch1.json`、`eval/final.json`；训练曲线：`train/trainer_log.jsonl`。

#### 数据、协议与训练完成性

通用起点和检测起点运行的四个构建数据文件 SHA-256 完全一致，包括 5652 条平衡训练记录和三套各 2002 条的开发条件；两次运行均为 759/321 个 case 级互斥的 train/dev、32 个支持类别、5 epochs、1770 steps、LoRA rank 64/alpha 128 和学习率 `2e-4`。因此双起点比较没有数据抽样或步数差异。

| 项目 | 通用模型起点结果 |
|---|---:|
| DataA train / dev cases | 759 / 321；交集 0 |
| 平衡训练记录 | 5652；Yes/No 各 2826 |
| 训练步数 | 1770/1770，5.0 epochs |
| 训练耗时 | 4019.0 秒；总模型设置加训练 4078.3 秒 |
| 16 GPU 计算阶段平均利用率 | 75.87% |

训练 loss 从首步 `0.5311` 下降到末步 `0.0427`，停止原因为 `planned_steps_completed`；计算阶段没有 OOM、非有限 loss 或未完成训练。该次计算阶段不足完整两小时，因此 GPU 审计没有完整两小时窗口，但 77 次采样的整体平均利用率为 75.87%。

#### 通用模型起点的主指标与视觉控制

| 模型/输入条件 | Balanced ACC | Overall AP | Macro AP | ROC-AUC | Paired question ACC |
|---|---:|---:|---:|---:|---:|
| 未训练通用起点 + 正确视频 | 57.99% | 63.56% | 69.77% | 64.89% | 23.78% |
| 训练 1 epoch + 正确视频 | 52.60% | 64.66% | 67.02% | 67.57% | 6.29% |
| 训练 5 epochs + 正确视频 | 74.28% | 80.92% | 83.52% | 81.69% | 55.34% |
| 训练 5 epochs + 对立标签视频 | 25.72% | 34.03% | 36.50% | 18.31% | 6.79% |
| 训练 5 epochs + 无视频 | 50.00% | 50.00% | 50.00% | 50.00% | 0.00% |

预设检查全部通过：最终 macro AP 相对未训练通用起点提高 `13.74` 点；正确视频 Balanced ACC 相对无视频提高 `24.28` 点、相对对立标签视频提高 `48.55` 点。无视频退回随机水平，对立标签视频使性能反向下降，说明通用起点训练后同样实际依赖视频，而不是依赖固定问题或回答频率。

32 类中 31 类最终 AP 不低于 65%，28 类相对各自未训练起点提高。唯一低于 65% 的 `arc-tracking` 最终 AP 为 54.91%，并相对起点下降 9.77 点；该类开发集仅 9 对正负样本，单类结果波动较大。其余最低的 `unsteady` AP 为 68.25%，因此总体通过不是由少数大类单独驱动。

#### 通用起点与检测起点的对齐比较

| 阶段/指标 | 通用 Qwen3-VL 起点 | DataB 检测 SFT 起点 | 通用减检测 |
|---|---:|---:|---:|
| 未训练起点 Balanced ACC | 57.99% | 58.94% | -0.95 点 |
| 未训练起点 Macro AP | 69.77% | 68.32% | +1.45 点 |
| 未训练起点 Paired question ACC | 23.78% | 27.17% | -3.40 点 |
| 训练 5 epochs Balanced ACC | 74.28% | 73.33% | +0.95 点 |
| 训练 5 epochs Overall AP | 80.92% | 80.65% | +0.27 点 |
| 训练 5 epochs Macro AP | 83.52% | 81.70% | +1.82 点 |
| 训练 5 epochs ROC-AUC | 81.69% | 81.09% | +0.60 点 |
| 训练 5 epochs Paired question ACC | 55.34% | 53.25% | +2.10 点 |

两个起点都在第一轮出现硬分类阈值偏向 `No`、继续训练后恢复，并且最终均以较大余量通过所有视觉依赖验收。检测起点相对通用起点在最终指标上低 `0.27` 至 `2.10` 点，但未训练起点指标本身有正有负，且当前每个起点只有一次训练运行、没有跨随机种子或逐样本跨模型显著性检验；因此只能记录为轻微总体差异，不能据此宣称 detection SFT 造成统计显著退化，更不支持“灾难性遗忘”。逐类上通用最终 AP 在 21/32 类更高、检测起点在 11/32 类更高，说明模型谱系会改变类别偏好，但不改变两者均可学习的主结论。

结论标记：`通过`。这组双起点消融建立了平衡二元目标是可学且具有视觉依赖的，也表明已有检测 checkpoint 保留了足以继续注入相机能力的表征。它仍不建立相机辅助后的真伪检测收益、原检测能力保留或 CameraBench 官方测试泛化。

立即下一步维持不变：主线继续使用已有 DataB 检测 SFT checkpoint 及其 camera adapter，执行第 14 节“无 camera 文本、原检测提示词”的检测保留诊断。通用起点完成了模型谱系消融，不需要再训练一套通用起点检测模型，也不应因其约 1 至 2 点的相机指标优势切换主线。

## 14. 相机二元问答适配器的原检测提示词保留诊断

### 这个实验测什么

在 detection checkpoint 已经学会视觉依赖的相机二元问答后，检查最终 camera LoRA 是否能在不提供任何 camera 文本的原始 DataA 检测 prompt 下保留真假分类、Real/Fake 成对正确率和解释证据格式。这是联合辅助训练前的能力保留诊断，不是 camera 提升检测的方法实验。

### 模型、数据与单一改变因素

- 原始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- Camera 模型：同一原始模型合并 `/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final`。
- Detection 数据：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`。
- 固定开发身份：`/input/workflow_58770161/workspace/test/cameramotion_det/tools/data/camera_motion_splits/dataA_test.json` 中 321 个 case；构建器必须得到 321 个完整 Real/Fake pair、642 条记录。
- 唯一改变因素是是否挂载最终 binary-camera LoRA。两者使用相同 16 帧、原 detection system/user prompt、图像像素上限、贪心生成和评测脚本；推理不加入 camera caption、labels、bbox、mask 或光流。
- 执行说明：`docs/camera_detection_retention_gate_20260713.md`。

### 主要设置与验收标准

- 16 GPU 推理，默认每卡 2 个独立生成进程、共 32 个数据分片；`image_max_pixels=262144`、`max_new_tokens=2048`、`prompt_mode=record`。
- 两个模型覆盖率均不低于 99%，camera 模型 `<answer>` 格式有效率不低于 95%。
- Camera 模型相对原 checkpoint 的 Balanced ACC、Fake F1、pair accuracy 各下降不超过 3 个百分点。
- 解释 evidence 输出率、temporal IoU、bbox IoU 和 Evidence@0.3 全部报告，但不作为当前硬门，因为起点在 DataA 上的证据指标本身较低。
- 通过只表示直接能力保留，不表示 camera 改善检测；未通过也不否定 camera 能力，而是要求联合训练显式混入 detection replay，不能继续把 camera-only adapter 当检测模型。

### 泄漏、分布与下一步

固定 321 个 DataA case 已被项目多次用于诊断，只称开发集；它们与 camera VQA 的 759 个训练 case 按 case 隔离，但不属于全新论文测试。当前只测 DataA，VIF-Bench 保留尚未测试。结果通过后进入等步数 `detection-only` 与 `detection replay + binary camera auxiliary` 最小联合训练门；结果未通过则仍进入该联合门，但提高 replay 约束并把保留作为硬指标。通用 Qwen3-VL 起点结果可后补为模型谱系消融，不阻塞本实验。

2026-07-13 并行调度更正：第 14 节 DataA 保留诊断已在第一台服务器执行；第二台服务器并行执行第 15 节 VIF-Bench 外部分布保留诊断。更正原因是 VIF-Bench 诊断不依赖 DataA 结果，且能独立区分局部编辑检测遗忘与通用全生成视频检测遗忘。

### 2026-07-13 正式结果

这次实际测试的是：在同一 40step_v3 固定 DataA 开发集、同一原检测 prompt 和无 camera 文本推理条件下，比较原 DataB 检测 checkpoint 与合并最终 binary-camera LoRA 后的模型。评测覆盖 321 个 Real/Fake pair、642 条记录，两个模型的预测记录均完整返回。

结果来源：

- 服务器持久化目录：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/detection_checkpoint_start/eval/`。
- 本地附件：`E:/newgaibeishi/eval (2).zip`。
- 总门结果：`eval/camera_detection_retention_summary.json`。
- 明细：`eval/base/dataa_detection_base_summary.json` 与 `eval/camera_adapter/dataa_detection_camera_adapter_summary.json`。

| 模型 | 覆盖率 | 格式有效率 | Balanced ACC | Fake Recall | Fake F1 | Pair ACC |
|---|---:|---:|---:|---:|---:|---:|
| 原 DataB 检测 checkpoint | 100.00% | 99.84% | 50.31% | 31.78% | 39.01% | 13.08% |
| 合并 binary-camera LoRA | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Camera 减原模型 | 0.00 点 | -99.84 点 | -50.31 点 | -31.78 点 | -39.01 点 | -13.08 点 |

原模型在该 DataA 开发集上预测 Fake/Real/Unknown 分别为 `202/439/1`；camera 模型为 `0/0/642`，即 642 条全部无法由评测器解析为 Real/Fake。所有预设保留检查中只有两边覆盖率通过；camera 格式、Balanced ACC、Fake F1 和 Pair ACC 保留均未通过。

| 解释证据指标 | 原检测 checkpoint | Camera 模型 |
|---|---:|---:|
| 预测 evidence 样本率 | 31.46% | 0.00% |
| mean best temporal IoU | 24.76% | 0.00% |
| mean best bbox IoU | 10.01% | 0.00% |
| Evidence hit `t0.3/b0.3` | 12.46% | 0.00% |

结论标记：`未通过`。当前结果足以否定“把只训练 camera VQA 的最终 LoRA 顺序挂回检测模型即可直接保留原检测接口”这一做法；它不证明 camera 视觉能力不能通过联合多任务训练帮助检测，也不能把 0 分解释为语义检测能力必然归零，因为附件没有包含原始生成文本。642/642 全部 Unknown 更像输出接口或任务格式发生整体接管，而不是普通分类性能波动；仍需检查至少一个 `inference/camera_adapter/rank_*/*.json` 的原始 `response`，区分模型在回答 Yes/No、生成空文本、生成其他模板或模型合并异常。

立即下一步：不重跑同一 DataA 门，不把 camera-only adapter 当检测模型。先读取原始 camera 回复；同时等待第 15 节 VIF-Bench 保留结果。若原始回复是稳定 Yes/No，则联合训练必须混入原 detection prompt/answer replay 并以格式保留为硬门；若为空或乱码，则先审计 adapter merge 和推理加载，不能直接归因于灾难性遗忘。

2026-07-13 根因补充：从 `inference/camera_adapter/rank_0/camera_adapter-rank0_dataa_rank00.json` 和 `rank_1/camera_adapter-rank1_dataa_rank01.json` 抽查的原始回复均为字面 `Yes` 或 `No`，对应 `answer=UNKNOWN`、`answer_parse_source=missing`。样例同时覆盖 Real/Fake 输入，排除空生成、乱码和评测器漏掉已有 Real/Fake 单词；这确认失败的直接原因是 camera VQA 单任务 LoRA 接管了回答词汇与输出接口，而不是 adapter merge 完全失效。

因此将历史结论从“待原始回复归因”更正为“确认任务接口接管”。现有结果仍不能测量其内部是否残留真假排序语义，也不支持把问题表述为全部视觉知识的灾难性遗忘：同一 adapter 已在相机二元问答门中证明 camera 视觉能力存在。方法设计上的直接约束是下一轮不能继续 `detection checkpoint -> camera-only LoRA -> 直接检测` 的顺序配方；应从 detection checkpoint 进行交错的 `detection replay + binary camera auxiliary` 联合训练，并同时验收相机 AP、检测格式和检测指标。

## 15. VIF-Bench 相机适配器外部分布检测保留诊断

### 这个实验测什么

比较原始 DataB 检测 SFT checkpoint 与合并最终 DataA 平衡二元相机问答 LoRA 后的模型，在 VIF-Bench 原检测协议上的能力差异。它回答 camera-only adapter 是否损伤既有全生成视频检测能力；不是 camera 改善检测的实验，也不向推理提供任何 camera 文本。

### 状态、模型与数据

- 日期：2026-07-13。
- 状态：`结论不足（Base 基线完成；Camera 分支未执行）`。
- 原始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- Camera 模型：同一模型合并 `/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final`。
- 测试数据：VIF-Bench 当前 16 个 index shard 指向的抽帧数据；当前服务器入口为 `/input/workflow_58770161/workspace/test/cameramotion_det/eval/v4train-main/test_index_splits/splits_16`，逐源视频数量由预检审计记录，当前不猜测。
- 推理与官方评测：同一目录的 `infer2_5_3.sh` 与 `eval.py`。
- 执行说明：`docs/vifbench_camera_adapter_retention_gate_20260713.md`。

### 单一改变因素与主要设置

- 唯一改变因素是是否合并最终 binary-camera LoRA；两个模型使用完全相同的 VIF-Bench index、16 帧输入、确定性生成、system prompt 和 no-camera user suffix。
- `PROMPT_MODE=no_camera`，不传 `CAMERA_CONTEXT_JSONL`；预检保存两个 prompt 文件的 SHA-256，并拒绝 user suffix 中的 camera placeholder。
- 16 GPU；默认原模型和 camera 模型并行，每张 96G GPU 各一个模型进程，共两个推理进程。若运行时不兼容可设 `PARALLEL_MODELS=0` 顺序运行，实验定义不变。
- 原始逐样本预测、合并模型和合并预测放在 `/tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start`；只把审计、评测 JSON/CSV 和 pipeline log 持久化到 `/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/vifbench_detection_checkpoint_start`。

### 验收标准

- 两个模型预测覆盖率均至少 99%，`<answer>` 格式有效率均至少 99%，并覆盖相同且非空的生成模型子集。
- Camera 模型相对原模型的跨生成模型平均 Balanced ACC 和 Fake F1 降幅各不超过 3 个百分点。
- 同时报告每个生成模型的 ACC、Fake Recall、Fake F1 变化；逐生成模型下降暂不设硬阈值，防止小子集波动替代总体判断。
- 新汇总复刻原 `eval.py` 将非 Real 输出编码为 Fake 的官方口径，同时单列格式有效率和 strict-valid pair 指标，避免格式错误被掩盖。

### 泄漏、分布差异与立即下一步

VIF-Bench 没有参加 DataA camera adapter 训练，因此可作为适配器训练之外的外部分布保留诊断；但项目此前已经多次查看 VIF-Bench 指标，所以不能称为全新的论文最终 held-out model-selection test。训练接收 camera VQA 监督而本次检测推理不接收 camera 文本，这里是有意进行的无 camera 文本能力保留压力测试，不能描述为 camera-conditioned 方法结果。

立即下一步：与第 14 节 DataA 结果组成二维决策。两者都保留时进入等步数联合辅助训练；仅 DataA 下降时加强 DataA detection replay；仅 VIF-Bench 下降时加强 DataB replay；两者都下降时停止顺序叠加 camera adapter，先做带明确 detection replay 的小规模联合训练门，不直接启动大规模联合训练或 RL。

2026-07-13 路径更正：首次预检发现当前 V4Train 副本的 `test_index_splits` 位于 `v4train-main/` 根目录而不是其 `eval/` 子目录；已修正记录并让 runner 自动兼容两种布局。该更正只影响文件发现，不改变数据、提示词或实验定义。

### 2026-07-13 Base-only 正式结果与 Camera 未执行说明

这次实际完成的是原 DataB 检测 checkpoint 在严格 `no_camera` prompt 下的 VIF-Bench 全量基线。16 个 rank 均完成，共得到 3160/3160 条匹配预测；Camera 分支在模型加载前失败，因此本结果只建立 Base 控制值，不建立 camera retention 差值。

结果来源：

- NAS：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/vifbench_detection_checkpoint_start/eval/base_vifbench_eval.json`。
- 官方日志与 CSV：同目录的 `base_official_eval.log`、`base_official_paired_metrics.csv`。
- 终端结果附件：`C:/Users/29499/.codex/attachments/ba1f4532-2319-4bc2-83a9-86ef3a067305/pasted-text.txt`。

| 项目 | Base 结果 |
|---|---:|
| 预期 / 实际 / 匹配预测 | 3160 / 3160 / 3160 |
| 覆盖率 | 100.00% |
| 格式有效率 | 99.97%（3159/3160） |
| 生成模型子集 | 19 |
| 跨生成模型平均 Balanced ACC | 79.18% |
| 跨生成模型平均 Fake Recall | 89.33% |
| 跨生成模型平均 Fake F1 | 80.47% |

逐生成模型最难的两个子集是 `HunyuanVideo-I2V`（Balanced ACC 54.85%，Fake F1 47.72%）和 `Wan2.1-VACE-1.3B-T`（63.03%，60.90%）；较高的子集包括 `gen4-turbo`（84.82%，86.51%）、`kling-v1`（84.75%，86.60%）和 `pixverse-v4-5`（84.54%，86.46%）。完整 19 子集结果保留在 Base JSON，不在记录中重复展开。

当前严格同提示词 Base 值低于旧记录的 ACC 83.96%/F1 84.72%约 4.78/4.25 点；这是不同提示词与当前严格协议之间的历史对照，不是模型参数变化的受控消融。后续联合模型必须与本次 Base 采用相同 prompt hash、index 和评测脚本，不能再拿旧数值作为直接控制组。

Camera 未执行的直接原因是：第一次 adapter merge 在写出权重分片后因 `mistral_common`/Transformers 的 processor 加载冲突中止，旧 runner 随后只凭 `config.json` 复用了半成品目录；16 个 Camera 进程均因 `text_config.rope_scaling=None` 在构造模型时失败，预测文件数为 0。该问题属于运行基础设施失败，不是 Camera 模型的 VIF 分数。代码已改为临时目录原子合并、processor/config 审计和 `.merge_complete` 标记，防止再次复用半成品。

结论标记：`结论不足`。Base 基线有效；Camera retention 没有被测试。由于第 14 节已经在 642 条 DataA 原检测提示样本上确认 camera-only LoRA 的 Yes/No 接口接管，继续为同一顺序配方重跑 3160 条 Camera VIF 推理的信息增量不足，因此不补跑 Camera 全量分支。该 Base 结果作为下一轮 `detection replay + binary camera auxiliary` 联合模型的严格 VIF 控制值。

## 16. 同 16 帧二元相机辅助与检测回放联合训练三分支验证

### 这个实验测什么

从完整 DataB 检测 checkpoint 出发，让二元相机 VQA 与检测任务使用同一套 16 帧视觉序列，比较仅检测回放、正确相机监督和逐条翻转相机监督三个等记录数、等训练步数分支。它先验证正确监督是否比错误监督学到视觉相关相机能力并保留 Real/Fake 检测接口，再用 `pass@8` 与组内奖励方差判断是否值得进入短程 GRPO。第一轮尚不执行 RL。

### 日期、状态与模型谱系

- 日期：2026-07-13。
- 状态：`代码已就绪，待服务器构造数据和执行`。
- 三个分支共同起点：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 训练方式：Qwen3-VL-8B LoRA-SFT，冻结视觉塔与多模态投影层，训练语言侧 LoRA。
- 完整执行说明：`docs/camera_joint_sft_gate_execution_20260713.md`。

### 数据与准确路径

- DataA detection：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`。
- DataA camera labels/caption：`/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`。其中 labels 构造二元监督；caption 只写入 split 审计，不作为本轮训练目标。
- DataA 视觉输入：由上述 detection JSON 指向 `/tmp/cameramotion_det/dataA_v1/autolabel/dataa_vace_grounded_cot_frames_40step_v3` 下的每例 16 帧。
- DataB detection replay：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`。
- DataB camera 伪标签：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`，只用于 replay 分层采样，不进入 detection prompt。
- DataA 按 case、VACE 来源和 coarse motion bucket 做 70:30 分层划分；real/fake pair 不得跨 split。正式构建要求 1080 个完整 case，旧 split 不复用。
- 检测评测：DataA 新 test split 与 VIF-Bench；相机评测：DataA 新 test split 的 real 16 帧。

### 三个分支和单一改变因素

| 中文分支 | 训练内容 | 作用 |
|---|---|---|
| 仅检测回放分支 | DataA train detection + DataB detection replay + 等量额外 detection | 控制继续训练、样本数和计算量 |
| 正确相机监督分支 | 相同 detection pool + 全部可平衡 primitive 的正确 Yes/No 相机样本 | 要验证的相机辅助条件 |
| 翻转相机监督分支 | 相同 detection pool + 问题和 16 帧不变、每条 Yes/No 答案翻转 | 排除任务格式、相机任务数量和计算量造成的假提升 |

正确与翻转分支逐条使用相同问题和相同 16 帧，答案必定相反；原数据每个 primitive 的 Yes/No 本身平衡，因此翻转后总体答案边际完全不变。每个 primitive 只使用一个固定问题，不把同一视频复制成 25 个提示词版本。检测 prompt 保持原样，训练和检测推理均不向 detection user prompt 注入 camera 文本。

### 主要训练和评测设置

- 相机分支使用每个 primitive 的全部可平衡 Yes/No 样本；检测 pool 在 DataA train 与 DataB replay 的基础上确定性过采样到与相机记录 1:1。仅检测分支再用等量检测样本替换相机槽位，因此三分支记录数完全相同。
- 本地旧 1076-case 真实结构干跑为 5528 条相机记录、5528 条检测记录、每分支 11056 条；正式 1080-case 数量以服务器审计为准，不把本地旧数据数字登记为正式实验结果。
- LoRA rank 64、alpha 128、dropout 0.05、学习率 `2e-4`、5 epochs、16 GPU、每卡 batch 1、梯度累积 1、cosine scheduler、warmup 0.03、bf16，按 epoch 保存。
- 使用 5 epochs 的依据是第 13 节同协议真实结果：第 1 epoch 尚未学稳，第 5 epoch 才以较大余量通过相机视觉依赖门。使用 1014 条相机样本训练 1 epoch 会把监督不足混入方法判断，故不作为正式默认值。
- 相机训练输出严格为 `Yes` 或 `No`。评测直接比较 Yes/No token logit，报告 Overall/Macro AP、Balanced ACC、ROC-AUC 和 paired question accuracy，不依赖自由生成格式。
- 视觉依赖控制分别换入同问题下相反答案的视频帧和移除全部帧；RL 就绪度对每例采样 8 次，报告正确答案 `pass@8`、两个动作探索率和组内奖励方差。
- DataA/VIF 检测推理使用原检测 prompt，且不提供任何 camera label/caption。

### 验收标准

- 数据完整性：1080 case、train/test 无交集、pair 不跨 split、train/dev 至少 20 个相机 primitive 有正负支持、三分支等量、翻转后 Yes/No 边际不变、无 detection prompt camera 文本泄漏。
- 正确相机监督相对翻转监督的 Macro AP 提高至少 3 点，或 Balanced ACC 提高至少 5 点。
- 正确分支在匹配帧相对相反答案帧的 Balanced ACC 至少下降 10 点，或相对无帧至少下降 8 点，证明回答依赖视觉而不是固定问题先验。
- RL 可验证奖励为严格 Yes/No 格式 `0.1` 加正确答案 `0.9`；就绪度要求留出覆盖完整、`pass@8` 存在正确动作，并且至少 20% 样本具有非恒定组内奖励。边界情况只允许短程 GRPO，不启动全量 RL。
- DataA 与 VIF-Bench 检测指标和格式必须完整报告。检测提点是强正信号，但不是本轮 SFT 的单独硬门；严重接口接管或正确分支明显差于两个控制则不进入 RL。

### 泄漏、分布差异与已知限制

- 起始 checkpoint 已见过完整 DataB，故 DataB 只能作为检测回放，任何 DataB 内部高指标都不称为 held-out。
- DataA test 未被起始 DataB checkpoint 训练，但会用于本轮方案选择，因此称开发留出集，不直接包装为最终论文测试集。
- VIF-Bench 没有参与本轮训练，可作为外部分布保留比较；但项目已多次查看其结果，最终论文仍需明确其开发使用历史。
- DataA camera labels 的来源质量继承 CameraBench 标注流程；正确优于翻转只能证明监督包含可利用视觉信息，不自动证明相机运动已改善真假检测。Caption 本轮不参与训练，避免把不可验证的长文本质量混入第一门。
- 训练提供 camera 辅助目标，检测推理不提供 camera 文本；这是内部能力注入设计，不是缺失 camera 条件的压力测试。

### 立即下一步

先在服务器执行只检查文件的 preflight，再构造正式 1080-case split 并检查审计 JSON。审计通过后运行三个 5-epoch LoRA-SFT 分支；先完成相机能力、翻转监督、视觉依赖和 `pass@8`，随后做 DataA 无相机文本检测保留。只有正确分支同时通过这些条件，才运行三分支 VIF-Bench 和短程 GRPO。当前没有模型结果，不提前填写通过或失败。

## 17. DataB 自动解释的 DeepfakeJudge-7B 可靠性门

### 这个实验测什么

在使用开源 Judge 筛选 DataB 自动解释之前，先验证 DeepfakeJudge-7B pointwise 是否会依据当前视频的有序帧和局部证据降低错配解释、错误 bbox、错误时间段和错误伪影类别的评分。它评价的是自动 CoT 的视觉忠实度，不重新定义 DataB 原始 Real/Fake 身份。

### 日期、状态、模型与数据

- 日期：2026-07-13。
- 状态：`代码已就绪，待服务器执行`。
- Judge：`MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ`，服务器模型路径默认为 `/tmp/1res/models/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ`。
- DataB：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`，共 6766 条，Real/Fake 各 3383 条。
- 候选解释：每条记录最后一个 assistant 消息中的 `<think>...</think><answer>...</answer>`。
- 可信身份：独立从第一张图片路径中的精确 `real`/`fake` 目录取得，禁止从候选 `<answer>` 取得。
- 完整执行说明：`docs/datab_deepfakejudge_gate_20260713.md`。

### 单一改变因素和控制

按 Real/Fake 与来源分层抽取 200 条。原始条件保持帧、可信身份和候选 CoT 全部匹配；控制条件每次只改变一种证据：换入另一条同标签视频帧、移动 bbox、移动时间段或替换 V4+ 伪影类别。所有条件使用同一模型、同一评分提示、相同视觉分辨率和确定性生成。

### 主要设置与输出位置

- 有序图片输入保持 JSON 中原顺序；每张图片 `max_pixels=262144`，输出最多 512 tokens。
- 使用 `torchrun` 在 16 张 GPU 上按记录分片，每张 GPU 加载一个 7B pointwise Judge；不训练模型。
- 门验收的输入、预测、CSV 和汇总均为一次性验证输出，放在 `/tmp/1res/datab_deepfakejudge_gate`，当前不放 NAS、不上传 OSS。
- 构造阶段同时做确定性格式审计：身份与候选答案一致性、V4+ 类别、时间范围、归一化 bbox 和可疑乱码。

### 验收标准

- 预测覆盖率至少 98%，`<reasoning>`/`<score>` 格式有效率至少 95%。
- 原始解释严格高于同标签错配帧的比例至少 70%。
- bbox、时间和类别控制中，至少一个拥有不少于 30 个有效配对，且原始解释严格高于控制的比例至少 65%。
- 通过只证明 Judge 在当前域具有初步视觉敏感性，不证明自动标注质量；仍必须做 100 条盲法人工校准。

### 泄漏、分布差异和立即下一步

DeepfakeJudge 主要基于单图真实/生成/编辑伪造及人工局部证据训练，而 DataB 是多帧全生成视频，存在图像到视频、局部编辑到全生成和分类体系差异。这里用同标签错配帧控制避免 Judge 仅凭可信标签或候选语言风格打分，但控制胜出率也会受到原始自动解释本身质量影响，因此门失败时需要查看逐条样本，不能直接归因于模型或数据中的单一一方。

立即下一步：服务器运行 200 条门验收并提供 `datab_deepfakejudge_gate_summary.json`。通过后抽取 100 条原始解释做人工盲评并校准接受阈值；未通过则停止全量 Judge，不为 6766 条生成无效分数。

## 18. 完整 DataB 检测模型的 Skyra 风格 GRPO 奖励动力学诊断

### 这个实验测什么

从已经在完整 DataB 上完成检测 SFT 的 Qwen3-VL-8B 出发，在相同数据、采样顺序和训练超参数下，只更换规则奖励函数，观察 GRPO 的分类正确率、Fake 预测比例、FP/FN、格式、证据数量、KL、熵、策略损失、梯度范数和组内奖励方差如何变化。本实验的目标是学习并验证后训练曲线的分析方法，不用于证明相机条件化方法有效。

### 状态、模型与数据

- 日期：2026-07-14。
- 状态：`执行中；数据、verl 补丁、16 图像 vLLM 和真实 RLHFDataset 预检均通过，用户明确跳过 1 步门并直接启动论文式奖励 100 步长程诊断`。
- 起始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 原始训练数据：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`，6766 条，初始 Fake/Real 各 3383 条。
- verl 版本：commit `2c9e19ef2f0619a2e9e9d4fc813dab8e717e3ab9`，服务器目录 `/input/workflow_58770161/workspace/test/cameramotion_det/third_party/verl-2c9e19ef2f0619a2e9e9d4fc813dab8e717e3ab9`。
- 转换规则：17 帧样本确定性均匀取 16 帧；唯一 11 帧样本剔除；类别下采样平衡后，训练集 6252 条，GRPO 过程诊断集 512 条。
- 小型可复用数据与审计保存到 NAS：`/input/workflow_58770161/workspace/test/cameramotion_det/res/skyra_grpo_diagnostics/data`。
- 逐样本 rollout 暂存在 `/tmp/1res/skyra_grpo_diagnostics/<run>/rollouts`；压缩曲线、TensorBoard、运行清单和日志持久化到 NAS 的 `res/skyra_grpo_diagnostics/<run>`。当前不保存 checkpoint，不需要上传 OSS。

### 单一改变因素与控制条件

第一轮只运行论文文字定义的非对称分类与证据计数奖励（`paper_asymmetric_inspection`），先确认完整的 rollout、reward、GRPO advantage、反向更新和指标落盘链路。随后每次只改变奖励定义，并保持模型起点、DataB split、seed、prompt、采样组大小、优化器和步数一致：

- 对称零误判奖励：检验论文所述的 Fake 偏置。
- 只奖励答案：隔离证据计数项的作用。
- 外层格式替代证据计数：区分格式遵循与证据检查行为。
- 严格唯一证据奖励：只有分类正确且类别、时间和 bbox 有效、去重后的证据才得分。
- 只计可命中证据块：主动暴露重复框带来的 reward hacking。
- 公开仓库实际逻辑复现：仅用于识别公开实现与论文描述不一致造成的曲线差异，不作为正式训练方案。

### 主要训练设置

- 16 张 96 GB GPU；actor 使用 16 卡 FSDP，rollout 使用 vLLM、tensor parallel 2。
- 冻结视觉塔，训练语言模型；学习率 `5e-7`，GRPO，KL loss 系数 `0.02`，每个 prompt 采样 4 个回答的一步门，短程诊断再改为 8 个回答。
- 一步门：prompt batch 16、64 条轨迹、最多 512 个回复 token、只运行 1 个 optimizer step。
- 短程诊断：prompt batch 16、128 条轨迹、最多 768 个回复 token、默认 40 步。
- 推理图像上限 16，`max_pixels=262144`；`max_num_batched_tokens=6144`，开启 chunked prefill。
- 不在第一轮保存 checkpoint；逐步保存 rollout 和完整 TensorBoard 曲线，用于判断正确率、偏置、奖励饱和、零方差组、KL 与优化稳定性。

### 验收标准

一步端到端门必须同时满足：verl 数据预检通过；Fake/Real 样本均能经 Qwen3-VL processor 形成四通道 position ids；16 个图像输入均被保留；16 卡训练进程完成 1 步且退出码为 0；生成 rollout、reward 分量、组内零方差率、KL、policy loss 和 grad norm 均能落盘；关键数值不是 NaN/Inf。

通过一步门只说明基础设施与奖励合同正确，不说明 GRPO 有效。进入 40 步后，主要观察奖励增长是否由分类正确率带动，还是由 Fake 比例、重复框数、错误回答仍获正奖励率带动；若后者上升，则把它判定为奖励偏置或 reward hacking，而不是训练改善。

### 泄漏、分布差异与立即下一步

512 条诊断集从本轮 GRPO 更新中留出，但继承的 DataB SFT checkpoint 已经看过这些样本，因此它不是严格 held-out 测试集，只能用于观察策略更新前后的行为变化。训练与诊断都来自完整生成视频 DataB，也不能据此证明对 DataA 局部编辑或 VIF-Bench 的泛化。该实验不使用 camera 数据，不应写成相机条件化方法结果。

2026-07-14 执行变更：DataB 构建、verl 三项兼容补丁、16 图像 vLLM smoke 和真实 `RLHFDataset` 预检均已通过。用户因约 8 小时无人值守且允许训练继续运行，明确选择跳过 1 步 optimizer smoke，并把 `paper_asymmetric_inspection` 从 40 步延长为 100 步长程诊断。该变更增加了首步训练失败后浪费算力以及后程 reward hacking 的风险，但不改变数据、奖励、seed、组大小或优化参数。

立即下一步：等待 100 步运行完成，保留每步 TensorBoard、训练日志、奖励分量、组内零方差率和逐样本 rollout；训练结束后自动上传 rollout 到 OSS，再执行 `/input/training/keep.sh`。次日先审计早期、中期和后期曲线及 reward hacking 指标，不立即把同一配置延长到完整 epoch。

### 2026-07-14 论文式非对称奖励 100 步结果

这次实际测试的是：继承完整 DataB 检测 SFT checkpoint 后，使用 `0.8 × 非对称真假奖励 + 0.2 × 原始证据块计数奖励` 做 100 步 GRPO，训练过程是否稳定，以及奖励提高究竟来自分类、证据格式还是可被利用的计数代理。它没有保存终点 checkpoint，也没有在固定外部测试集上比较训练前后，因此不能证明检测泛化提升。

结果来源：

- 服务器 TensorBoard：`/input/workflow_58770161/workspace/test/cameramotion_det/res/skyra_grpo_diagnostics/paper_asymmetric_inspection_formal_100step/tensorboard/`。
- 本地 event 副本：`E:/newgaibeishi/newtest1/grpo_tensorboard/tensorboard/`。
- 本地曲线、完整标量与统计：`E:/newgaibeishi/newtest1/grpo_tensorboard/analysis_20260714/`。

100 步均有完整标量，每步 16 个 prompt、每个 prompt 采样 8 次，共 12,800 条 rollout；对应 1,600 个 prompt 位置，约为 6,252 条训练集的 25.59%，不是完整一轮数据。

| 指标 | 全程均值 | 前 10 步均值 | 后 10 步均值 | 后 10 - 前 10 |
|---|---:|---:|---:|---:|
| 总奖励 | 0.9679 | 0.9511 | 0.9819 | +0.0307 |
| Rollout 分类正确率 | 96.12% | 95.08% | 96.64% | +1.56 点 |
| 非对称分类奖励分量 | 0.9584 | 0.9466 | 0.9636 | +0.0170 |
| 证据计数奖励分量 | 1.0058 | 0.9695 | 1.0550 | +0.0856 |
| 每回答原始证据块数 | 1.7738 | 1.6852 | 1.8945 | +0.2094 |
| 错误回答仍获正奖励率 | 3.48% | 3.91% | 3.05% | -0.86 点 |
| 零组内奖励方差 prompt 比例 | 52.56% | 50.63% | 65.63% | +15.00 点 |
| 平均组内奖励标准差 | 0.05965 | 0.06393 | 0.04201 | -0.02192 |
| 平均回复长度/token | 181.51 | 175.90 | 193.64 | +17.74 |
| 参考策略 KL loss | 0.000191 | 0.000089 | 0.000398 | +0.000310 |

12,800 条 rollout 中有 12,303 条分类正确、179 条 Real→Fake、312 条 Fake→Real、6 条答案不可解析；预测 Fake 比例为 49.60%，GT Fake 比例为 50.69%，没有观察到单边 Fake 塌缩。格式有效率为 99.95%。梯度范数均值为 1.168，除 step 24 的 5.795 单点外没有持续放大；熵从前 10 步均值 0.2145 轻微升至后 10 步 0.2226，KL 平滑上升且绝对值仍小，没有数值发散或策略熵塌缩。

总奖励前后窗口增加 0.0307，其中约 44.3% 来自分类奖励变化，约 55.7% 来自证据计数奖励变化。原始证据块与结构有效且去重后的证据块同步增加，重复块均值仅 0.0091、非法块均值仅 0.0013，说明当前没有通过大量重复标签或非法坐标制造奖励；但奖励函数不核对证据是否真的命中视觉伪影，所以证据块从 1.69 增至 1.89、回复长度同步增加 17.74 token，仍属于明显的“优化可计数代理”，不能解释为证据质量改善。

全程 497 条错误回答中有 446 条仍得到正的绝对奖励，占错误回答的 89.74%。这并不等于它们在 GRPO 中一定获得正 advantage，因为 GRPO 使用同一 prompt 组内相对标准化；但它说明原始证据计数项没有被分类正确性门控，错误答案也可用格式正确的证据块抬高分数。该比例没有随训练继续上升，因此 100 步内没有出现失控式 reward hacking，但奖励合同存在明确泄漏。

平均只有 47.44% 的 prompt 组具有非零组内奖励差异，即每步 16 组中平均约 7.59 组提供有效相对学习信号；后 10 步降为每步约 5.5 组。`ppo_epochs=1` 时当前策略在第一次前向与 old policy 相同，所以 `ppo_kl=0` 和 `clipfrac=0` 是本配置的结构性结果，不能用来证明更新幅度为零；参考模型 KL loss 的平滑上升才是策略逐渐偏离起点的有效指标。

结论标记：`通过（仅指 100 步训练链路和奖励动力学诊断）`。原奖励作为可直接延长的正式训练配方不通过：训练很稳定，但起点分类能力已经很高，后程越来越多组没有 advantage，奖励提高又主要依赖不验证视觉真实性的证据计数。当前结果不支持把同一配置延长到完整 epoch，也不支持声称 GRPO 改善了检测能力。

立即下一步：固定同一起点、seed、数据顺序和 40 步预算，优先比较“只奖励真假答案”与“分类正确后才奖励结构有效且去重证据”两个分支；每个分支必须保存终点 checkpoint，并在固定的 512 条诊断集和小型外部分布子集上做训练前后同协议评测。只有当组内非零方差保持、固定评测正确率不下降且证据质量指标提高时，才考虑延长训练。

## 记录维护说明

- 新实验开始时先在本文件新增中文实验定义和验收标准。
- 用户提供结果后，在对应小节补充指标、结论和下一步，不创建含义重复的新代号章节。
- 未知值保留为 `待补充`，不根据上下文猜测。
- `docs/final_experiment_plan_20260708.md` 是受保护文件，不在本记录维护过程中修改。
