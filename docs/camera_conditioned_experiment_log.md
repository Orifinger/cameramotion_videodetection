# 相机条件化 AIGC 视频检测实验记录

本文件是当前论文项目的统一实验记录。目标是让新会话不依赖代号或聊天记忆，也能看懂每个实验具体测了什么、使用了哪些数据、得到什么结果，以及下一步为什么这样安排。

## 当前实验索引

| 日期 | 中文实验名称 | 状态 | 这个实验测什么 | 当前结论 |
|---|---|---|---|---|
| 2026-07-08 | 完整 DataB 检测模型的 VIF-Bench 基线 | 已完成；训练来源审计已补充 | 仅使用自动标注检测数据训练后，在通用全生成视频测试集上的检测能力 | DataB 为 4000 条 ViF-CoT-4K 加 2766 条 GenBuster；ViF-Bench 只作开发 benchmark，计划使用的独立 GenBuster-Bench `benchmark` 集须先通过精确零重叠审计 |
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
| 2026-07-13 | 有序抽帧二元相机辅助与检测回放联合训练三分支验证 | DataA 与 ViF-Bench 检测迁移均未通过；当前配方停止 | 在相同有序抽帧输入和检测回放下，正确二元相机监督是否比逐条翻转监督及仅检测对照学到视觉相关能力，同时保留检测接口 | ViF-Bench 上正确相机辅助 Balanced ACC 为 76.81%，低于仅检测续训的 77.18%、翻转控制的 77.22% 和原模型的 79.18%；独立相机 VQA 与检测回放交错 SFT 没有形成检测正迁移 |
| 2026-07-15 | 正确相机二元前置强化学习与分阶段检测恢复 | 主实验计划撤回，保留为辅助消融 | 从已经通过视觉依赖门的正确相机联合 SFT 模型出发，短程 Camera-PPRL 是否能在无相机文本推理时改善 ViF-Bench；随后检测回放能否恢复检测且保留相机能力 | 相机-only 奖励没有直接包含 Real/Fake，也没有要求检测决策使用相机中间变量；在执行前撤回主实验地位，避免把优化器变化误当任务耦合 |
| 2026-07-15 | 检测主导的相机中间变量联合 SFT/GRPO 三对照门 | 代码与本地真实数据 dry-run 通过，待服务器执行 | 在同一次生成中先预测相机运动再输出 Real/Fake，并让检测正确奖励主导整条 rollout，正确相机奖励是否优于等算力的仅检测奖励和打乱相机奖励 | 1024 条训练记录中 DataA/DataB 各 512、Real/Fake 各 512；打乱标签改变 99.22% 样本且保持 `source × Real/Fake` 标签边际；DataA 作局部诊断，ViF 的 Real/Fake 三对照是开发主门 |
| 2026-07-15 | 三分类相机运动硬路由检测专家验证 | 未通过；停止三专家训练 | 在不向检测 prompt 提供相机文字时，按同一 16 帧预测的无运动/轻微运动/复杂运动选择 detection 专家，是否优于同数据共享模型、同协议原始模型和循环错误路由 | held-out DataA 总体 ACC 73.46%、macro recall 58.64%、pair consistency 92.59%，但轻微运动 recall 仅 4.90%，未达到每桶至少 40% 的预设门槛；三分类中间桶塌缩，不能进入四分支检测训练 |
| 2026-07-15 | 静止/有运动二路硬路由复核 | 通过 | 不重新训练或推理，把冻结三分类 top-1 固定映射为静止与有运动，检验塌缩是否来自不合理的中间硬类别 | ACC 83.80%、Balanced ACC 83.96%，静止/有运动 recall 分别为 84.21%/83.71%，pair consistency 95.37%；三个 VACE 来源均稳定，real/fake 路由分布 TV 仅 0.31%，允许进入二专家检测门 |
| 2026-07-15 | 二路相机硬路由检测专家门 | 未通过；停止硬路由主线 | 冻结视觉 Router 后，静止/有运动检测专家是否在无 camera 文本的 ViF-Bench 上优于同数据共享模型、原始模型和交换错误路由 | 正确路由 Balanced ACC 74.50%，低于原始模型 79.18%、共享模型 76.30% 和交换错误路由 78.03%；19 个生成器中仅 1 个胜出，全部预注册检测门失败 |
| 2026-07-17 | 二路检测专家交叉离线诊断 | 诊断完成；确认专家语义反转 | 在不训练和不重新推理的情况下，分别比较两个专家在静止/有运动路由子集上的表现，定位错误路由胜出的原因 | 静止子集上有运动专家 Balanced ACC 高 4.23 点，有运动子集上静止专家高 3.56 点；对应生成器胜负为 18/19 和 16/19，说明相机分桶形成了反向而非预期的检测专门化 |
| 2026-07-18 | DataB 显式 Camera labels+caption 配对检测 SFT | 未通过；显式相机文本条件路线停止 | 从同一 Qwen3-VL-8B-Instruct 出发，在完全相同的 5739 条 DataB 上训练 5 epoch，唯一改变是 user prompt 是否追加匹配的 CameraBench labels+caption，检验显式相机条件能否提高 ViF-Bench 最终 Real/Fake 指标 | 训推 prompt 与相机 sidecar 契约一致且覆盖 3160/3160；Camera 分支 Balanced ACC 76.42%、Fake F1 75.75%，均低于无 Camera 的 79.09%/79.44%，且 19 个生成器中仅 2 个胜出 |
| 2026-07-19 | DataB 到 ViF-Bench 的相机条件化几何残差最小验证 | 待执行；数据审计与代码已完成 | 不使用 DataA、CoT 或相机文本，检验正确相机几何补偿后的残差是否稳定优于原始运动和错配几何控制 | DataB 去重后 5639 个可用视频，约 77.7% 含运动；相机桶单独可达 57.08% Balanced ACC，已按来源、真假和运动桶控制偏置，等待冻结特征门结果 |
| 2026-07-13 | DataB 自动解释的 DeepfakeJudge-7B 可靠性门 | 代码已就绪，待服务器执行 | 专用开源深伪解释 Judge 在 DataB 上是否真正依据有序帧、bbox、时间和类别评价自动 CoT，而不是只评价语言流畅度 | 先做 200 条分层样本及视觉错配控制；通过后才进入人工校准和全量筛选 |

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

### 2026-07-14 DataB 训练来源路径审计

这次实际检查的是本地正式训练文件 `ourdata/dataB/v4vif_2766busterall_trainall.json` 中每条记录第一帧的来源路径，目的是判断现有 benchmark 是否与起始 detection checkpoint 的训练视频重叠。该审计只检查数据谱系，不评价模型性能。

| DataB 来源 | 记录数 |
|---|---:|
| ViF-CoT-4K | 4000 |
| GenBuster 合计 | 2766 |
| GenBuster `train` 路径 | 1524 |
| GenBuster `test` 路径 | 1242 |
| 总计 | 6766 |

结论标记：`通过（数据谱系审计）`。未经排除精确视频 ID 的 GenBuster test 不能作为当前 checkpoint 的外部测试，因为至少 1242 条训练记录直接来自其 `test` 路径。ViF-CoT-4K 与 ViF-Bench 是 Skyra 定义的训练集/benchmark 组合，仍需对本地训练帧目录与本地 ViF-Bench test index 做精确 case ID 或视频哈希重叠审计；即使没有重叠，ViF-Bench 已在本项目中反复用于方法选择，因此后续定位为开发 benchmark，而不是未揭盲的最终测试集。

2026-07-14 更正：计划用于论文评测的是 GenBuster-200K 中单独发布的 `benchmark` 集（GenBuster-Bench），不是上述 `train/test` 目录。上一段关于训练泄漏的否定只适用于直接拿已进入 DataB 的 `test` 视频评测，不能据此否定独立 `benchmark` 集。正式使用前仍必须比较 DataB 训练视频与 benchmark 的稳定视频 ID，并在可行时补内容哈希审计；零重叠后可将其作为外部测试。为避免继续发生测试集调参，ViF-Bench 用于开发筛选，GenBuster-Bench 应尽量保留到方法冻结后评测，并分开报告其 In-Domain、Out-of-Domain 与 In-the-Wild 轨道。

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
- 状态：`三个分支训练与相机能力验收完成；无 camera 文本 DataA 检测迁移未通过；本配方停止 VIF-Bench 与 RL`。
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

### 2026-07-13 正式数据构建与注册审计

这次实际测试的是：正式 40step_v3 DataA、DataB detection 和 DataB camera 分层信息能否构造出无 case 泄漏、三分支等量、相机与检测为 1:1、图片路径完整且可被当前 LlamaFactory 注册的训练文件。它只验证数据工程与实验对照成立，不验证模型已经学到相机能力或检测得到提升。

结果来源：

- 临时完整审计：`/tmp/1res/camera_joint_sft_gate/data/camera_joint_sft_data_summary.json` 与 `/tmp/1res/camera_joint_sft_gate/data/llamafactory_install_summary.json`；
- NAS 小结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/`；
- LlamaFactory 数据目录：`/input/workflow_58770161/workspace/test/test_selfcot/Skyra/train/LLaMA-Factory/data`。

| 数据审计项 | 正式结果 |
|---|---:|
| DataA case split | 1080 total；756 train；324 test；交集 0 |
| 仅检测回放分支 | 11160 条 detection |
| 正确相机监督分支 | 5580 detection + 5580 camera，共 11160 条 |
| 翻转相机监督分支 | 5580 detection + 5580 camera，共 11160 条 |
| DataB 平衡 detection replay | 1512 条 |
| DataA train detection | 1512 条 |
| 为达到 1:1 所做的 detection 确定性过采样 | 2556 条 |
| 三分支记录数相等 | 是 |
| 图片存在性检查 | 通过 |
| 每分支唯一图片路径 | 48218 |
| 每分支 smoke 数据 | 96 条 |

三个正式文件 SHA-256 分别为：仅检测 `f761e6453d09c54b376f1b3bc782cee583753a8f07c3f988de3e2dd31a2ef0e5`、正确相机 `97194b4c23955ff2bfc4b7824e6016a32f1f2168a2b4a06e96c14434880d40d9`、翻转相机 `54fd25135441638af3330b038f0cea187d61dd326daca90e37a2ccc56c6f420c`。三个分支的 source counts 与预先定义一致，正确和翻转分支任务比例均为严格 1:1，且 `equal_branch_sizes=true`、`images_checked=true`。

结论标记：`通过（数据构建与注册）`。这建立了第一轮训练输入和控制条件可执行；它不建立正确监督优于翻转监督、视觉依赖、检测接口保留或 RL 可训练性。

### 2026-07-14 两步 LoRA 工程 smoke

这次实际测试的是：正确相机联合分支的 96 条 smoke 数据能否在现有 Qwen3-VL-8B 检测 checkpoint 上完成图像预处理、分布式前向、反向传播和 LoRA 优化器更新。它只检查训练链路可执行，不评价 loss 收敛、相机能力或检测效果。

结果来源：`/tmp/1res/camera_joint_sft_gate/smoke/trainer_log.jsonl`。该目录是一次性 smoke 输出，保留在易失 `/tmp`，不上传 NAS 或 OSS。

| Smoke 指标 | 结果 |
|---|---:|
| 计划/完成 optimizer steps | 2 / 2 |
| Step 1 loss / LR | 2.4283 / `5.0e-6` |
| Step 2 loss / LR | 3.0638 / `2.5e-6` |
| 训练步骤耗时 | 10 秒 |
| 总运行耗时 | 16 秒 |
| OOM、异常退出或非有限 loss | 0 |

结论标记：`通过（工程 smoke）`。两步内 loss 不单调不构成异常，因为样本和任务不同且 smoke 不用于判断收敛；该结果足以解除正式训练的工程阻塞，但不建立任何模型效果结论。

### 2026-07-14 正确相机与翻转相机分支正式训练完成性

这次实际测试的是：在相同检测起点、相同 5580 条检测 replay、相同 5580 个相机任务槽和相同优化设置下，正确 Yes/No 相机监督与逐条翻转监督两个分支能否完整执行 5 epochs。这里只检查训练是否按计划结束以及 loss 是否有限，不比较泛化能力。

结果来源：

- 正确相机监督：`/tmp/1res/camera_joint_sft_gate/train/correct_camera/trainer_log.jsonl`；
- 翻转相机监督：`/tmp/1res/camera_joint_sft_gate/train/shuffled_camera/trainer_log.jsonl`；
- 串行调度日志：`/tmp/1res/camera_joint_sft_gate/correct_then_shuffled_launcher.log`。

| 正式训练指标 | 正确相机监督 | 翻转相机监督 |
|---|---:|---:|
| 计划/完成 optimizer steps | 3490 / 3490 | 3490 / 3490 |
| 完成 epochs | 5.0 | 5.0 |
| 最后一个已报告 loss | 0.0024 | 0.0051 |
| 最后一个已报告 LR | `4.3068e-11` | `4.3068e-11` |
| 训练运行时间 | 3:17:28 | 3:17:27 |
| train samples/s | 待补充 | 4.71 |
| train steps/s | 待补充 | 0.295 |
| OOM、异常退出或非有限 loss | 0 | 0 |

两个分支完成后串行脚本已进入 `/input/training/keep.sh`。结论标记：训练完成性为`通过`，方法效果为`结论不足`。正确和翻转分支都获得很低的训练 loss，说明两套目标均可被拟合；这不证明正确监督利用了视频，也不能用两者训练 loss 大小判断方法优劣，必须在同一 held-out 相机条件上比较 AP、Balanced ACC 和视觉控制。

正式 adapter、epoch checkpoints 和完整训练日志仍位于易失 `/tmp`。权重尚未通过相机与 DataA 审计，因此当前不上传 OSS；两个小型 `trainer_log.jsonl` 应复制到 NAS 训练记录目录。立即下一步是先核验 final adapter 文件完整并持久化小日志，再训练仅检测回放对照，随后统一运行三个分支的相机评测。

#### 2026-07-14 完整训练日志与 adapter 文件复核

补充结果来源：本地附件 `E:/newgaibeishi/trainerlogs.zip`，包含正确相机与翻转相机两个完整 `trainer_log.jsonl`；服务器文件审计来自对应 `/tmp/1res/camera_joint_sft_gate/train/` 目录。

| Epoch 末附近指标 | 正确相机 loss | 翻转相机 loss |
|---|---:|---:|
| Epoch 1，step 690 | 0.1774 | 0.1856 |
| Epoch 2，step 1390 | 0.1015 | 0.1110 |
| Epoch 3，step 2090 | 0.0520 | 0.0551 |
| Epoch 4，step 2790 | 0.0136 | 0.0208 |
| Epoch 5，step 3490 | 0.0024 | 0.0051 |

| 完整日志审计 | 正确相机监督 | 翻转相机监督 |
|---|---:|---:|
| JSONL 总记录 / loss 记录 | 350 / 349 | 350 / 349 |
| 首个 step 10 loss | 4.7091 | 4.7648 |
| 全程最小 / 最大 loss | 0.0016 / 4.7091 | 0.0024 / 4.7648 |
| 非有限 loss | 0 | 0 |
| Final adapter config/weights | 均存在 | 均存在 |
| 目录大小 | 14G | 14G |
| Epoch checkpoints | 698、1396、2094、2792、3490 | 698、1396、2094、2792、3490 |

复核结论仍为：训练完成性`通过`，方法效果`结论不足`。两条学习曲线的步数、LR 和耗时严格对齐，且都能拟合各自监督，说明翻转分支是有效的等计算量反事实控制；它没有训练失败。只有正确分支在 held-out 正确帧上显著优于翻转分支，并在相反答案帧或无帧控制上退化，才能建立视觉相机能力结论。

#### 2026-07-14 执行顺序更正：先做双分支 held-out 相机门

状态：`代码已就绪，待执行`。在训练仅检测回放分支之前，先用已经完成的正确相机和翻转相机两个 final adapter 做低成本相机能力门。更正原因是仅检测分支不参与“正确监督是否优于错误监督”和“回答是否依赖画面”这两个前置判断；若前置门失败，第三次 5-epoch 训练没有继续价值。

- 模型谱系：共同起点仍为 `/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`，分别挂载 `/tmp/1res/camera_joint_sft_gate/train/correct_camera` 与 `/tmp/1res/camera_joint_sft_gate/train/shuffled_camera`。
- 评测数据：正式 70:30 case split 中未参与本轮联合训练的 DataA test camera 二元问题，具体文件为 `/tmp/1res/camera_joint_sft_gate/data/camera_dev_matched_frames.jsonl`、`camera_dev_opposite_frames.jsonl` 和 `camera_dev_no_frames.jsonl`。
- 单一改变因素：正确与翻转模型在 matched frames 上比较；正确模型额外在相反答案帧和无帧上测试视觉依赖。三个条件是不同实验输入，不混称 camera-conditioned 结果。
- 推理设置：16 GPU 分片、相同 16 帧、`image_max_pixels=262144`，直接比较 `Yes`/`No` token logit，不生成外部 camera caption，也不执行 AIGC detection prompt。
- 验收标准：两个模型覆盖率均至少 99%、支持 primitive 至少 20；正确模型相对翻转模型 Macro AP 至少高 3 点或 Balanced ACC 至少高 5 点；正确模型 matched 相对 opposite Balanced ACC 至少高 10 点，或相对 no-frame 至少高 8 点。
- 输出位置：逐样本打分是一次性验证数据，放 `/tmp/1res/camera_joint_sft_gate/camera_predictions`；正式小型评测与汇总 JSON 放 `/tmp/1res/camera_joint_sft_gate/camera_eval` 并复制到 NAS `res/camera_joint_sft_gate/camera_eval/`。
- 泄漏与限制：这是本轮 case-level train/test 隔离的开发留出集，但项目已反复使用 DataA 做方案选择，因此不称全新论文 test；该门不测 Real/Fake 检测提升或检测输出格式保留。

立即下一步：运行双分支相机门。通过后先做正确分支 `pass@8` 奖励方差检查，再训练仅检测回放分支；未通过则停止第三次训练并检查逐类相机指标。

#### 2026-07-14 双分支 held-out 相机门结果

这次实际测试的是：两个从同一检测 checkpoint 出发、训练计算量相同的联合 LoRA，唯一改变相机辅助目标是正确还是逐条翻转；在 case-level 隔离的 DataA 开发集上，比较 matched 画面的二元相机得分，并对正确模型施加相反答案画面和无画面控制。

结果来源：`/tmp/1res/camera_joint_sft_gate/camera_eval/correct_vs_flipped_camera_gate_summary.json`；持久化副本为 `/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/camera_eval/correct_vs_flipped_camera_gate_summary.json`。

| 模型与输入条件 | Coverage | Balanced ACC | Overall AP | Macro AP | ROC-AUC | Pair ACC |
|---|---:|---:|---:|---:|---:|---:|
| 正确监督模型，matched 画面 | 100.00% | 74.44% | 83.54% | 86.28% | 84.32% | 52.67% |
| 翻转监督模型，matched 画面 | 100.00% | 30.22% | 35.66% | 40.15% | 22.86% | 9.56% |
| 正确监督模型，相反答案画面 | 100.00% | 25.56% | 33.25% | 36.34% | 15.68% | 3.78% |
| 正确监督模型，无画面 | 100.00% | 50.00% | 50.00% | 50.00% | 50.00% | 0.00% |

| 预注册关键差值 | 结果 | 阈值 |
|---|---:|---:|
| 正确减翻转 Macro AP | +46.13 点 | 至少 +3 点 |
| 正确减翻转 Balanced ACC | +44.22 点 | 至少 +5 点 |
| matched 减相反答案画面 Balanced ACC | +48.89 点 | 至少 +10 点 |
| matched 减无画面 Balanced ACC | +24.44 点 | 至少 +8 点 |

结论标记：`通过`。正确监督显著优于错误监督，替换为相反标签对应画面后性能反转，无画面时回到随机水平，因此本轮联合 SFT 确实学到了依赖当前视频画面的相机运动映射，而不是只学 Yes/No 格式或标签边际。Pair ACC 仅为 52.67%，表明逐问题排序能力强但成对全部答对仍有空间；本结果不建立 Real/Fake 检测提升、检测格式保留或外部 CameraBench 泛化。

2026-07-14 数据描述更正：早期记录将输入简称为“同 16 帧”，但评测错误栈和重建审计确认个别样本的有效抽帧数并非固定 16。已修复相反画面控制中交换路径但未同步 `<image>` 数量的问题，并增加 token/path 强制审计；训练目标和两个已完成 adapter 未改变。后续统一称“同一有序抽帧序列”，不把变长样本误写成固定 16 帧。

立即下一步：先运行正确监督模型的 `pass@8` 采样和奖励方差检查。该检查只决定短程 RL 是否具有可探索信号；无论是否适合 RL，仍需训练等记录数、等步数的仅检测回放对照，之后才能比较相机辅助是否保留并改善 DataA/VIF-Bench 检测。

#### 2026-07-14 正确监督模型的 GRPO 前置采样结果

这次实际测试的是：对 900 条 held-out 二元相机问题各采样 8 次，在当前严格“回复必须恰好为 Yes 或 No”的奖励接口下，检查格式通过、正确答案覆盖、两种动作探索和组内奖励方差。结果来源为 `/tmp/1res/camera_joint_sft_gate/rl_readiness/correct_camera.json`；小型正式结果已由脚本复制到 NAS `res/camera_joint_sft_gate/rl_readiness/correct_camera.json`。

| 指标 | 结果 | 阈值/期望 |
|---|---:|---:|
| 样本覆盖率 | 100.00% | 至少 99% |
| 每题 8 次采样完整率 | 100.00% | 100% |
| Format pass@8 | 0.00% | 至少 90% |
| Correct-answer pass@8 | 0.00% | 至少 50% |
| 同题同时探索 Yes/No | 0.00% | 至少 10% |
| 组内奖励非恒定率 | 0.00% | 至少 20% |
| 平均组奖励 | 0.0000 | 待探索 |
| 平均唯一回复数 | 1.1256 | 待探索 |

结论标记：`未通过`。当前生成式接口下 7200 次 rollout 全部没有被严格解析成单独的 `Yes/No`，因此奖励统一为 0，不能直接启动 GRPO。该结果与上一门的首 token logit 排序能力不矛盾，也不能归因为相机语义能力消失；在查看原始 `response` 之前，尚不能区分带标点/模板、沿用 detection CoT 并被 8-token 截断、解码切分错误或真正的生成动作塌缩。

立即下一步：统计 raw rollout 的回复频次、前缀、长度和截断形态。若只是格式或 8-token 截断，则修正采样协议后重算同一门，不重新训练；若回复确实是单一错误动作且增加生成长度也无探索，则停止当前 SFT 起点上的 GRPO。仅检测回放对照不依赖此 RL 结论，仍按原计划训练。

2026-07-14 根因更正：raw rollout 共 7200 条，全部为 `<think>\n\n</think>\n\nYes` 或对应的 `No`，没有空回复、截断 CoT 或无关文本。频数为 Yes 3680、No 3520；按 gold 交叉计数为 Yes→Yes 2794、Yes→No 806、No→No 2714、No→Yes 886，逐 rollout 语义正确率 76.50%。因此前述 `not_ready` 不能作为有效 RL 结论：失败来自评测器只接受裸 `Yes/No`，而模型稳定输出 Qwen thinking 模板的空包装。解析器已收紧修正为只额外接受“空 think 包装 + Yes/No”，不接受非空 reasoning、标点或其他附加文本；下一步直接复用现有 rollout 重算，不重复 GPU 推理。

2026-07-14 修正解析后的最终结果：

| 指标 | 修正结果 | 预注册阈值 |
|---|---:|---:|
| Coverage / 每题 8 次完整率 | 100.00% / 100.00% | 至少 99% / 100% |
| Format pass@8 | 100.00% | 至少 90% |
| Correct-answer pass@8 | 82.78% | 至少 50% |
| 同题同时探索 Yes/No | 12.56% | 至少 10% |
| 组内奖励非恒定率 | 12.56% | 至少 20% |
| 平均组奖励 | 0.7885 | 待探索 |
| 平均唯一回复数 | 1.1256 | 待探索 |

结论标记：`结论不足（borderline）`。格式、正确答案覆盖和双动作探索均通过，但只有 113/900 左右的问题组产生奖励方差，约 87.44% 的标准 GRPO group 不提供学习信号；因此当前起点不支持直接投入完整 RL，只允许在下游检测 SFT 已证明有价值之后做预先限定步数的短程验证。当前主线先训练等计算量的仅检测回放对照，并比较三分支的相机能力和无 camera 文本检测结果；不因 RL borderline 延迟该必要对照。

#### 2026-07-14 仅检测回放对照训练完成

这次完成的是第三个等记录数、等步数控制分支：在与两个相机分支相同的检测 replay 槽之外，辅助槽继续使用检测数据，不加入正确或翻转相机监督。它用于控制额外训练步数和额外 detection replay 本身的影响。

结果来源：服务器 `/tmp/1res/camera_joint_sft_gate/train/detection_only/` 文件审计。

| 完成性项目 | 结果 |
|---|---|
| Final `adapter_config.json` | 存在 |
| Final `adapter_model.safetensors` | 存在 |
| 训练目录大小 | 14G |
| Epoch checkpoints | 698、1396、2094、2792、3490 |
| Final step | 3490 |
| Final loss | 0.0003（step 3490） |
| 完整训练耗时 | 3:18:17 |

结论标记：训练完成性`通过`，方法效果`结论不足`。第三个控制分支完成 3490/3490 步和 5.0 epochs，末次 loss 为 0.0003，总耗时 3:18:17，与另外两个分支的步数和耗时对齐；但尚未提供相机指标或 Real/Fake 检测指标，不能仅根据低训练 loss 推断它优于或劣于相机辅助分支。

立即下一步：只运行仅检测回放模型的 held-out 相机 logit 评测，复用已有正确/翻转模型结果生成三分支汇总；随后进入无 camera 文本 DataA 检测比较。当前三个 14G adapter 仍在易失 `/tmp`，待三分支相机和 DataA 审计后再决定上传 OSS 的正式保留集合。

2026-07-14 三分支汇总实现审计更正：在执行仅检测回放模型评测前发现旧汇总器虽然展示该模型指标，却没有把“正确相机监督优于仅检测回放”设为硬检查。现已在不查看该模型结果的前提下补充预先门槛：正确监督相对仅检测回放 Macro AP 至少提高 3 点，或 Balanced ACC 至少提高 5 点；并继续要求其超过翻转监督。更正原因是第三个分支必须真正控制额外训练步数与额外 detection replay，而不能只作为表格展示项。

#### 2026-07-14 三分支 held-out 相机能力最终汇总

这次实际测试的是：三个从同一 DataB detection checkpoint 出发、记录数和训练步数相同的联合 LoRA，在同一 case-level 隔离 DataA 相机开发集上的二元相机识别能力。唯一变化是辅助槽继续使用 detection、使用正确相机标签，或使用逐条翻转相机标签；检测推理和 AIGC 标签没有参与本结果。

结果来源：`/tmp/1res/camera_joint_sft_gate/camera_eval/joint_sft_camera_gate_summary.json`；持久化副本位于 `/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/camera_eval/joint_sft_camera_gate_summary.json`。

| 模型/条件 | Balanced ACC | Overall AP | Macro AP | ROC-AUC | Pair ACC |
|---|---:|---:|---:|---:|---:|
| 仅检测回放，matched | 62.00% | 65.32% | 71.80% | 66.57% | 36.44% |
| 正确相机监督，matched | 74.44% | 83.54% | 86.28% | 84.32% | 52.67% |
| 翻转相机监督，matched | 30.22% | 35.66% | 40.15% | 22.86% | 9.56% |
| 正确相机监督，相反答案画面 | 25.56% | 33.25% | 36.34% | 15.68% | 3.78% |
| 正确相机监督，无画面 | 50.00% | 50.00% | 50.00% | 50.00% | 0.00% |

| 关键差值 | 结果 | 验收阈值 |
|---|---:|---:|
| 正确减仅检测回放 Macro AP | +14.48 点 | 至少 +3 点 |
| 正确减仅检测回放 Balanced ACC | +12.44 点 | 至少 +5 点 |
| 正确减翻转 Macro AP | +46.13 点 | 至少 +3 点 |
| 正确减翻转 Balanced ACC | +44.22 点 | 至少 +5 点 |
| matched 减相反答案画面 Balanced ACC | +48.89 点 | 至少 +10 点 |
| matched 减无画面 Balanced ACC | +24.44 点 | 至少 +8 点 |

结论标记：`通过`。在控制额外训练步数和 detection replay 后，正确相机监督仍显著超过仅检测回放；逐条翻转监督使映射反转，画面错配和无画面控制也按因果预期退化。因此当前联合 SFT 已建立“相机能力被真实注入且依赖当前视觉输入”，不是模板、标签边际或额外训练计算造成的假增益。`passed_for_short_rl` 只表示相机能力门通过且存在有限 RL 探索，不表示 AIGC 检测方法通过；RL 仍受 12.56% 奖励方差率限制。

立即下一步：固定无 camera 文本和原 detection prompt，在同一 DataA case-level test 上比较 base、仅检测回放、正确相机监督、翻转相机监督四个模型。只有正确相机监督相对仅检测回放在 Balanced ACC、Fake F1 或 Pair ACC 上有稳定正增量且格式不退化，才进入 VIF-Bench；否则先分析 camera auxiliary 与检测决策没有形成迁移的原因，不启动短程 RL。

#### 2026-07-14 DataA 无相机文本四模型检测迁移门定义

- 状态：`代码已就绪，待服务器执行`。
- 具体问题：在不提供 camera caption、camera label 或任何额外相机文本时，正确相机辅助联合 SFT 是否比等计算量仅检测回放和翻转相机辅助更好地检测 DataA 局部编辑视频，同时保持原 Real/Fake 与解释输出接口。
- 模型谱系：共同 base 为 `/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`；三个 final adapter 分别位于 `/tmp/1res/camera_joint_sft_gate/train/detection_only`、`correct_camera` 和 `shuffled_camera`。
- 评测数据：`/tmp/1res/camera_joint_sft_gate/data/dataa_test_detection.json` 中 case-level 隔离的 324 个 DataA cases、648 条 Real/Fake records；训练侧使用同一 split 的 756 个 cases，不允许 case 交叉。
- 单一改变因素：三个联合模型仅改变辅助槽是额外 detection、正确相机二元标签或逐条翻转标签；四个模型使用同一原 detection prompt、同一有序抽帧序列、`image_max_pixels=262144`，推理时均无 camera 文本。
- 主要指标：格式有效率、Balanced ACC、Fake F1、严格 Real/Fake Pair ACC；时间和 bbox IoU 作为解释证据次要指标，不作为本轮唯一通过依据。
- 验收标准：四模型 coverage 至少 99%、格式有效率至少 95%；正确相机监督相对仅检测回放和翻转监督均须在 Balanced ACC 或 Pair ACC 上提高至少 2 点，同时另一主指标与 Fake F1 均不得下降超过 1 点。门通过后再做 paired uncertainty 分析，不用当前开发门替代统计显著性。
- 泄漏与分布限制：DataA test cases 未参加本轮三分支训练，但已被项目多次用于方法诊断，因此属于开发留出而非全新论文 final test；base 已看过完整 DataB，DataB 不能作为 held-out；DataA fake 为局部 VACE 编辑，结论不能直接外推到 VIF-Bench 全生成视频。
- 输出与存储：逐样本预测、三个临时 merged models 放 `/tmp/1res/camera_joint_sft_gate/dataa_four_model_compare`，属于可重建的大型/验证产物，不上传 OSS；汇总 JSON 和小型 eval 文件复制到 NAS `res/camera_joint_sft_gate/dataa_four_model_compare/`。

立即下一步：一次运行 base 与三个联合分支的 DataA 推理和汇总。通过则计算 paired bootstrap/McNemar 并进入 VIF-Bench；未通过则停止 RL/VIF 扩展并分析正确相机能力为何没有迁移到检测决策。

#### 2026-07-14 DataA 无相机文本四模型检测迁移结果

这次实际测试的是：base 与三个等计算量联合 LoRA 在同一 324 个 case、648 条 Real/Fake DataA 开发记录上使用原 detection prompt 推理；所有模型都不接收 camera caption、camera label 或外部相机模型输出。结果来源为 `/tmp/1res/camera_joint_sft_gate/dataa_four_model_compare/dataa_four_model_detection_gate_summary.json`，持久化副本位于 NAS `res/camera_joint_sft_gate/dataa_four_model_compare/`。

| 模型 | Format | Balanced ACC | Fake Recall | Fake F1 | Pair ACC |
|---|---:|---:|---:|---:|---:|
| 原 DataB detection checkpoint | 99.85% | 50.00% | 28.09% | 36.04% | 11.11% |
| 仅检测回放 | 100.00% | 63.43% | 60.19% | 62.20% | 35.80% |
| 正确相机辅助 | 100.00% | 60.03% | 57.72% | 59.08% | 29.01% |
| 翻转相机辅助 | 100.00% | 60.03% | 56.17% | 58.43% | 27.78% |

| 正确相机辅助的差值 | Balanced ACC | Fake F1 | Pair ACC |
|---|---:|---:|---:|
| 相对仅检测回放 | -3.40 点 | -3.12 点 | -6.79 点 |
| 相对翻转相机辅助 | 0.00 点 | +0.66 点 | +1.23 点 |
| 相对原 base | +10.03 点 | +23.04 点 | +17.90 点 |

解释证据也没有支持相机辅助增益：仅检测回放的 mean temporal IoU / bbox IoU / evidence hit 为 0.5159 / 0.3150 / 41.05%，正确相机辅助为 0.5006 / 0.3091 / 39.20%；翻转辅助的 evidence hit 同为 41.05%。

结论标记：`未通过`。正确相机辅助未超过仅检测回放，并在三个主要检测指标上明显下降；它与翻转相机监督的检测结果近似，说明此前已经证明的相机视觉能力没有在原 detection prompt 下形成标签语义相关的检测迁移。正确/翻转模型相对 base 的提升不能归因于相机标签，因为更强的仅检测回放模型表明主要收益来自 DataA detection replay 和继续训练。

该结果足以否定当前等计算量的“独立 detection 样本与独立 camera VQA 样本交错 SFT”配方，但不证明相机信息原则上对 AIGC 检测无用。一个重要限制是正确/翻转分支用一半训练槽学习 camera VQA，而仅检测回放把这些槽继续用于 detection；因此本门严格回答的是“同等训练计算投入 camera VQA 是否优于继续做 detection”，答案是否定的。它同时暴露了任务连接缺失：camera 和 detection 监督共享模型与帧，却没有任何单条训练目标要求模型在检测时调用相机判断。

立即下一步：停止本配方的 VIF-Bench 和短程 GRPO，不通过调温度、增加 RL 步数或继续训练掩盖迁移失败。先复用现有 checkpoint 做低成本训练阶段审计，判断早期 checkpoint 是否存在“相机能力已出现但检测尚未下降”的窗口；若不存在，下一轮只能改为同一检测回答内显式预测相机状态再进行 Real/Fake 判断的联合目标，并重新设置低成本正确/翻转控制门。

#### 2026-07-14 联合训练早期 checkpoint 相机-检测窗口审计定义

- 状态：`代码已就绪，待服务器执行`。
- 具体问题：final checkpoint 的检测负迁移是否主要来自 5 epochs 过度训练；Epoch 1/2 是否存在正确相机监督已经增加相机能力，同时 DataA 检测相对同阶段仅检测回放不降反升的 Pareto 窗口。
- 模型谱系：仅复用 `detection_only/checkpoint-{698,1396}` 与 `correct_camera/checkpoint-{698,1396}`，共同 base 和训练数据不变；第一轮不评测翻转 checkpoint，只有出现候选窗口才补同阶段翻转控制。
- 相机评测：`/tmp/1res/camera_joint_sft_gate/data/camera_dev_matched_frames.jsonl`，比较同 step 正确相机与仅检测回放的 Macro AP、Balanced ACC 和 Pair ACC。
- 检测评测：`/tmp/1res/camera_joint_sft_gate/data/dataa_test_detection.json`，继续使用原 detection prompt 和无 camera 文本条件，比较同 step 的 Balanced ACC、Fake F1、Pair ACC 与格式有效率。
- 单一改变因素：同一 step、相同训练总量下，辅助槽是正确相机 VQA 还是额外 detection；不与 final detection-only 跨 step 直接做主验收。
- 候选标准：正确相机相对同 step 仅检测回放的相机 Macro AP 至少 +3 点或 Balanced ACC 至少 +5 点；DataA 检测 Balanced ACC 或 Pair ACC 至少 +2 点，另一主指标与 Fake F1 不得下降超过 1 点；coverage 至少 99%、格式至少 95%。
- 决策规则：先测 step 698、1396。任一步成为候选，才评测同 step 翻转监督并检查标签语义特异性；两步都没有候选且检测差值没有朝正方向改善，则停止独立任务交错 SFT，不补 step 2094/2792、不跑 VIF/RL。只有差值呈明确改善趋势但尚未过线，才允许补一个 step 2094。
- 泄漏与限制：仍使用已反复诊断的 DataA 开发留出，只用于方法选择；该审计不能提供论文 final test 结论。checkpoint 和 merged model 位于 `/tmp`，原训练目录已要求上传 OSS；本轮新 merged models 可重建，不上传。

立即下一步：在一套 16 GPU 上顺序运行四个早期 adapter 的相机 logit 与 DataA 检测评测，输出 `/tmp/1res/camera_joint_sft_gate/checkpoint_window/checkpoint_window_summary.json`。

#### 2026-07-14 联合训练早期 checkpoint 相机-检测窗口审计结果

这次实际测试的是：在 Epoch 1（step 698）和 Epoch 2（step 1396），正确相机监督是否已经增加同源留出集相机能力，同时相对同 step 仅检测回放控制改善 DataA 局部编辑检测。DataA 检测继续使用原 detection prompt，推理不提供 camera 文本。

结果来源：

- 临时完整汇总：`/tmp/1res/camera_joint_sft_gate/checkpoint_window/checkpoint_window_summary.json`；
- NAS 小结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/checkpoint_window/checkpoint_window_summary.json`。

| Step | 相机 Macro AP：仅检测→正确相机 | 相机 BA：仅检测→正确相机 | DataA BA：仅检测→正确相机 | DataA Fake F1：仅检测→正确相机 | DataA Pair ACC：仅检测→正确相机 | Evidence hit：仅检测→正确相机 |
|---:|---:|---:|---:|---:|---:|---:|
| 698 | 71.78%→81.09%（+9.31） | 62.56%→66.67%（+4.11） | 59.57%→55.86%（-3.70） | 61.01%→62.37%（+1.36） | 29.94%→25.31%（-4.63） | 41.36%→50.31%（+8.95） |
| 1396 | 71.54%→84.58%（+13.03） | 62.11%→66.67%（+4.56） | 63.43%→59.72%（-3.70） | 57.45%→60.87%（+3.42） | 33.33%→29.32%（-4.01） | 34.88%→41.67%（+6.79） |

两个 step 的覆盖率和检测格式有效率均为 100%。正确相机分支在两步都明显增加相机 Macro AP，却同时把 DataA Fake Recall 提高约 9.88/13.27 点、Real Recall 降低约 17.28/20.68 点，因此 Fake F1 和局部证据命中有所提高，但 Balanced ACC 与同一 real/fake pair 同时判对率下降。这说明相机辅助并非对检测输出毫无影响，而是当前独立任务配方主要诱发了偏向 `Fake` 的决策偏置，没有形成更好的真假分离。

结论标记：`未通过`。step 698、1396 均不存在预注册的相机-检测 Pareto 候选，且检测差值没有向正方向改善，因此失败不能主要归因于训练到第 5 epoch 才过拟合；不补 step 2094/2792，也不为当前配方启动 RL。该结论只否定 DataA 同域局部编辑开发门上的当前独立任务交错 SFT，不证明它在 VIF-Bench、GenBuster 或其他全生成外部分布上必然下降。

2026-07-14 范围更正：DataA 留出集在本实验中的作用是检验相机监督能否迁移到有配对、时间与 bbox 真值的局部编辑检测，并用于方法开发；它不是通用全生成 benchmark。DataB 起始 checkpoint 已见过完整 DataB，故不能用 DataB 内部分割作为未见测试。VIF-Bench 等外部全生成数据只能补充回答通用检测保留/泛化，不能替代 DataA 对局部编辑机制与证据定位的检验。

#### 2026-07-15 ViF-Bench 无相机文本四模型开发诊断定义

- 状态：`代码已就绪，待服务器执行`。
- 具体问题：正确相机监督在 DataA 局部编辑检测上产生的负迁移，是否也会出现在全部由生成模型合成的 ViF-Bench；或者全生成视频中的全局运动异常更容易受益于已注入的相机能力。
- 模型谱系：共同起点为 `/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`；比较原起点、`train/detection_only`、`train/correct_camera`、`train/shuffled_camera` 四个最终 5-epoch 模型，不在 ViF-Bench 上挑选早期 checkpoint。
- 测试输入：服务器 `eval/v4train-main/test_index_splits/splits_16` 指向的固定 ViF-Bench 有序帧。四个模型使用同一 system prompt、同一 user 后缀、确定性解码和 `no_camera` 条件；推理不提供 camera label、caption 或其他外部相机文本。
- 单一改变因素：三个联合训练分支训练总记录数和步数相同，辅助槽分别是额外 detection、正确二元相机 VQA 或逐条翻转相机 VQA。正确相机相对仅检测控制检验计算投入价值，相对翻转控制检验正确相机语义特异性。
- 主要指标：按生成器分别计算并宏平均 Balanced ACC、Fake F1、Real/Fake Recall、预测 Fake 比例、覆盖率和格式有效率。
- 候选标准：覆盖率与格式有效率均至少 99%；正确相机相对两个控制都须在 Balanced ACC 或 Fake F1 至少提高 1 点，另一项下降不超过 0.5 点；相对两个控制的生成器级 Balanced ACC 胜率均至少 60%；正确模型 Real/Fake Recall 均至少 45%。
- 泄漏与限制：ViF-CoT-4K 是起点训练来源之一，ViF-Bench 是 Skyra 单独发布的 benchmark，但本项目已反复查看 ViF-Bench，因此本轮只用于开发诊断，不能包装成未揭盲最终测试。GenBuster-Bench `benchmark` 与 MintVid 继续保留到方法冻结后。
- 存储：三个临时合并模型和四份预测属于可重建诊断产物，只放 `/tmp/1res/camera_joint_sft_gate/vif_four_model_compare`，不上传 OSS；评测 JSON、CSV 和紧凑日志自动保存到 NAS `res/camera_joint_sft_gate/vif_four_model_compare`。

立即下一步：先运行无模型加载的 `STAGE=preflight`，通过后运行完整四模型诊断。若正确相机不能同时超过两个控制，停止当前独立 Camera VQA/Detection 交错 SFT，不进入 RL；若通过，再补配对不确定性分析，而不是立即查看保留的最终 benchmark。

#### 2026-07-15 ViF-Bench 无相机文本四模型开发诊断结果

这次实际测试的是：原 DataB 检测模型以及三个等记录数、等步数联合 LoRA，在相同 3160 条 ViF-Bench 输入、相同原检测提示词和无相机文本条件下的 Real/Fake 检测。三个续训分支唯一改变辅助槽是额外检测数据、正确二元相机问答或逐条翻转相机答案。

结果来源：

- NAS 汇总：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/vif_four_model_compare/vif_four_model_detection_gate_summary.json`。
- 终端结果附件：`C:/Users/29499/.codex/attachments/bad42d74-97f2-42c5-85ae-301e2c41e80c/pasted-text.txt`。

| 模型 | 覆盖率 | 格式有效率 | Balanced ACC | Real Recall | Fake Recall | Fake F1 | 预测 Fake 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 DataB 检测模型 | 100.00% | 99.97% | 79.18% | 69.03% | 89.33% | 80.47% | 60.15% |
| 仅检测续训模型 | 100.00% | 99.97% | 77.18% | 63.98% | 90.39% | 79.41% | 63.20% |
| 正确相机辅助模型 | 100.00% | 99.91% | 76.81% | 62.04% | 91.57% | 79.37% | 64.76% |
| 翻转相机监督模型 | 100.00% | 99.94% | 77.22% | 62.90% | 91.54% | 79.54% | 64.32% |

| 受控比较 | Balanced ACC 差值 | Fake F1 差值 | 19 个生成来源上的 Balanced ACC 胜率 |
|---|---:|---:|---:|
| 正确相机辅助减仅检测续训 | -0.38 点 | -0.04 点 | 31.58%（6/19） |
| 正确相机辅助减翻转相机监督 | -0.41 点 | -0.17 点 | 26.32%（5/19） |
| 正确相机辅助减原检测模型 | -2.38 点 | -1.10 点 | 待补充 |

正确相机辅助没有超过任何控制。相对仅检测续训，它的 Real Recall 再下降 1.94 点、Fake Recall 上升 1.18 点；相对原检测模型，Real Recall 下降 6.99 点、Fake Recall 上升 2.24 点、预测 Fake 比例增加 4.62 点。这个方向与 DataA 早期/最终 checkpoint 审计一致：相机辅助主要推动更强的 Fake 倾向，没有改善真假分离。逐来源胜率也远低于预设的 60%，不是少数总体指标波动掩盖了广泛提升。

结论标记：`未通过`。平衡二元 VQA 已经证明模型能学到视觉依赖的相机 primitive，但把独立 camera VQA 与 detection replay 直接交错进行 token-level SFT，并不会自动让相机能力参与 Real/Fake 决策；正确标签与翻转标签在检测端近似不可区分，正确标签没有形成语义特异的正迁移。这否定当前独立任务交错 SFT 配方，不证明 camera motion 对 AIGC 检测原则上无用，也不排除显式联合输出或以最终 Real/Fake 为主奖励的任务耦合方案。

ViF-Bench 已在项目开发中反复查看，因此本结果是开发诊断，不是全新论文最终测试。训练阶段接收二元相机问答监督，检测推理不接收相机文本；这是内部能力迁移测试，不能描述成推理时使用 camera context 的条件化结果。

立即下一步：停止为当前“独立 camera VQA + detection replay 交错 SFT”追加 epoch、比例、学习率或短程 camera-only RL。下一轮最小实验必须在同一次输出或奖励中同时约束最终 Real/Fake 与可验证相机中间变量，并继续保留仅检测奖励和错误相机语义两个控制；主要成败指标仍是无相机文本的 ViF-Bench Real/Fake 检测，方法冻结后再使用独立 GenBuster-Bench/MintVid。

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

## 18. 正确相机二元前置强化学习与分阶段检测恢复

### 这个实验测什么

从已经证明依赖当前画面的正确相机联合 SFT 模型出发，检验短程、可验证奖励的 Camera-PPRL 是否能把相机运动感知迁移到无相机文本的 AIGC 视频检测；保存直接结果后，再检验低强度 DataB detection replay 能否恢复检测能力而不抹掉相机能力。

### 日期、状态、模型谱系

- 日期：2026-07-15。
- 状态：`开跑前审查完成，待执行`。
- 原始检测模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- Camera-PPRL warm start：上述检测模型与 `/tmp/1res/camera_joint_sft_gate/train/correct_camera` 的正确相机联合 SFT adapter 合并后的模型。
- 检测恢复 warm start：正确相机联合 SFT 与 Camera-PPRL 合并后的模型。恢复分支不覆盖直接 PPRL 模型和结果。
- 完整执行说明：`docs/camera_pprl_overnight_20260715.md`。

### 训练与评测数据

- Camera-PPRL：`/tmp/1res/camera_joint_sft_gate/data/camera_train_correct.json`，从 DataA train cases 的 5580 条二元相机记录中按 primitive 轮转选择 1024 条，始终保留完整 Yes/No 配对。
- Camera held-out：`camera_dev_matched_frames.jsonl`、`camera_dev_opposite_frames.jsonl`、`camera_dev_no_frames.jsonl`，来自 case-level 隔离的 DataA test cases。
- 检测恢复：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`；该数据已被起始 checkpoint 见过，只是 replay，不是 held-out 证据。
- 检测开发评测：固定完整 ViF-Bench，沿用原 detection system/user prompt；所有检测推理均不提供 camera caption、camera label 或外部相机预测。

### 单一改变因素与控制

- 直接迁移比较：正确相机联合 SFT warm start 对比在同一起点上追加 Camera-PPRL；唯一新增因素是 1024 条平衡相机问题上的可验证 GRPO。
- 恢复比较：直接 Camera-PPRL 对比其后追加 0.5 epoch DataB detection replay；唯一新增因素是检测恢复训练。
- 恢复分支拥有额外计算，因此若只有恢复后模型通过，只能标记为`恢复候选，待控制`。随后必须补“同一联合 SFT 起点直接做等量 detection recovery、但不做 PPRL”的控制分支。
- 正确/翻转相机 PPRL 消融不在本轮优先队列；只有直接或恢复分支产生候选后再补，避免在主效应尚不存在时消耗整夜算力。

### 固定设置与验收标准

- Camera-PPRL：LoRA rank 32、alpha 64、学习率 `1e-6`、1 epoch、每题 8 次 rollout、temperature 1.0、top-p 1.0、beta 0.04；正确答案奖励 0.9、严格短格式奖励 0.1；冻结视觉塔和多模态 projector。
- 检测恢复：LoRA rank 16、alpha 32、学习率 `5e-6`、0.5 epoch；完整 DataB detection replay。
- 计算：16 张 96G GPU；colocate vLLM tensor parallel 4，将 16 个进程分为 4 个 rollout 组。
- 直接 PPRL 通过条件：ViF coverage/格式均至少 99%；Balanced ACC 或 Fake F1 至少提高 1 点，另一项下降不超过 0.5 点；Real/Fake Recall 均至少 45%；相机 Macro AP 下降不超过 2 点且 matched 相对 opposite 的 Balanced ACC 至少高 10 点。
- 直接和恢复分支都必须同时超过第一台服务器的仅检测回放 ViF 参照，且生成器级 Balanced ACC 胜率至少 60%；只超过正确相机 SFT 起点不算候选。
- 恢复分支使用同一 ViF 标准，相机 Macro AP 允许最多下降 5 点，且 matched 相对 opposite 的 Balanced ACC 仍须至少高 10 点；即使通过也不能在缺少等计算控制时归因于相机 PPRL。

2026-07-15 开跑前审查更正：原定义只要求 PPRL 超过正确相机联合 SFT 起点，可能把“从较差起点小幅回升但仍低于仅检测回放”误判为候选。现已把 NAS `/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/vif_four_model_compare/branches/detection-only/eval/camera_adapter_vifbench_eval.json` 接为必要强参照，并增加生成器级胜率检查。若首台服务器结果尚未到位，训练照常完成，但汇总只能标记为`等待检测对照`。同时把 `dynamic_sample`/`max_resample_times=3` 设为预检硬要求，并将 rollout temperature 从 readiness 使用的 0.7 调到多模态 GRPO 常用的 1.0，因为既有采样只有 12.56% 的问题组具有奖励方差；32 条 smoke 必须证明平均零方差组比例不高于 80%，否则不启动正式训练。smoke 还对实验性的 vLLM LoRA-only 同步增加一次自动降级重试。最终审查继续要求跨服务器 ViF 对照具有相同预期样本数和相同生成器集合，并要求 detection recovery 后仍保持 matched-vs-opposite 视觉依赖；预检还提前核验 DataB replay 全部图片和 ViF 索引/帧目录，避免在长时训练后才发现容器数据不完整。这些修正不改变训练问题、数据、奖励、学习率或训练轮数，只修复探索不足、错误判定和无人值守兼容性风险。

### 泄漏、分布差异与存储

ViF-Bench 已在本项目中反复查看，只能作为开发 benchmark，不能替代方法冻结后的 GenBuster-Bench 与 MintVid。Camera-PPRL 使用 DataA 局部编辑视频的相机标签，而 ViF 是完整生成视频，存在局部编辑到全生成的分布差异；本轮正是检验这种阶段式能力能否跨任务、跨分布迁移，不预设一定成功。

一次性合并模型、rollout 和逐样本预测放 `/tmp/1res/camera_pprl/correct_camera_1024`；固定训练 JSON、审计、正式汇总和紧凑日志复制到 NAS `res/camera_pprl/correct_camera_1024`；三个 compact adapter 自动上传到 `oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_pprl/correct_camera_1024/`。全部成功后才执行 `/input/training/keep.sh`，脚本不以 GPU 利用率阈值判定方法失败。

立即下一步：服务器先运行无模型加载的 `STAGE=preflight`，通过后启动 `STAGE=all`。结果优先读取 `camera_pprl_final_summary.json`；若直接 PPRL 通过，先补正确/翻转 PPRL 消融；若仅恢复分支通过，先补等计算 recovery-only 控制；二者都未通过则停止扩大当前二元 Camera-PPRL。

2026-07-15 任务定义更正：本节的相机-only Camera-PPRL 在正式执行前撤回主实验地位，代码保留为优化器/相机能力辅助消融。更正原因是其 rollout 只回答相机 Yes/No，奖励中没有 Real/Fake，因而即使相机奖励上升，也不能证明检测策略使用了相机信息；随后追加 detection recovery 又引入额外计算，不能修复这一归因缺口。主线改为下面的“检测主导的相机中间变量联合 SFT/GRPO 三对照门”。这是一项开跑前设计更正，不是对已有模型结果的改写。

## 19. 检测主导的相机中间变量联合 SFT/GRPO 三对照门

### 这个实验测什么

在同一次自回归生成中，模型先从有序帧输出短相机运动标签，再输出最终 `Real/Fake`。检测正确奖励作用于整条 rollout，测试正确的逐样本相机监督能否让最终真假检测优于两个等算力对照，而不是再次用相机 VQA 指标代替检测结果。

### 日期、状态与模型谱系

- 日期：2026-07-15。
- 状态：`代码与本地真实数据 dry-run 通过，待服务器执行`。
- 原始检测模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 已验证相机能力：原始检测模型挂载 `/tmp/1res/camera_joint_sft_gate/train/correct_camera` 后的合并模型；它只作为三个 GRPO 分支共同的能力起点。
- 公共联合输出 warm start：上述相机能力模型在 1024 条联合记录上做 1 epoch LoRA SFT，目标严格为 `<camera_motion>[...]</camera_motion>` 后接 `<answer>Real/Fake</answer>`。三个 GRPO 分支都从完全相同的合并 warm start 出发。
- 完整执行说明：`docs/camera_detection_joint_grpo_execution_20260715.md`。

### 训练和评测数据

- DataA detection：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`。
- DataA camera：`/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`。
- DataB detection replay：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json` 经既有平衡 replay 构建器抽取的记录；起点 checkpoint 已见过 DataB，因此它只用于 replay，不是 held-out 证据。
- DataB camera：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`，由 CameraBench 模型产生的伪标签；该噪声是已知限制。
- 联合训练固定 1024 条：DataA 512 条，即 256 个完整 real/fake case pair；DataB 512 条；总计 Real/Fake 各 512 条。DataA real/fake 同源 pair 使用同一相机标签，避免相机标签直接编码真假。
- DataA 开发门：既有 70:30 case-level split 的 324 个未训练 cases、648 条 Real/Fake 记录；继承 checkpoint 未见这批 DataA，但项目已反复用它选方案，因此称开发留出集，不称最终未见测试。
- 通用开发门：完整 ViF-Bench。它不输入相机 label/caption，但已被项目反复查看，只用于开发复核。
- 方法冻结后的最终通用测试：优先使用 GenBuster-200K 的独立 `benchmark` 集，须先做精确视频/帧哈希零重叠审计；MintVid 视数据可用性补充。当前实验不提前查看这些最终结果。

### 单一改变因素与对照

- 正确相机联合奖励：同一 rollout 奖励为 `0.65 × Real/Fake 正确 + 0.30 × 相机标签集合 F1 + 0.05 × 严格联合格式`。
- 打乱相机联合奖励对照：输入帧、prompt、Real/Fake 标签、样本顺序、训练步数和奖励权重完全相同，只把逐样本相机真值在 `source × Real/Fake` 内做固定置换。置换保持标签边际分布，DataA real/fake pair 仍共享同一置换后标签。
- 仅检测奖励对照：输入、联合输出格式、训练记录和步数相同，奖励改为 `0.95 × Real/Fake 正确 + 0.05 × 严格联合格式`，相机块存在但不计分。
- 三个分支共同使用正确相机联合 SFT warm start，因此比较的是“检测奖励下继续提供正确逐样本相机 credit”是否有增量，不把共同起点差异混入对照。
- `0.65 > 0.30 + 0.05`，所以任何检测错误且相机/格式满分的回答都低于检测正确的回答；Real/Fake 在奖励排序中具有硬主导地位。

### 固定训练和推理设置

- 公共 warm SFT：LoRA rank 16、alpha 32、学习率 `2e-6`、1 epoch，冻结视觉塔和多模态对齐层。
- 三个 GRPO 分支：LoRA rank 16、alpha 32、学习率 `8e-7`、1 epoch、每题 8 个 rollout、temperature 1.0、top-p 1.0、beta 0.04、最大 3 次动态重采样；冻结视觉塔和对齐层。
- 计算：16 张 96G GPU，vLLM colocate、tensor parallel 4。先跑 64 条 SFT smoke，再跑 64 条正确相机 GRPO smoke；smoke 平均零奖励方差组比例不得高于 80%。
- 训练和推理都不向 user/system prompt 提供 gold camera caption 或 label。相机标签是模型自己从帧生成的中间 token，并位于 `<answer>` 之前。
- 第一轮不训练自由文本 CoT，也不奖励解释质量；避免无法自动验证的解释奖励引入 reward hacking。检测主效应通过后，再单独恢复解释输出并继续用 Real/Fake 作为主指标。

### 验收标准

- DataA 诊断：四个模型覆盖率至少 99%、格式有效率至少 95%；正确相机分支相对仅检测和打乱相机两个对照，均须在 Balanced ACC 或 pair accuracy 至少提高 2 点，且另一主指标与 Fake F1 下降不超过 1 点；相对共同 warm start只要求三项下降均不超过 1 点。该结果记录局部编辑迁移，但不阻断 ViF-Bench。
- ViF-Bench：覆盖率和格式有效率至少 99%；正确相机分支相对两个对照均须在宏平均 Balanced ACC 或 Fake F1 至少提高 1 点，另一项下降不超过 0.5 点；相对共同 warm start 不得下降超过 0.5 点。
- 通用方法主门以 ViF-Bench 为准：只有正确相机分支在 ViF 同时超过仅检测和打乱相机对照，才允许把通用检测增量归因于逐样本相机监督。只提高相机 F1、只超过共同起点、或只在 DataA 提升都不算方法成功；DataA 未通过也不提前取消 ViF 复核。
- 相机标签 F1、格式率和 reward variance 只用于确认训练机制正常，不替代 Real/Fake 验收。

### 泄漏、分布差异和存储

- DataA fake 是局部编辑，DataB 与 ViF fake 主要是全生成，两者分布差异被明确保留：DataA 测局部机制，ViF 测全生成迁移，最终 GenBuster benchmark/MintVid 测冻结方法的通用性。
- DataB camera 是伪标签，正确分支若未超过打乱标签对照，不能归因于相机噪声以外的机制；先停止而不是增加 epoch。
- 一次性 merged models、rollout 和逐样本预测放 `/tmp/1res/camera_detection_joint_grpo/v1`；属于可重建验证产物，不放 NAS、不上传 OSS。
- 数据摘要、评测 JSON/CSV 和紧凑训练日志复制到 NAS `res/camera_detection_joint_grpo/v1`。
- 公共 warm adapter 和三个 compact GRPO adapter 是昂贵且可复用的大文件，生成后上传到 `oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_detection_joint_grpo/v1/`；不上传一次性合并模型。

### 本地实现验证与立即下一步

本地现有 DataA/DataB 镜像 dry-run 成功构建 1024 条记录：DataA/DataB 各 512、Real/Fake 各 512、DataA train/eval case 交集为 0；打乱相机标签改变 99.21875% 记录，同时保持 `source × Real/Fake` 下相机标签集合边际完全一致。该结果只验证数据和对照实现，不是模型效果。

立即下一步：服务器依次运行 `STAGE=preflight`、`STAGE=build`、`STAGE=smoke_sft`、`STAGE=train_warm_sft` 和 `BRANCH=correct_camera STAGE=smoke_grpo`。工程 smoke 通过后训练三个分支，记录 DataA Real/Fake 局部编辑诊断，并无条件完成 ViF-Bench 三对照主门；只有 ViF 未满足检测验收时才停止扩大，不用相机 VQA 指标挽救结论。

2026-07-15 执行优先级更正：DataA fake 是单一 VACE 局部编辑，和论文希望提升的通用全生成检测分布不同，因此 DataA 从硬停机门改为必须记录的机制诊断。完整流程即使 DataA 未通过也继续 ViF-Bench；方法是否进入 GenBuster `benchmark` 最终测试由 ViF 三对照 Real/Fake 结果决定。DataA 的 `failed` 和 ViF 的 `no_camera_gain` 都是正常实验结论，不作为 shell 工程错误；流程仍归档并执行 keep-alive，只有环境、数据、训练、推理或 smoke 报错才非零退出。更正只改变阶段间停止规则，不修改训练数据、模型、奖励或两套评测本身。

## 20. 三分类相机运动硬路由检测专家验证

### 这个实验测什么

在检测模型不接收 camera caption、camera label 或任何外部相机文本的前提下，先从检测模型实际读取的同一组有序帧预测 `no-motion / minor-motion / complex-motion`，再选择对应的 detection LoRA，检验相机运动条件化的专家分工是否能改善最终 `Real/Fake`，而不是继续用相机 VQA 指标代替检测指标。

### 设计动机与可证伪假设

前面的平衡二元相机问答已经证明 Qwen3-VL-8B 能从画面学习相机 primitive，但“正确相机 VQA 与 detection replay 交错 SFT”在 DataA 和 ViF-Bench 都没有超过仅检测与翻转相机控制。正确和翻转相机监督在检测端近似不可区分，说明把两个任务放进同一 SFT 数据流并不会自动建立相机能力到 `Real/Fake` 决策的连接。本实验因此不再要求单一 LoRA 自发完成隐式任务迁移，而是把相机条件显式用于选择检测决策边界。

形式上，路由器先由相同有序帧得到 `r = g(x)`，检测输出为 `y = f_r(x)`。这轮实验包含三个相互补充的可证伪比较：

- `预测路由 > 共享模型`：检验按相机运动进行条件化 specialization 是否比同数据的单一决策边界更有效。
- `预测路由 > 循环错误路由`：检验收益是否依赖当前视频的正确路由语义，而不只是三个 LoRA 带来的额外参数量。循环错误路由复用相同专家预测，只改变专家选择。
- `预测路由 > 同协议原始检测模型`：防止共享续训本身退化时，把“少退化”误写成方法提升。

硬路由只是低成本机制门，不预设为最终论文架构。若该门通过，才有依据进一步实现单模型内部 soft gate 或 camera-conditioned residual adapter；若失败，则停止这个路由家族，不继续用相同数据配方追加 RL。

### 日期、状态和模型谱系

- 日期：2026-07-15。
- 状态：`held-out DataA 三分类路由门未通过；停止当前三专家训练`。总体路由指标达标，但 `minor-motion` 中间桶几乎不可识别，不能把总体准确率写成方法通过。
- 检测起点：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 路由模型：从同一检测起点训练独立三桶 router adapter，位置为 `/tmp/1res/camera_hard_route_gate/v1/train/router`。旧二元相机 adapter 的标签构建忽略了独立 `static` 标签，不能直接把其 `no-motion` 分数当成 `static/no-motion` 合并类；新 router 显式修复该标签契约并仍只使用有序抽帧，不使用原视频输入。
- 检测分支：共享 detection LoRA、无相机运动专家、轻微运动专家、复杂运动专家，四者都从同一检测起点独立训练。
- 代码版本：GitHub 分支 `codex/caspr-gate1`，提交 `1bf31cb`（`Add camera hard-routing validation gate`）。
- 完整执行说明：`docs/camera_hard_route_gate_execution_20260715.md`。

### 训练与路由数据

- DataA detection：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`。
- DataA camera：`/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`。
- DataB detection replay：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`；起点 checkpoint 已见过该数据，所以它只用于 replay，不是 held-out 证据。
- DataB camera 伪标签：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`。
- DataA 继续使用按 source family 与 coarse route bucket 分层的 70:30 case split；同一 case 的 real/fake 永不跨 split。只有 train cases 进入专家训练，test cases 只用于路由校准和局部检测诊断。
- `static`、`no_motion` 与 `no-camera-motion` 统一映射为 `no-motion`；若多标签同时含 coarse bucket，优先级固定为 `complex-motion > minor-motion > no-motion`；无可用 bucket 的训练 case 排除并审计，推理时可回退共享模型。
- DataB 只取 detection frame path 能与 camera annotation path 对齐的记录，并在每个 route bucket 内分别平衡 Real/Fake。

路由器训练只使用 DataA train cases 的 `real` 有序 16 帧。这样既不让局部生成伪影成为相机类别捷径，也不把同一 case 的 real/fake 重复算作两个独立相机训练样本。每个 case 分别构造“是否无运动”“是否轻微运动”“是否复杂运动”三个 Yes/No primitive；每道问题内部 Yes/No 等量，三道问题的记录数也严格相同。held-out DataA 则同时评测 real 与 fake 两套帧，用来检查局部编辑前后路由是否保持一致。

LlamaFactory 中注册的数据名为：

- `camera_hard_route_router`：三分类 router 的平衡二元训练记录。
- `camera_hard_route_shared`：三个检测专家数据的精确并集。
- `camera_hard_route_no_motion`、`camera_hard_route_minor_motion`、`camera_hard_route_complex_motion`：三个互斥专家数据。

正式服务器构建要求 DataA 为 1080 cases。低于预期、train/test case 重叠、共享数据不是三个专家的精确并集、任一专家缺少 Real/Fake、检测 prompt 泄漏 camera 文本或图片路径缺失都会直接报错，不会静默跳过后继续训练。

### 单一改变因素和对照

- 共享对照读取三个专家训练数据的精确不重叠并集；三个专家各只读取自己的 route bucket。每条训练记录只进入一个专家，因此三个专家合计的数据量等于共享对照的数据量。
- 三个专家和共享对照都只训练原 detection 目标，prompt 中没有 camera 文本。相机信息只决定推理时选哪个 LoRA，不进入 detection prompt。
- ViF-Bench 先对三个专家和共享模型各做一次完整预测，再离线构造三种条件：预测相机路由、循环错误路由、共享模型。正确与错误路由复用完全相同的逐模型预测，唯一变化是专家选择。
- ViF-Bench route manifest 从 `test_index.rank*.json` 指向的同一 16 帧生成，并镜像既有 `ViFBench.py` 的 `timestamps.txt` 加 `1.png ... N.png` 顺序；不使用只覆盖部分样本的原视频，避免额外时序信息与原视频可用性造成不公平和选择偏差。

最终实际比较四个条件：

| 条件 | 训练/推理含义 | 用途 |
|---|---|---|
| 同协议原始检测模型 | 原始 checkpoint 使用本轮同一 ViF index、prompt 与解码设置重跑 | 绝对基线，消除旧提示词协议差异 |
| 共享检测续训模型 | 单一 LoRA 读取三个专家数据的精确并集 | 等数据量、等优化样本控制 |
| 预测相机路由 | router 从当前 16 帧预测桶并选择对应专家；低置信度回退共享模型 | 候选方法 |
| 循环错误路由 | `no -> minor -> complex -> no`，低置信度样本仍按相同规则回退共享模型 | 相机路由语义因果控制 |

三个专家合计看到的训练记录集合与共享模型完全相同，但硬路由拥有三个独立 LoRA，参数存储高于共享模型。因此仅仅超过共享模型仍可能来自容量或 specialization；必须同时超过错误路由，才能说明“为当前视频选择哪一个专家”具有信息价值。

### 路由输入、打分与 manifest 契约

- DataA 路由校准和 ViF 路由都使用与检测器相同的有序抽帧，不额外读取完整视频。ViF 默认 index 为 `eval/v4train-main/eval/test_index_splits/splits_16`，若部署目录采用上一层结构则自动兼容 `eval/v4train-main/test_index_splits/splits_16`。
- ViF 每个 index frame directory 必须具有 `timestamps.txt` 和严格的 `1.png ... 16.png`。原始 mp4 缺失不影响实验；若检测所需的帧或时间戳缺失，构建阶段直接给出具体路径并失败，绝不只在“有原视频的子集”上报告结果。
- 对每个视频分别计算三个问题首个答案 token 的 `Yes logit - No logit`。三路分数经过相对 softmax 得到 top-1 与 top-2 margin；这些值只用于相对路由与回退，不宣称是校准后的真实类别概率。
- route manifest 保存 `video_id`、三路原始分数、相对分数、`predicted_bucket`、最终 `route_bucket`、`cyclic_route_bucket`、top probability、margin、是否回退共享模型及回退原因。
- 低置信度阈值只允许在 held-out DataA 上根据 coverage、三分类性能和配对一致性确定一次，之后原样冻结用于 ViF-Bench。ViF 没有 gold camera 标签，禁止再根据 ViF 检测结果反向调路由阈值。

### 完整实验流程

1. **环境与数据审计**：`STAGE=preflight` 只检查模型、四个源数据、LlamaFactory、ViF index、依赖和 16 张 GPU；`STAGE=build` 构造 DataA split、router、共享/专家检测数据并注册 LlamaFactory；`STAGE=smoke` 验证两步训练链路。
2. **路由能力门**：`STAGE=train_router` 训练三桶 router；`STAGE=calibrate_dataa_route` 在 held-out DataA real/fake 帧上计算三路 logits、路由混淆矩阵、每桶 recall、macro recall 与 pair consistency。只有该门通过才训练检测专家。
3. **检测 specialization**：`STAGE=train_all` 依次训练共享 LoRA 和三个专家；也可用 `train_shared / train_no_motion / train_minor_motion / train_complex_motion` 在独立服务器运行。所有 detection 训练目标与原检测任务一致，不包含 camera caption、label 或 route token。
4. **冻结路由并生成 ViF manifest**：`build_vif_route_inputs -> score_vif_route -> aggregate_vif_route`。使用 DataA 冻结的阈值，对 ViF index 中每一个检测样本生成路由；输出必须和 index 的 `video_id` 集合精确相等。
5. **ViF 四模型推理与离线合成**：共享分支同时重跑原始 base 和共享 LoRA；三个专家各自推理一次。`compose_vif` 不再加载模型，只从四套完整预测离线组成共享、预测路由和循环错误路由，并输出严格门汇总与既有官方评测日志。

### 固定设置和验收标准

- 三桶 router LoRA：rank 16、alpha 32、dropout 0.05、学习率 `1e-4`、3 epochs；DataA train real 帧上的三个 coarse Yes/No 问题，每题内部 Yes/No 等量且三题记录数严格相同。
- detection LoRA：rank 16、alpha 32、dropout 0.05、学习率 `5e-5`、2 epochs；冻结视觉塔和多模态 projector；16 张 96G GPU。
- 路由分数：对三个互斥 coarse question 计算首 token `Yes-No` logit，再做三路相对 softmax。保存 top-1、top-2 margin、低置信度回退与循环错误路由；该相对值不宣称为校准概率。
- DataA 路由前置门：coverage 100%，三分类 accuracy 至少 60%，macro recall 至少 55%，每桶 recall 至少 40%，同一 real/fake pair 的预测路由一致率至少 80%。未通过时不训练四个 detection LoRA。
- ViF-Bench 检测门：原始 base、共享模型、预测路由、错误路由的 coverage/格式均至少 99%；预测路由相对同协议原始 base 和共享模型分别在 Balanced ACC 或 Fake F1 至少提高 0.5 点且另一项下降不超过 0.5 点；预测路由相对循环错误路由至少提高 1.0 点且另一项下降不超过 0.5 点。原始 base 在同一次执行中重跑，不能混用旧 83.96 与当前严格协议 79.18。
- ViF 路由摘要同时报告 Real/Fake 的 route distribution total variation 和配对 route 一致率。若路由本身高度相关于真假，必须作为 benchmark shortcut 风险报告，不能据此声称模型完成了 camera-aware artifact reasoning。

检测主指标始终是宏平均 Balanced ACC 与 Fake F1；同时保留 Fake Recall、Real Recall、格式有效率、覆盖率和每个生成来源上的 Balanced ACC 胜率。相机三分类 accuracy、macro recall 和 pair consistency 只负责确认路由器可用，不能替代 `Real/Fake` 结果。

### 已完成的本地实现审计

本地旧镜像 DataA 为 1076 cases，仅用于代码 dry-run，不替代服务器正式 1080-case 统计。构建器得到 702 条三桶 router 记录：三个问题各 234 条且各自 Yes/No 平衡，总体 Yes/No 各 351；共享 detection 记录 6414 条；无运动、轻微运动、复杂运动专家分别为 1272、1114、4028 条，各桶 Real/Fake 完全平衡；DataA train/test case 交集为 0；共享记录 ID 与三个专家的不重叠并集完全一致。DataB 中有 1027 条 frame path 未与 camera path 对齐并被排除；训练并未为覆盖率强行猜测相机标签。共享数据中存在 89 次源数据重复内容，代码分别记录内容哈希和源记录唯一 ID，避免把原数据重复误判为跨专家泄漏。

实现已通过项目回归测试 `120 passed, 1 skipped`，同时通过 Python 编译、两个 shell 入口语法检查与 Git diff 审计。跳过项不属于本实验。上述仍只是数据、协议和实现审计，不是模型结果，结论标记为`结论不足`。

### 服务器 DataA 路由校准结果（2026-07-15）

结果来源：`/tmp/1res/camera_hard_route_gate/v1/routes/dataa_route_summary.json`。这是按既定 70:30 case split 得到的 held-out DataA 路由开发门；它检验相机路由可用性和局部编辑前后稳定性，不检验 `Real/Fake` 检测提升。

| 指标 | 结果 | 预设门槛 | 判定 |
|---|---:|---:|---|
| score coverage | 100.00% | 100% | 通过 |
| 三分类 accuracy | 73.46% | 至少 60% | 通过 |
| macro recall | 58.64% | 至少 55% | 通过 |
| real/fake pair route consistency | 92.59% | 至少 80% | 通过 |
| accepted coverage | 99.38% | 记录项 | - |
| accepted accuracy | 73.60% | 记录项 | - |
| 每桶 recall | 最低 4.90% | 每桶至少 40% | **未通过** |

| Gold bucket | 支持数 | Recall | 主要预测去向 |
|---|---:|---:|---|
| `no-motion` | 114 | 84.21% | 96 正确，9 判为轻微，9 判为复杂 |
| `minor-motion` | 102 | 4.90% | 5 正确，46 判为静止，51 判为复杂 |
| `complex-motion` | 432 | 86.81% | 375 正确，41 判为静止，16 判为轻微 |

结论标记为`未通过`。模型可靠区分了两个端点，且同一 real/fake pair 的路由高度一致，说明局部编辑没有明显破坏相机判断；但中间档几乎对半落向两个端点。这更符合连续、有序运动强度被硬切成三类后的中间边界失效，而不是可供三个 detection 专家使用的稳定语义分区。因此不能继续运行 `train_all`，也不能用总体 73.46% 掩盖 `minor-motion` 塌缩。

将 gold 和预测都事后合并为 `no-motion` 对 `minor+complex motion` 时，仅由当前混淆矩阵可得二分类 accuracy 83.80%、macro recall 83.96%；合并不会降低现有 92.59% 的 pair consistency。这是看过同一开发集后的派生诊断，不是新的独立验证结果。它只支持下一步预先固定“静止/有运动”二路 Router、共享检测对照和错误路由对照后再做一次低成本验证，不能追认本次三分类门通过。

### 结果文件与判定顺序

第一阶段只需要读取：

- 数据构建审计：`/tmp/1res/camera_hard_route_gate/v1/data/camera_hard_route_data_summary.json`。
- DataA 路由门：`/tmp/1res/camera_hard_route_gate/v1/routes/dataa_route_summary.json`。

进入完整 ViF 后读取：

- ViF 输入覆盖审计：`/tmp/1res/camera_hard_route_gate/v1/routes/vifbench_route_input_summary.json`。
- ViF 三分类 route manifest：`/tmp/1res/camera_hard_route_gate/v1/routes/vifbench_route_manifest.jsonl`。
- ViF 路由分布摘要：`/tmp/1res/camera_hard_route_gate/v1/routes/vifbench_route_summary.json`。
- 最终四条件门：`/tmp/1res/camera_hard_route_gate/v1/vifbench/composed/camera_hard_route_gate.json`。

判定顺序固定为：先检查工程 coverage/格式，再检查 DataA 路由门；ViF 阶段先检查路由 manifest 是否覆盖完整 index 和是否存在明显真假 route shortcut，最后才读取 `Real/Fake` 四条件差值。不能因为某个生成器单独提升而忽略宏平均未过门，也不能用相机分类指标挽救检测失败。

### 泄漏、分布差异和结论边界

- 起点 checkpoint 已经见过 DataB，因此 DataB 只能作为检测 replay；任何 DataB 训练内评测都不能证明泛化。
- DataA fake 是 VACE 局部编辑，DataB 与 ViF fake 主要是全生成。DataA 路由门验证的是相机分类与局部编辑不变性，不能证明通用 AIGC 检测提升；通用开发结论必须来自 ViF 的 `Real/Fake` 对照。
- DataB camera 标签来自 CameraBench 模型而非人工 gold，可能包含噪声和来源偏差。循环错误路由与每生成器胜率用于检查正确路由语义是否真的产生增量，但不能完全消除伪标签偏差。
- ViF-Bench 已被反复用于开发，只能作为方法筛选集。方法冻结后还需要在零重叠审计通过的 GenBuster-200K `benchmark` 集和可用的 MintVid 上评测，才能作为论文外部泛化证据。
- 硬路由增加 router 推理和三个 LoRA 的存储，不是最终效率最优结构。通过只建立“相机条件化决策边界存在有用信号”；不自动证明相机运动导致伪影，也不证明自由文本解释质量提高。

### 存储与立即下一步

- 一次性 score、逐样本预测、合并模型和 adapter 放 `/tmp/1res/camera_hard_route_gate/v1`；第一轮校准不通过时不上传 OSS。
- split、数据摘要、route manifest 和评测 JSON 复制到 NAS `res/camera_hard_route_gate/v1`。
- DataA 路由门通过后先单独上传将被 ViF manifest 复用的 router；只有 ViF 硬路由门通过后，再把完整 `train/` 下的四个 detection adapter 上传到 `oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_hard_route_gate/v1/train/`。

立即下一步：不要执行 `STAGE=train_all`，当前三专家路线在预设门槛下已经失败。先复用现有 logits 做无需训练的“静止/有运动”二路 Router 契约审计，并预先固定二路验收标准；只有二路门通过，才构造共享 detection 对照、静止专家和有运动专家。不要通过追加三分类 epoch、改 DataA 门槛或启动同方向 RL 来追逐本次开发集结果。

完整阶段命令与代码文件清单以 `docs/camera_hard_route_gate_execution_20260715.md` 为准。当前代码入口为 `scripts/camera_hard_route_gate/run.sh`，数据构建器为 `tools/build_camera_hard_route_gate.py`，ViF 路由/预测合成为 `scripts/camera_hard_route_gate/route_manifest.py`，独立专家 ViF 推理复用并扩展 `scripts/camera_detection_retention/run_vifbench.sh`。

## 21. 静止/有运动二路硬路由复核

### 这个实验测什么

三分类路由在 held-out DataA 上可靠识别 `no-motion` 与 `complex-motion`，但 `minor-motion` recall 仅 4.90%，并近似对半落向两个端点。本复核不重新训练、不重新打分，而是预先固定 `no-motion -> no-motion`、`minor-motion/complex-motion -> motion`，检查现有视觉 Router 是否至少能提供稳定的“静止/有运动”条件变量。

### 日期、状态、输入和单一改变因素

- 日期：2026-07-15。
- 状态：`通过，允许进入二专家 Real/Fake 检测门`。
- Router 谱系：原 DataB detection checkpoint `checkpoint-2115` 上训练的三分类 Router adapter；本实验不更新任何参数。
- 输入：`/tmp/1res/camera_hard_route_gate/v1/routes/dataa_route_manifest.jsonl`，即已经完成的 held-out DataA 三分类逐视频结果。
- 输出：`/tmp/1res/camera_hard_route_gate/v1/routes/dataa_binary_route_manifest.jsonl` 与 `dataa_binary_route_summary.json`；脚本同时复制到 NAS 的 `res/camera_hard_route_gate/v1/routes/`。
- 单一改变：只把冻结的三分类 top-1 标签合并为两类。正确路由为 `no-motion` 对 `motion`，错误路由控制固定为两类互换；不重新选择阈值，不比较多种合并方式后挑最好结果。
- detection prompt、图片帧和模型推理均不参与本步骤，因此不存在 camera 文本进入检测 prompt 的问题，也不会消耗 GPU。

### 预设验收标准

| 指标 | 门槛 |
|---|---:|
| manifest coverage | 100% |
| 二路 accuracy | 至少 75% |
| 二路 Balanced ACC | 至少 75% |
| `no-motion` 与 `motion` 各自 recall | 均至少 70% |
| 同一 real/fake pair 路由一致率 | 至少 90% |

结果还必须报告每个 VACE source family 的指标和 real/fake 路由分布差异，但不根据来源子集反复改变映射或门槛。通过只说明二路条件变量可用，不说明 `Real/Fake` 检测有提升；真正的任务耦合仍须由后续等数据共享模型、正确二路专家和交换错误路由在 ViF-Bench 上的检测差值证明。

### 服务器结果（2026-07-15）

结果来源：`/tmp/1res/camera_hard_route_gate/v1/routes/dataa_binary_route_summary.json`。所有预设检查均通过。

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| coverage | 100.00% | 100% |
| accuracy | 83.80% | 至少 75% |
| Balanced ACC | 83.96% | 至少 75% |
| `no-motion` recall | 84.21% | 至少 70% |
| `motion` recall | 83.71% | 至少 70% |
| real/fake pair consistency | 95.37% | 至少 90% |

三个来源的 Balanced ACC 分别为：VACE-1.3B dataset 82.80%、VACE-1.3B textedit 86.55%、VACE-14B 86.06%；对应 pair consistency 为 95.33%、98.00%、93.33%，没有单一来源独占总体收益。real 与 fake 的预测路由分别为 91/233 和 92/232（静止/有运动），分布 total variation 仅 0.31%，未见 Router 直接利用局部真假差异的明显捷径。

该门标记为`通过`，但仍不是检测结果。静止类 precision 只有 52.46%，说明二路 Router 并非高精度相机 benchmark 模型；允许继续的依据是两类 recall、来源稳定性、配对一致性和真假分布控制共同过门。后续必须依靠正确路由相对共享模型与交换错误路由的 `Real/Fake` 增量判断任务是否真正耦合。

### 偏差与结论边界

该二路定义是在查看同一 DataA 三分类开发门后提出的，属于明确记录的事后方法修订，不是独立测试结果。DataA 继续只作开发与机制诊断；若本门通过，二路映射必须立即冻结，禁止再根据 ViF-Bench 检测标签调整。若本门失败，停止硬路由家族；若通过，只进入“共享模型 + 静止专家 + 有运动专家”的等数据检测门，不追认三分类实验成功。

立即下一步：二路映射已经冻结，不再根据 ViF 标签修改。进入下一节的等数据共享模型、静止专家和有运动专家训练；现有 Router adapter 必须保留并用于 ViF route manifest。

## 22. 二路相机硬路由检测专家门

### 这个实验测什么

冻结已经通过 DataA 开发门的视觉 Router，训练一个读取全部检测记录的共享 LoRA、一个只读 `no-motion` 记录的静止专家和一个读取 `minor-motion + complex-motion` 记录的有运动专家。ViF-Bench 推理时不提供 camera caption、camera label 或 route token；Router 与检测器读取同一组有序 16 帧，Router 只负责选择专家。

### 日期、状态和模型谱系

- 日期：2026-07-15。
- 状态：`未通过；当前静止/有运动硬路由主线停止`。
- 三个 detection 分支均从 `/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115` 独立开始，不从相机 VQA 或其他 detection LoRA 串行续训。
- Router 固定为 `/tmp/1res/camera_hard_route_gate/v1/train/router`，不再更新参数；二路映射固定为 `no-motion` 对 `motion = minor-motion + complex-motion`。
- 新实验工作目录：`/tmp/1res/camera_binary_route_gate/v1`；持久化小结果目录：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_binary_route_gate/v1`。

### 数据和等算力契约

- 输入检测数据来自已经审计的 `camera_hard_route_gate/v1/data/hard_route_*.json`，不重新划分 DataA，不改变 DataB 选择规则。
- `camera_binary_route_no_motion` 只含原 `no-motion` 检测记录。
- `camera_binary_route_motion` 是原 `minor-motion` 与 `complex-motion` 两个互斥专家记录的精确并集。
- `camera_binary_route_shared` 必须是上述两个专家的不重叠精确并集；两个专家合计看到的 record ID、样本数和 epoch 数与共享模型完全相同。
- 每个分支内部必须同时含 Real/Fake 且数量相等，检测 prompt 不得出现 camera 文本。实际服务器记录数以 `camera_binary_route_data_summary.json` 为准，不根据本地旧镜像猜测。

三个 LoRA 使用同一设置：rank 16、alpha 32、dropout 0.05、学习率 `5e-5`、2 epochs，冻结视觉塔与多模态 projector，16 张 96G GPU。共享与专家拥有不同参数量，因此“正确路由超过共享”仍可能受 specialization/容量影响；交换错误路由复用完全相同的两套专家预测，只交换专家选择，是判断相机路由语义是否真正有用的关键控制。

### ViF-Bench 条件和验收标准

四个最终条件为：同协议原始 detection checkpoint、等数据共享 LoRA、冻结 Router 选择的正确二路专家、两路互换的错误 Router。四者使用同一 ViF index、检测 prompt、帧和解码协议，camera 文本不进入检测模型。

预设门槛：四个条件 coverage 与格式有效率均至少 99%；正确路由相对原始模型和共享模型，Balanced ACC 或 Fake F1 至少提高 0.5 点且另一项下降不超过 0.5 点；正确路由相对交换错误路由至少提高 1.0 点且另一项下降不超过 0.5 点。还要报告每个生成器的 Balanced ACC 胜率和 ViF real/fake 路由分布差异。只有同时超过共享、原始和错误路由才标记为`通过`。
### ViF 二路 route manifest 结果（2026-07-16）

结果来源：`/tmp/1res/camera_binary_route_gate/v1/routes/vifbench_binary_route_summary.json`。该步骤只运行冻结 Router 并生成专家选择，不包含任何 detection 预测，因此状态记为`工程通过、方法结论不足`。

| 指标 | 结果 |
|---|---:|
| ViF 视频覆盖 | 3160/3160 |
| `no-motion` | 2439（77.18%） |
| `motion` | 721（22.82%） |
| Real 的 `motion` 比例 | 31.52% |
| Fake 的 `motion` 比例 | 22.34% |
| Real/Fake route distribution TV | 9.18% |
| 配对 real/fake 同路由率 | 78.95% |
| 可配对数量 | 2979 |

两路均有数百个样本，不属于单路完全塌缩，可以进入专家检测评测。但 ViF 上的 route 分布与 DataA 明显不同：DataA 的真假 TV 仅 0.31%、pair consistency 为 95.37%，ViF 则分别为 9.18% 和 78.95%。若只根据 route 做真假猜测，取 `motion -> Real`、`no-motion -> Fake`，由当前分布可得到约 54.59% 的理论 Balanced ACC，说明 route 带有弱真实性先验。该现象可能来自完整生成视频真实改变了相机运动，也可能来自 Router 跨域偏差；没有 gold camera 标签时不能区分。

因此该偏差不作为停止条件，但必须进入最终解释。专家训练数据在每个二路桶内 Real/Fake 等量，减少了显式标签先验；最终仍只有“正确路由同时超过共享模型和交换错误路由”才能支持 camera-conditioned specialization。若正确路由只超过共享却不超过交换路由，或收益可以由 route-only 54.59% 先验解释，则本方法不通过。
### ViF 四条件检测结果（2026-07-17）

结果来源：`/tmp/1res/camera_binary_route_gate/v1/vifbench/composed/camera_binary_route_gate.json`；精简结果为同目录的 `camera_binary_route_gate_compact.json`。四个条件 coverage 均为 100%，格式有效率均超过 99.90%，因此差异不是缺样本或输出格式造成。

| 条件 | Balanced ACC | Fake Recall | Fake F1 |
|---|---:|---:|---:|
| 原始 detection checkpoint | 79.18% | 89.33% | 80.47% |
| 等数据共享 LoRA | 76.30% | 88.42% | 77.98% |
| 正确二路 Camera Router | 74.50% | 91.04% | 77.78% |
| 交换错误 Router | 78.03% | 88.92% | 79.61% |

| 正确路由相对条件 | Balanced ACC 差值 | Fake Recall 差值 | Fake F1 差值 |
|---|---:|---:|---:|
| 原始模型 | -4.69 点 | +1.70 点 | -2.69 点 |
| 共享模型 | -1.80 点 | +2.62 点 | -0.19 点 |
| 交换错误路由 | -3.54 点 | +2.12 点 | -1.83 点 |

所有预注册检查均失败。正确路由只提高 Fake Recall，却同时显著降低 Balanced ACC 和 Fake F1，表现为更偏向预测 Fake，而不是提高真假区分能力。逐生成器 Balanced ACC 胜率在相对原始、共享和错误路由三个比较中都只有 1/19（5.26%），不是少数生成器拖累宏平均。

最关键的因果控制是交换错误路由：它复用完全相同的静止专家和有运动专家预测，只交换专家选择，却比正确路由高 3.54 点 Balanced ACC 和 1.83 点 Fake F1。因此当前相机语义分配没有形成有用的 detection specialization，反而把多数 ViF 视频送向泛化更差的决策边界。共享 LoRA 本身相对原始模型也下降 2.88 点 Balanced ACC 和 2.50 点 Fake F1，说明该续训数据与 LoRA 配方总体未保留原始 ViF 能力。

结论标记为`未通过`。不得把交换错误路由事后改名为候选方法，也不在 ViF 标签上重新选择静止/有运动映射、路由阈值或专家标签。当前结果不进入 GenBuster 最终测试，不追加同方向硬路由、PPRL 或 GRPO。允许的下一步仅是复用现有全量专家预测做无需 GPU 的 expert-crossover 诊断，区分“某个专家全局更强”“训练数据量/域不平衡”与“真正存在分桶交叉收益”；该诊断只用于总结失败原因，不追认方法成功。

### 分布差异、存储和结论边界

DataA 是局部 VACE 编辑，DataB/ViF 主要是完整生成；DataA 二路门只证明 Router 可用，不能保证 detection specialization 外推。ViF 已反复作为开发 benchmark，二路检测门通过后还必须冻结方法并在零重叠 GenBuster `benchmark` 上评测。训练 adapter、合并模型和逐样本预测放 `/tmp`；小型数据审计、route manifest 和评测摘要复制到 NAS。三个正式 adapter 训练完成且准备进入 ViF 时，应在容器退出前上传 OSS。

立即下一步：停止硬路由训练与外部 benchmark 扩展。先在现有 ViF 全量预测上做零 GPU 的 expert-crossover 诊断，再回到论文主目标重新选择能够直接约束 `Real/Fake` 的 camera 耦合方式。

## 23. 二路检测专家交叉离线诊断

### 这个实验测什么

复用已经生成的 ViF 静止专家与有运动专家全量预测，不重新加载模型、不训练、不重新推理。分别在全部视频、Router 判为 `no-motion` 的视频和 Router 判为 `motion` 的视频上，计算两个专家的 video-level Balanced ACC、Real/Fake recall、Fake F1、预测 Fake 比例、逐样本正确性差异和逐生成器胜负。

### 日期、状态和输入

- 日期：2026-07-17。
- 状态：`诊断完成；失败机制分类为 experts_are_semantically_reversed`。
- Router manifest：`/tmp/1res/camera_binary_route_gate/v1/routes/vifbench_binary_route_manifest.jsonl`。
- 静止专家预测：`/tmp/1res/camera_binary_route_gate/v1/vifbench/no_motion/inference/camera_adapter/splitresults`。
- 有运动专家预测：`/tmp/1res/camera_binary_route_gate/v1/vifbench/motion/inference/camera_adapter/splitresults`。
- 完整结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_binary_route_gate/v1/diagnostics/expert_crossover.json`。
- 精简结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_binary_route_gate/v1/diagnostics/expert_crossover_compact.json`。

### 判读规则

该诊断没有方法 pass/fail，只分类失败机制：

- `semantic_crossover`：静止专家在静止子集更强，运动专家在运动子集更强；说明 specialization 存在，优先怀疑 Router 跨域或组合协议。
- `motion_expert_dominates_or_no_motion_expert_is_weaker`：运动专家在两个子集都不弱；优先怀疑训练数据量、优化步数或静止专家退化。
- `no_motion_expert_dominates_or_motion_expert_is_weaker`：静止专家在两个子集都不弱；优先怀疑另一专家退化。
- `experts_are_semantically_reversed`：两个专家分别在相反子集更强；优先检查标签契约、域偏差以及 camera 分桶是否与真正检测难点反向。

脚本还报告同一视频上两个专家谁单独判断正确，并给出 McNemar exact p-value；逐生成器只用于确认结论是否由少数来源驱动。

### 服务器结果（2026-07-17）

ViF-Bench 共 3160 条，其中 Router 判为静止 2439 条、判为有运动 721 条；两套专家在三个范围内均有完整预测。

| 评测范围 | 静止专家 Balanced ACC | 有运动专家 Balanced ACC | 胜出专家与差值 |
|---|---:|---:|---|
| 全部 ViF | 74.87% | 76.56% | 有运动专家 +1.69 点 |
| Router 判为静止 | 72.35% | 76.58% | 有运动专家 +4.23 点 |
| Router 判为有运动 | 78.76% | 75.20% | 静止专家 +3.56 点 |

| 评测范围与专家 | Real Recall | Fake Recall | 预测 Fake 比例 |
|---|---:|---:|---:|
| 静止路由 + 静止专家 | 51.33% | 93.38% | 91.31% |
| 静止路由 + 有运动专家 | 62.83% | 90.33% | 87.86% |
| 有运动路由 + 静止专家 | 75.00% | 82.51% | 78.36% |
| 有运动路由 + 有运动专家 | 69.23% | 81.17% | 77.53% |

逐生成器结果同样发生交叉：在静止路由子集，有运动专家赢 18/19 个生成器；在有运动路由子集，静止专家赢 16/19 个生成器。该现象不是少数生成器或单一来源造成。全部视频上有运动专家的 Balanced ACC 更高，但静止专家在有运动子集反而明确更强，因此结果也不能简化为“静止专家整体训练失败”。

同一视频的原始正确率统计受 ViF 中 165 条 Real、2995 条 Fake 的严重类别不平衡影响。例如静止路由内，静止专家因更偏向预测 Fake 而有更高的原始 accuracy，但其 Real Recall 只有 51.33%，所以 Balanced ACC 反而低于有运动专家。主判读继续使用预注册的 Balanced ACC、分标签 recall 和逐生成器胜负，不用原始 accuracy 覆盖结论。

结论标记：`诊断完成`。预期的专门化应当是“静止专家在静止子集更强、有运动专家在有运动子集更强”，实际却完全相反。正确 Router 因而在两个子集上都系统性选择了较差专家，直接解释了上一节交换错误路由比正确路由高 3.54 点 Balanced ACC 的现象。优先原因不是简单的单专家全局强弱，而是 DataA/DataB 上按全局 camera motion 划分出的训练分布，与 ViF 上真正决定检测难度和决策边界的因素反向或跨域失配；ViF 缺少 camera gold，仍不能把其中多少归因于 Router 跨域误判、多少归因于专家训练分布完全拆开。

### 结论边界

该诊断直接使用已经查看过的 ViF 标签和预测，只能解释失败，不能用于选择新路由、交换专家名称或把错误控制改造成候选方法。无论输出哪种模式，前一节硬路由检测门都保持`未通过`。诊断完成后再决定下一条 camera 耦合路线，当前不启动新训练。

立即下一步：停止静止/有运动硬路由、专家改名、阈值搜索和同方向 RL。若只为补全失败机制，可在 held-out DataA 上让两位专家做同样的交叉评测，以区分“训练域内已经反转”和“只在 ViF 跨域后反转”；该诊断不影响主线决策。论文方法主线应回到共享检测模型，并让 camera 变量在同一条训练样本和同一个 `Real/Fake` 损失中直接改变证据聚合或检测决策，而不是继续把数据按全局相机类别拆成互不共享的专家。

## 24. DataB 显式 Camera labels+caption 配对检测 SFT

### 这个实验测什么

直接检验显式提供相机信息能否改善最终 AIGC 视频检测：两个模型都从同一个 Qwen3-VL-8B-Instruct 出发，只在同一批 DataB 上训练相同的 5 epoch。无 Camera 分支保留原 user prompt；Camera 分支仅在原 user prompt 末尾追加当前视频的 CameraBench `labels + caption`。system prompt、assistant 检测 CoT、`<answer>`、images 和所有其他字段均不改变。

本实验不测试路由、Camera VQA、DataA、RL 或缺失相机鲁棒性。主结果只比较两个训练后模型在 ViF-Bench 对应推理条件下的 Real/Fake 检测指标。

### 日期、状态和模型谱系

- 日期：2026-07-18。
- 状态：`未通过；显式相机文本条件路线停止`。两分支训练完成及 ViF-Bench 配对结果由用户于 2026-07-19 提供；trainer log 与最终权重哈希待补充。
- 两个分支共同起点：`/home/admin/Qwen3-VL-8B-Instruct`。
- 不使用旧 DataB detection checkpoint 作为训练起点。
- 无 Camera 分支输出：`/tmp/1res/datab_explicit_camera_sft/v1/train/no_camera`。
- Camera 分支输出：`/tmp/1res/datab_explicit_camera_sft/v1/train/with_camera`。

### 训练数据与唯一改变因素

- DataB detection：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`。
- DataB camera：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`。
- 原 detection 共 6766 行、6666 个唯一 frame directory；camera sidecar 共 5639 行且 path 唯一。
- 映射后保留 5739 行，对应 5639 个唯一 camera path；其余 1027 行从两个分支同时排除。
- 保留原始顺序及 100 条同视频不同检测解释记录。5739 条中 Fake 2866、Real 2873。
- 无 Camera 数据是这 5739 条源记录的原样深拷贝；Camera 数据只追加以下 block，不加入额外使用说明：

```text
<camera_motion>
<labels>label-1; label-2; ...</labels>
<caption>CameraBench caption</caption>
</camera_motion>
```

- 本地真实数据构建审计确认：两个分支均为 5739 行，逐条 system、assistant、images 与其他字段相等；无 Camera 分支不含 camera block，Camera 分支每条恰好包含一个完整 block。

### 训练设置

- 完整参数 SFT，只训练语言模型；冻结视觉塔和多模态投影层。
- 16 GPU，每卡 batch size 1，gradient accumulation 1。
- 5 epoch，学习率 `1e-5`，cosine scheduler，warmup ratio 0.1，bf16；每 500 step 保存，只保存模型权重，允许覆盖已有输出目录。
- `cutoff_len=49152`、`image_max_pixels=262144`、`packing=false`，其余设置复现原始 `dataB.yaml`。
- 两个分支必须使用相同基础模型文件、LlamaFactory 版本和训练配置；只允许 dataset name 与 output directory 不同。

### ViF-Bench 评测定义

- 无 Camera 模型继续使用原 ViF-Bench detection system/user prompt，不追加相机信息。
- Camera 模型在完全相同的 detection user prompt 末尾追加由冻结 CameraBench 模型为该 ViF 视频预测的 `labels + caption`，格式必须与训练一致。
- ViF predicted-camera sidecar：`/input/workflow_58770161/workspace/test/camb/camerabench_outputs/vifbench_cameramotion_labels_v2/datab_cameramotion_labels_v2.jsonl`；schema 与 DataB camera sidecar 相同。规范化后 3160/3160 唯一匹配，覆盖率 100%。
- 专用推理预检要求 sidecar 对当前 ViF 16 个 index shard 达到 100% 唯一匹配；不允许缺失样本静默使用 `unknown`。若不能达到 100%，本轮全量 ViF 主比较不得启动。
- 主比较只读取两模型的 ViF-Bench ACC、Real/Fake recall、Fake F1 及逐生成器结果；不把 CameraBench 自身相机指标作为方法成功证据。

### 验收、偏差与结论边界

- 工程有效性：两个训练集必须各 5739 行，推理覆盖与输出格式必须一致且至少 99%。
- 方法判定：Camera 分支应在相同 ViF 协议下稳定超过无 Camera 分支的核心 Real/Fake 指标；具体检测差值和是否进入独立 GenBuster `benchmark` 在结果到达后记录，不预填结果。
- DataB camera 和 ViF camera 都来自 CameraBench 模型而非人工 gold，因此本实验验证的是“外部预测的显式相机上下文是否有用”，不证明 Qwen3-VL 检测器自行学会相机估计。
- Camera caption 可能包含场景内容，因此若结果提升，只能先归因于完整 CameraBench labels+caption 条件；后续是否增加 labels-only 消融由主结果决定。本轮不提前扩展分支。
- ViF-Bench 已被多次用于开发，只能作为当前路线筛选集；若通过，方法冻结后仍需在零重叠的 GenBuster `benchmark` 上确认外部泛化。

### 2026-07-19 ViF-Bench 配对结果

结果来源：

- 服务器主摘要：`/tmp/1res/datab_explicit_camera_sft/v1/vifbench/eval/explicit_camera_vifbench_comparison.json`。
- NAS 持久化目录：`/input/workflow_58770161/workspace/test/cameramotion_det/res/datab_explicit_camera_sft/v1/vifbench/eval/`。
- 用户回传结果包：`E:/newgaibeishi/eval (3).zip`。

训推一致性审计：

- 两分支均从同一 Qwen3-VL-8B-Instruct 出发，在同一 5739 条 DataB 上使用完全相同的 5 epoch full-SFT 配置。
- 无 Camera 分支训练时不含 camera block，推理时使用原检测 user prompt，同样不提供 camera sidecar。
- Camera 分支训练时在原 user prompt 末尾追加匹配的 `labels + caption`；推理时 `infer_with_camera` 传入 Camera 专用 suffix 和逐视频 sidecar，运行时逐条替换 `{camera_labels}`、`{camera_caption}` 后再构造 user prompt。
- 推理脚本内部模式名为 `gold_camera`，但实际 sidecar 来自冻结 CameraBench 模型的预测，不是人工 gold。脚本启用了缺失即报错，最终 3160/3160 均匹配，没有使用 `unknown` 占位。
- 预检确认 system prompt 与 DataB 训练一致、无 Camera 后缀一致、Camera 后缀是训练格式的精确追加、两个占位符各出现一次且没有额外相机指令。训练和推理均用分号连接 labels，camera block 的位置和字段顺序一致。

因此，本轮两个分支在所检查的 system/user prompt、camera block、sidecar 覆盖和输出协议上训推一致；Camera 分支推理时确实把当前 ViF-Bench 视频的预测相机信息提供给了 user prompt。

| 模型/推理条件 | 覆盖率 | 格式有效率 | Balanced ACC | Real Recall | Fake Recall | Fake F1 |
|---|---:|---:|---:|---:|---:|---:|
| 无 Camera 模型 + 无 Camera prompt | 100.00% | 99.84% | 79.09% | 72.43% | 85.74% | 79.44% |
| Camera 模型 + 预测 labels/caption prompt | 100.00% | 99.91% | 76.42% | 71.33% | 81.50% | 75.75% |
| Camera 减无 Camera | - | +0.06 点 | -2.67 点 | -1.09 点 | -4.24 点 | -3.68 点 |

逐生成器结果中，Camera 分支仅在 19 个共同生成器中的 2 个取得 Balanced ACC 提升，Fake F1 同样仅胜出 2 个；下降不是单一生成器造成的偶然波动，其中 HunyuanVideo-I2V 下降最大。

结论标记：`未通过`。本实验直接否定了“把 CameraBench 预测 labels+caption 显式拼入 user prompt，并以相同格式完成检测 SFT 和推理”这一具体配方能提升 ViF-Bench 检测的假设。它不证明相机运动对 AIGC 检测天然无用，也不评价模型内部结构化融合；caption 还含场景语义，因此结果只适用于当前完整 labels+caption 文本条件路线。

### 代码、存储与立即下一步

- 数据构建：`tools/build_datab_explicit_camera_sft.py`。
- LlamaFactory 注册：`tools/install_datab_explicit_camera_sft.py`。
- 训练入口：`scripts/datab_explicit_camera_sft/run.sh`。
- ViF 配对推理与评测入口：`scripts/datab_explicit_camera_sft/run_vifbench.sh`。
- ViF sidecar 规范化与逐字提示词审计：`tools/prepare_vifbench_camera_context.py`。
- ViF 两分支汇总比较：`tools/compare_datab_explicit_camera_vifbench.py`。
- 配置模板：`configs/datab_explicit_camera_sft/train_template.yaml`。
- 执行说明：`docs/datab_explicit_camera_sft_execution_20260718.md`。
- 可重建训练 JSON 与 full-SFT 模型放 `/tmp/1res/datab_explicit_camera_sft/v1`；小型摘要、manifest 和最终 YAML 持久化到 NAS `res/datab_explicit_camera_sft/v1`。
- 两个 full-SFT 输出为正式可复用大文件，训练完成后应在容器退出前上传执行说明中给出的 OSS 目录。

立即下一步：停止继续训练同类显式 labels+caption 拼接模型，也不因这次下降直接跑 labels-only/caption-only 全量消融。后续若继续使用 camera，必须让相机变量在同一检测目标中改变证据聚合或决策，并先用低成本对照证明该耦合优于等算力检测对照，再启动完整训练。

## 25. DataB 到 ViF-Bench 的相机条件化几何残差最小验证

### 这个实验测什么

在不使用 DataA、检测 CoT、相机文本或复杂联合损失的条件下，验证“估计并消除全局相机运动后，剩余时序几何异常是否为通用 AIGC 视频检测提供独立增益”。这一步只决定是否值得把相机几何中间变量接入 Qwen3-VL，不直接声称形成最终方法。

### 日期、状态和模型谱系

- 日期：2026-07-19。
- 状态：`待执行；数据审计与代码已完成`。
- 模型谱系：冻结 DINOv2-Small 外观编码器与 TorchVision RAFT-Large 光流模型，后接同容量小型 MLP 二分类探针；不使用任何现有 Qwen3-VL detection checkpoint。
- RAFT 权重：`/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth`。
- DINOv2-Small：`/home/admin/dinov2-small`。

### 训练与评测数据

- DataB detection：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`。
- DataB camera sidecar：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`。
- ViF-Bench 16 帧索引：`/input/workflow_58770161/workspace/test/cameramotion_det/eval/v4train-main/test_index_splits/splits_16`。
- ViF-Bench predicted-camera sidecar：`/input/workflow_58770161/workspace/test/camb/camerabench_outputs/vifbench_cameramotion_labels_v2/datab_cameramotion_labels_v2.jsonl`。
- DataA、DataA mask/bbox、检测 CoT 和 camera caption 均不进入分类器训练或输入。
- DataB 按唯一帧目录去重，并按 `来源 x Real/Fake x 相机桶` 固定分层为 85% train 与 15% validation；ViF-Bench 只作外部开发评测，阈值不得在 ViF 上选择。

### 2026-07-19 DataB 数据审计

| 审计项 | 数量/结果 |
|---|---:|
| DataB detection 原始记录 | 6766 |
| 可匹配 camera sidecar 的 detection 记录 | 5739 |
| 去重后的唯一视频 | 5639 |
| 重复帧目录组 | 100 |
| train / validation | 4794 / 845 |
| complex-motion | 3126（55.44%） |
| minor-motion | 1255（22.26%） |
| static/no-motion | 1251（22.18%） |
| 冲突相机桶 | 7（0.12%） |
| 仅用相机桶多数类映射的 Balanced ACC | 57.08% |

约 77.7% 的唯一视频包含轻微或复杂运动，因此“DataB 缺少足够相机移动”不是本轮主要瓶颈。相机桶本身与真假标签存在非零相关性，且 GenBuster 子来源中的静止比例差异更明显；这构成潜在 shortcut，而不是可直接利用的科学信号。

### 唯一改变因素与控制条件

四个探针使用相同样本、固定划分、DINO 外观特征、网络容量、五个随机种子和一个加权二分类 BCE；唯一差异是追加的运动块：

| 分支 | 输入 | 作用 |
|---|---|---|
| 外观基线 | DINOv2 帧级 CLS 时序统计 | 测量不依赖运动的基线 |
| 原始运动控制 | 外观 + RAFT 原始光流统计 | 判断普通 motion feature 是否已经足够 |
| 正确几何残差 | 外观 + 当前帧对 homography/epipolar residual | 本轮候选方法 |
| 错配几何控制 | 外观 + 循环错位几何模型产生的同维残差 | 排除仅因特征维数或几何统计增加而提升 |

CameraBench labels/caption 只用于分层、样本权重和分桶报告，不作为分类器输入。样本权重按 `来源 x Real/Fake x 相机桶` 的逆频率计算并截断，防止模型主要依赖来源或相机桶边际。

### 训练设置与预注册验收

- 每个探针：`64,32` 两层 MLP、weighted BCE、AdamW、最多 120 epoch、DataB validation early stopping。
- 五个固定随机种子集成；分类阈值只由 DataB validation 的 Balanced ACC 选择。
- 主指标：ViF AUROC、按主要相机桶 macro Balanced ACC、逐来源 Balanced ACC 和配对分层 bootstrap。
- 正确几何残差相对原始运动和错配几何的 ViF AUROC 都至少提高 1.0 个百分点。
- 正确几何残差相对两个控制的运动桶 macro Balanced ACC 都至少提高 1.0 个百分点。
- 两个 AUROC 差值的配对 bootstrap 95% CI 下界都大于 0。
- 静止/无运动桶相对原始运动下降不超过 1.0 个百分点。
- 至少三个可计算来源存在时，逐来源 Balanced ACC 胜率至少 60%；特征覆盖率至少 99%。

### 已知偏差与结论边界

- DataB 的 camera sidecar 来自 CameraBench 模型预测而非人工 gold，但本轮只用其做粗粒度分层；几何残差直接由视频帧和 RAFT 对应关系计算。
- DataB 与 ViF-Bench 的来源分布不同，ViF 又已被项目多次用于开发，因此通过只说明存在值得继续的外部开发信号，最终仍须在零重叠 GenBuster `benchmark` 上验证。
- 本轮使用 16 帧和二维 homography/fundamental geometry，不等价于完整 3D camera pose/depth，也不证明 CoT 或局部编辑检测提升。
- DataA 的生成质量与 CoT 质量问题被有意隔离；若本门通过，DataA 只在后续作为局部定位诊断和定性材料，不先参与主分类器训练。

### 存储、代码与下一步

- 可丢弃验证特征：`/tmp/1res/camera_geometric_residual_gate/v1/features/`。
- NAS 小型正式结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_geometric_residual_gate/v1/`。
- 执行说明：`docs/camera_geometric_residual_gate_execution_20260719.md`。
- 代码入口：`scripts/camera_geometric_residual_gate/run.sh`。

立即下一步：先执行 preflight 与 8 样本 smoke，再运行全量 DataB 到 ViF-Bench 冻结特征门。若门失败，停止相机几何主线；若通过，下一轮只把冻结几何块通过小 projector/gate 注入共享检测器，并继续使用原 detection SFT 损失，不立即加入 DataA、CoT、RL 或额外多任务损失。

## 记录维护说明

- 新实验开始时先在本文件新增中文实验定义和验收标准。
- 用户提供结果后，在对应小节补充指标、结论和下一步，不创建含义重复的新代号章节。
- 未知值保留为 `待补充`，不根据上下文猜测。
- `docs/final_experiment_plan_20260708.md` 是受保护文件，不在本记录维护过程中修改。
