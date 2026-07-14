# GRPO 后训练复现学习记录

本文件是当前学习会话的统一记录，专门保存 Skyra 风格 GRPO 复现、奖励函数分析、训练曲线诊断、checkpoint 固定评测和后续奖励消融。

- 这些实验用于学习和分析后训练，不属于相机条件化 AIGC 视频检测论文主线。
- 后续本会话产生的 GRPO 训练、推理、评测和结果修正只更新本文件，不写入 `docs/camera_conditioned_experiment_log.md`。
- 训练中使用 DataB 或 VIF-Bench 只表示复用现有检测任务环境，不应把本文件中的结果表述为 camera 方法贡献。
- 未知信息保留为 `待补充`；历史结论通过追加更正维护，不静默覆盖。

## 当前学习索引

| 日期 | 学习实验 | 状态 | 核心结论 |
|---|---|---|---|
| 2026-07-14 | 论文式非对称奖励 100 步训练 | 训练链路通过，检测效果当时结论不足 | 训练稳定；分类信号接近饱和，奖励增长更多来自证据计数 |
| 2026-07-14 | 完整 rollout 奖励行为审计 | 通过 | 未见重复框或非法框型 reward hacking，但仅 14% prompt 组直接提供答案差异信号 |
| 2026-07-14 | 保存 step 50/100 的可评测复跑 | 通过 | 两个 FSDP checkpoint 均成功合并，tokenizer/processor 输入合同与 base 一致 |
| 2026-07-15 | 完整 VIF-Bench 固定评测 | 未通过 | Fake Recall 上升，但 Real Recall 下降更多；Balanced ACC 和 Fake F1 均低于 base |

## 1. 完整 DataB 检测模型的 Skyra 风格 GRPO 奖励动力学诊断

### 这个实验测什么

从已经在完整 DataB 上完成检测 SFT 的 Qwen3-VL-8B 出发，在相同数据、采样顺序和训练超参数下，只更换规则奖励函数，观察 GRPO 的分类正确率、Fake 预测比例、FP/FN、格式、证据数量、KL、熵、策略损失、梯度范数和组内奖励方差如何变化。本实验的目标是学习并验证后训练曲线的分析方法，不用于证明相机条件化方法有效。

### 状态、模型与数据

- 日期：2026-07-14。
- 状态：`已完成；100 步训练链路与 rollout 合同通过，因未保存 checkpoint 而无法判断固定评测效果`。
- 起始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 原始训练数据：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`，6766 条，初始 Fake/Real 各 3383 条。
- verl 版本：commit `2c9e19ef2f0619a2e9e9d4fc813dab8e717e3ab9`，服务器目录 `/input/workflow_58770161/workspace/test/cameramotion_det/third_party/verl-2c9e19ef2f0619a2e9e9d4fc813dab8e717e3ab9`。
- 转换规则：17 帧样本确定性均匀取 16 帧；唯一 11 帧样本剔除；类别下采样平衡后，训练集 6252 条，GRPO 过程诊断集 512 条。
- 小型可复用数据与审计保存到 NAS：`/input/workflow_58770161/workspace/test/cameramotion_det/res/skyra_grpo_diagnostics/data`。
- 逐样本 rollout 原始产物位于 `/tmp/1res/skyra_grpo_diagnostics/<run>/rollouts`；压缩曲线、TensorBoard、运行清单和日志持久化到 NAS 的 `res/skyra_grpo_diagnostics/<run>`。本轮未保存 checkpoint；训练结束后已将 rollout 与小结果打包上传 OSS，位置见本节完整 rollout 审计。

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

2026-07-14 解释更正：复核 Skyra camera-ready 论文后，确认本轮 `paper_asymmetric_inspection` 的数学定义与论文公式一致，即 `0.8 × r_acc + 0.2 × r_check`；正确分类为 1，Fake→Real 为 0，Real→Fake 为 -0.2，检查奖励为 `min(log(1+N_check), log(4))`。因此证据块数量上升本身是论文奖励的预期优化方向，不能单独据此把训练判为失败或 reward hacking。前述“原奖励不宜直接延长”只表示现有曲线不足以支持盲目扩训，不表示模型已经训坏。

本轮只严格对齐了论文公开的奖励公式、16 帧输入、actor 学习率 `5e-7` 和 KL 系数 `0.02`；它不是官方完整复现。差异包括 Qwen3-VL-8B 对 Qwen2.5-VL-7B、DataB 自动解释对 ViF-CoT-4K 人工细标、16 张 PPU 对 8 张 H200，以及论文未公开而由本项目自行选择的 group size、batch、PPO epoch、回复长度和总步数。官方 GitHub 当前 `ladm.py` 与论文公式还存在实现和注释不一致，本轮有意按论文公式而非逐行复制仓库实现。

更正后的结论标记：`结论不足（针对检测效果）`，同时保留 `通过（针对训练链路和曲线诊断）`。Skyra 论文中的 RL 相对 SFT 也仅从 ViF-Bench ACC 90.11% 提升到 91.02%、F1 88.76% 提升到 90.27%，说明这种收益必须通过固定验证集检测，无法从训练奖励曲线可靠推断。本轮未保存终点 checkpoint，不能补做现有模型评测；下一步优先级修正为从同一起点按相同论文式奖励复跑并保存中间/终点 checkpoint，先比较固定 512 条策略漂移诊断集和 VIF-Bench 同协议结果。只有验证无提升或退化后，才进入奖励消融或学习率、KL、采样温度等调参。

### 2026-07-14 完整 rollout 奖励行为审计

这次新增审计的是 100 步训练产生的全部 12,800 条回答，而不是新的训练或模型评测。它检查同一 prompt 的 8 个候选之间究竟由分类差异还是证据计数差异产生 GRPO 信号，并检查证据数量增长是否伴随重复块、非法块、全帧框或全时段检查的恶化。

结果来源：

- 服务器 rollout：`/tmp/1res/skyra_grpo_diagnostics/paper_asymmetric_inspection_formal_100step/rollouts/`。
- OSS 运行目录：`oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/skyra_grpo_diagnostics/paper_asymmetric_inspection_formal_100step/`。
- 本地完整包：`E:/newgaibeishi/paper_asymmetric_inspection_formal_100step_analysis_bundle.tar.gz`。

完整性检查为 100 步、1,600 个 prompt 组、每组严格 8 个回答；1,600 个训练样本身份全部唯一，每个只出现一次。后 10 步平均每组 7.90/8 个回答文本不同，没有观察到逐字回答复制式塌缩。

| 组内学习信号 | 全程 1,600 组 | 前 10 步 | 后 10 步 |
|---|---:|---:|---:|
| 零任务奖励方差 | 52.56% | 50.63% | 65.63% |
| 存在答案/正确性差异 | 14.00% | 15.00% | 10.00% |
| 答案相同、仅证据计数不同 | 33.44% | 34.38% | 24.38% |
| 8 个候选全部分类正确 | 85.94% | 85.00% | 90.00% |

因此，本轮真正直接训练 Real/Fake 选择的 prompt 组只占 14.00%；更多非零信号来自同一分类答案下的证据块数量差异。后程零方差升高主要是起点已经很强、同组答案趋同且证据块数量也趋同，并非数值故障。

| 输出行为 | 前 10 步 | 中间 46–55 步 | 后 10 步 |
|---|---:|---:|---:|
| Rollout 分类正确率 | 95.08% | 96.88% | 96.64% |
| 恰好两条检查的回答比例 | 68.28% | 76.72% | 88.98% |
| 精确全帧 bbox 的检查比例 | 14.31% | 12.12% | 12.04% |
| 覆盖至少 90% 视频时长的检查比例 | 70.17% | 71.86% | 66.02% |
| 同时为大框和长时段的检查比例 | 10.93% | 9.69% | 9.36% |

模型明显收敛到“每个回答约两条合法检查”，但精确全帧框和长时段检查没有随训练增加，重复块与非法块也接近零，所以这 100 步没有出现通过复制证据或扩大时空范围来刷分的恶化。不过奖励只验证类别名、时间和 bbox 的语法及范围，不核对这些证据是否与画面或自动 CoT 真正一致；本包也不含可供人工复核的图像，因此不能声称解释忠实度提高。

对 497 条错误回答进一步复核后，虽然 446 条得到正的绝对任务奖励，但没有一条高于所在 prompt 组的任务奖励均值，也没有一条达到该组任一正确候选的最低任务奖励。原因是正确答案最低可获得的分类主项仍显著高于错误答案能从证据计数得到的上限；因此先前把“错误回答仍获正绝对奖励”直接称为奖励泄漏不够准确。它是奖励可解释性风险，但在本次实际候选组中没有把错误分类提升为正的组内相对学习方向。

结论标记：`通过（rollout 合同和训练稳定性）`；`结论不足（检测提升与解释忠实度）`。本轮没有发现分类被证据计数奖励反转，也没有发现明显格式型 reward hacking；但分类学习信号已接近饱和，训练主要在统一证据数量。下一步仍是原样复跑并保存 step 50/100 checkpoint，然后做固定 DataB 策略漂移诊断和同协议 VIF-Bench；没有 checkpoint 前不应通过继续阅读 rollout 来推断检测提点。

### 2026-07-14 论文式奖励可评测复跑

这次复跑测试同一论文式 GRPO 更新在固定评测上的真实影响。与第一轮相比，唯一改变是把 checkpoint 保存间隔从关闭改为每 50 步一次；数据、样本顺序、seed、prompt、奖励、学习率、KL、batch、每组候选数、回复长度和总步数全部保持不变，不能在本轮中夹带调参。

- 状态：`代码已就绪，待服务器执行`。
- 起始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 训练数据：`/input/workflow_58770161/workspace/test/cameramotion_det/res/skyra_grpo_diagnostics/data/datab_grpo_train.parquet`，6252 条平衡 DataB 记录。
- 运行名：`paper_asymmetric_inspection_evalable_100step`。
- 关键设置：`paper_asymmetric_inspection`、100 步、prompt batch 16、每 prompt 8 个候选、学习率 `5e-7`、KL 系数 `0.02`、seed `20260714`、最多 768 个回复 token。
- 唯一变化：`SAVE_FREQ=50`，预期生成 `global_step_50` 和 `global_step_100`。

输出存储按以下三类处理：

- 可丢弃训练中间输出：Ray 临时文件留在 `/tmp`，不保存。
- 持久化小结果：manifest、TensorBoard、曲线、日志、退出码和 checkpoint inventory 写入 `/input/workflow_58770161/workspace/test/cameramotion_det/res/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step/`。
- 可复用大结果：两个 FSDP checkpoint 与 rollout 写入 `/tmp/1res/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step/`；训练结束后自动上传到 `oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step/`，随后才执行 `/input/training/keep.sh`。

训练阶段通过条件是训练退出码、checkpoint inventory 和 OSS 上传退出码全部为 0，并且 step 50/100 均存在 `actor` checkpoint。训练通过本身仍不代表模型改进。随后必须用相同 prompt、index、生成参数和解析器比较起始模型、step 50 和 step 100：512 条 DataB 只称策略漂移诊断集，因为继承的 SFT 模型已看过；VIF-Bench 未进入本轮训练，但已被项目多次用于选型，也不称完全未触碰的最终测试集。固定评测完成前，结果状态保持 `结论不足`。

立即下一步：服务器只覆盖新版 `run.sh`、`run_evalable_100step.sh` 和 `README.md`，执行 `bash -n` 后后台启动 wrapper。训练结束先检查三个退出码和 checkpoint inventory，再合并 step 50/100 的 FSDP actor 权重用于固定评测；不先启动奖励消融。

### 2026-07-14 可评测复跑训练与模型合并结果

用户报告 `paper_asymmetric_inspection_evalable_100step_retry1` 已完成训练；当前提供的是 step 50/100 的 FSDP 合并终端输出和合并目录检查，不包含固定评测指标。训练、checkpoint audit、OSS 上传和 pipeline 四个退出码尚未在本记录中补充。

结果来源：

- FSDP checkpoint：`/tmp/1res/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step_retry1/checkpoints/`。
- 合并模型：`/tmp/1res/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step_retry1/merged_step_50` 和 `merged_step_100`。
- FSDP checkpoint 的 OSS 运行目录：`oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step_retry1/`；上传退出码待补充。

| 模型 | FSDP 分片加载 | 导出权重 | 合并目录大小 | 状态 |
|---|---:|---:|---:|---|
| step 50 | 16/16 | 4 个 safetensors | 17 GB | 合并完成 |
| step 100 | 16/16 | 4 个 safetensors | 17 GB | 合并完成 |

合并过程中 Transformers 对 tokenizer 输出 `fix_mistral_regex=True` 提示；该提示没有中断 Qwen3-VL 模型、processor 或 tokenizer 的保存，但在固定评测前必须比较起始模型与两个合并模型的 tokenizer 类、词表、特殊 token id 和代表性 prompt token ids，不能仅凭 `config.json` 和权重文件存在就忽略。

结论标记：`通过（checkpoint 保存与 FSDP 合并）`；`结论不足（检测效果）`。当前只证明两个训练时点已经转成可加载的 HF 目录，不证明 step 50 或 step 100 优于起始模型。

### 2026-07-14 tokenizer 审计与固定评测变更

起始检测模型、GRPO step 50 和 step 100 的等价审计均加载为 `qwen3_vl / Qwen3VLProcessor / Qwen2VLImageProcessorFast / Qwen2TokenizerFast`，同一代表性训练 prompt 都渲染为 926 个 token；聊天模板文本、token ids、词表、特殊 token 和 processor 类逐项一致，最终输出 `TOKENIZER_PROCESSOR_EQUIVALENCE: PASSED`。合并模型触发的 `fix_mistral_regex=True` 警告没有在本次 Qwen3-VL 输入合同中造成差异。

用户明确决定不再先跑 512 条 DataB 策略漂移诊断，直接评测完整 VIF-Bench。这个变更减少一次已被继承 SFT checkpoint 看过的数据内诊断，代价是如果输出协议异常，完整推理后才会发现；tokenizer/processor 审计已降低其中的模型合并风险。

- 状态：`准备执行完整 VIF-Bench`。
- 模型：原检测 checkpoint 的既有严格同提示词基线、`merged_step_50`、`merged_step_100`。
- 测试数据：VIF-Bench 当前 16 个 index shard，共 3160 条预期输入；它未参加本轮 GRPO，但已在项目中多次查看，不能称全新最终 held-out test。
- 单一改变因素：只替换 base、GRPO step 50、GRPO step 100 的模型权重；VIF index、16 帧输入、system prompt、no-camera user suffix、确定性生成和解析器全部相同。
- 推理不提供 camera caption 或 camera label；本实验也没有使用 camera 训练数据，不能写成相机条件化结果。
- 复用基线：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/vifbench_detection_checkpoint_start/eval/base_vifbench_eval.json`，此前覆盖 3160/3160、格式有效率 99.97%、跨生成模型平均 Balanced ACC 79.18%、Fake F1 80.47%。
- 完整性门：三个模型覆盖率和格式有效率均至少 99%，且生成模型子集完全一致。
- 提升门：相对 base，某个 GRPO checkpoint 的 Balanced ACC 与 Fake F1 均不得下降超过 0.1 个百分点，且二者至少一项提高 0.5 个百分点；若两个 checkpoint 都通过，选择两项均值更高者。
- 可丢弃输出：step 50/100 逐条预测与合并预测 JSON 放在 `/tmp/1res/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step_retry1/vifbench/`。
- 持久化小结果：比较 JSON、逐模型评测 JSON、官方 CSV 和日志写入 `/input/workflow_58770161/workspace/test/cameramotion_det/res/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step_retry1/vifbench/`。
- 可复用大结果：仍是已合并的两个 17 GB 模型；本次评测不产生新的可复用大文件，不追加 OSS 上传。

立即下一步：使用 16 GPU 并行运行 step 50 与 step 100 的完整 VIF-Bench，读取 `vifbench_grpo_checkpoint_comparison.json` 决定论文式 GRPO 是提升、持平还是退化；结果出来前不启动奖励消融或继续延长训练。

### 2026-07-15 完整 VIF-Bench 固定评测结果

这次实际比较的是原 DataB 检测 SFT checkpoint、完全相同论文式 GRPO 的 step 50 和 step 100，在同一完整 VIF-Bench index、原始 no-camera 检测提示词、确定性生成参数和解析器下的检测效果。三套结果覆盖相同的 3160 条输入和 19 个生成模型子集；本轮没有 camera 数据或 camera 文本。

结果来源：

- 比较汇总：`/input/workflow_58770161/workspace/test/cameramotion_det/res/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step_retry1/vifbench/vifbench_grpo_checkpoint_comparison.json`。
- step 50/100 逐模型评测、官方 CSV 与日志：同一 NAS 目录。
- 逐条预测：`/tmp/1res/skyra_grpo_diagnostics/paper_asymmetric_inspection_evalable_100step_retry1/vifbench/`，属于可丢弃评测输出。

| 模型 | 覆盖率 | 格式有效率 | Balanced ACC | Fake Recall | Fake F1 | 推导 Real Recall |
|---|---:|---:|---:|---:|---:|---:|
| 原 DataB 检测模型 | 100.00% | 99.97% | 79.18% | 89.33% | 80.47% | 69.03% |
| GRPO step 50 | 100.00% | 99.87% | 78.33% | 90.58% | 80.14% | 66.09% |
| GRPO step 100 | 100.00% | 99.87% | 77.85% | 91.33% | 80.01% | 64.36% |

| 相对原模型 | Balanced ACC | Fake Recall | Fake F1 | 推导 Real Recall |
|---|---:|---:|---:|---:|
| step 50 - base | -0.85 点 | +1.25 点 | -0.33 点 | -2.95 点 |
| step 100 - base | -1.33 点 | +2.00 点 | -0.46 点 | -4.67 点 |

完整性检查全部通过：三套模型覆盖率均为 100%，格式有效率均高于 99%，生成模型集合完全一致。因此下降不能归因于缺失预测、输出接口崩坏或测试子集不一致。由 `Real Recall = 2 × Balanced ACC - Fake Recall` 推导可见，GRPO 随步数增加逐渐提高 Fake 命中，但同时制造更多 Real→Fake，Real Recall 的损失明显大于 Fake Recall 的收益，最终 Balanced ACC 和 Fake F1 均下降。

结论标记：`未通过（完整 VIF-Bench 检测提升门）`。当前结果足以停止继续延长同一 `paper_asymmetric_inspection` 配方；它不证明 GRPO 对本任务无效，也不能区分漂移主要来自非对称分类奖励、证据计数奖励还是训练数据内高起点饱和。VIF-Bench 未参与本轮 GRPO，但已在项目中多次用于选型，因此这是可信的同协议外部分布诊断，不称全新最终 held-out test。

立即下一步：不再补跑 512 条数据内诊断来挽救该结论，也不直接调学习率。先复用现有逐条预测做逐生成模型变化和 Real/Fake 决策翻转审计，确认漂移是否广泛存在；随后若继续奖励消融，优先从同一起点做等步数的“只保留非对称真假奖励”分支，以隔离证据计数项，而不是延长当前 checkpoint。
