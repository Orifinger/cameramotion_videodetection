# DataB Skyra 风格 GRPO 奖励动力学诊断

这套代码不用于证明最终检测泛化性能，而是从完整 DataB 检测模型出发，用相同数据、采样顺序和训练参数比较不同奖励函数如何改变 GRPO 曲线、真假预测偏置和证据输出行为。

## 数据合同

- 起始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`
- 原始 DataB：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`
- verl：commit `2c9e19ef2f0619a2e9e9d4fc813dab8e717e3ab9`
- DataB 中 17 帧样本确定性均匀选取 16 帧；唯一 11 帧样本被剔除；随后将两类下采样到相同数量。
- 每类固定取 256 条作为 GRPO 过程诊断集。该集合被 GRPO 更新留出，但已经被继承的 SFT checkpoint 看过，不是真正的 held-out 测试集。

## 奖励版本

- `paper_asymmetric_inspection`：按论文文字与公式实现非对称分类奖励和正则证据计数奖励。
- `symmetric_zero_inspection`：两种误判的分类奖励都为 0，复现论文所述的 Fake 偏置消融。
- `asymmetric_outer_format`：用外层格式奖励替代 inspection reward。
- `asymmetric_answer_only`：只保留非对称真假分类奖励。
- `strict_unique_inspection`：只奖励分类正确且时间、类别、bbox 均有效的唯一证据块。
- `inspection_only_hackable`：只奖励可由正则命中的证据数量，用于主动观察 reward hacking。
- `official_repository_bug`：复现 Skyra 仓库公开 `ladm.py` 的实际返回逻辑，仅作为错误实现诊断。

## 记录的核心曲线

除了 verl 自带的 reward、KL、entropy、policy loss、grad norm 和耗时外，补丁会写入 TensorBoard：分类正确率、预测 Fake 比例、FP/FN、格式有效率、证据数量、严格有效证据数量、重复/无效框率、错误回答仍获正奖励比例，以及 GRPO 组内零奖励方差率。

所有逐样本 rollout 暂存在 `/tmp/1res/skyra_grpo_diagnostics/<run>/rollouts`。压缩后的曲线 CSV、摘要、TensorBoard 和日志写入 NAS 的 `res/skyra_grpo_diagnostics/<run>`。

`run.sh` 默认仍使用 `SAVE_FREQ=-1`，只做曲线诊断；需要固定评测时显式设置正整数保存间隔。`run_evalable_100step.sh` 固定复跑论文式奖励 100 步，并且只改变 checkpoint 保存：step 50 和 step 100 写入 `/tmp`，训练结束后将整个运行目录上传 OSS，再启动 `/input/training/keep.sh`。该 wrapper 不改变数据、seed、奖励、batch、group size、学习率、KL 或回复长度。

`run_vifbench.sh` 使用 V4Train 已有的原始 no-camera 检测提示词和全量 VIF-Bench index，同时评测合并后的 step 50 与 step 100。原检测 checkpoint 的严格同提示词结果直接复用持久化的 `base_vifbench_eval.json`，避免无信息增量的第三次完整推理。逐条预测留在 `/tmp`，比较 JSON、官方评测 CSV 和日志写入 NAS；这项评测不使用 camera 数据。
