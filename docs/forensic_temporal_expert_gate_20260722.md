# 原生尺度时序取证专家两层验证方案

## 结论先行

本轮不再继续证明“camera 有用”，而是验证一个更基础、可证伪的问题：**冻结视觉表征中的正确视频顺序，是否含有能泛化到全生成视频检测的增量取证信号；若有，它是否与强 Qwen 检测器的残余错误互补。**

只有两个条件同时成立，才值得把时序专家做成论文方法模块。任一条件失败，都停止在该架构上增加 MLLM token、路由器或 RL。

## 与旧实验的区别

旧 CTNE/RAFT 实验同时耦合了相机估计、光流残差和密度模型，失败后无法确定是相机假设、手工特征还是时序本身有问题。本轮隔离变量：

- 不使用 camera labels/caption；
- 不使用 RAFT、光流、单应性或相机补偿；
- 不微调 DINOv2；
- 只改变是否保留正确帧顺序；
- 训练目标始终是最终 Real/Fake；
- 用等容量乱序训练和同模型乱序输入构成因果控制。

## 数据

### DataB

完整使用 v4vif_2766busterall_trainall.json 的 6766 条。GenBuster 原始 train/test 只保留为元数据，不删除 1242 条。按来源、生成器、标签和 group 分层为 5 folds；fold 0 开发验证，fold 1-4 训练。同一 group 不跨 fold，帧数保持 11/16/17。

### ViF-Bench

3160 条仅用于开发门。模型、阈值和融合权重不得根据 ViF 标签拟合。ViF 已多次参与方向选择，不能作为最终无偏测试。

### GenBuster Closed Benchmark

本轮不读取、不统计、不调参。两个 Gate 都通过后才冻结配方、用完整 6766 条重训并进行一次最终测试。

## Gate 1：正确时序是否有增量

DINOv2-S/14 对每帧提取 CLS token 和局部 4x4 patch token。图像只做等比例缩小及 patch 对齐，不裁剪；模型使用全部已列帧并用长度 mask 处理可变帧数。

三个分支：

1. 顺序不敏感专家：空间 token 编码后做跨帧集合聚合。
2. 正确顺序专家：加入相邻帧差分和时序 GRU。
3. 乱序监督专家：与正确顺序专家等架构、等数据、等优化器，只打乱帧序。

额外把正确顺序专家在评测时喂确定性乱序帧。主指标是 ViF AUROC 与生成器宏平均 Balanced Accuracy，阈值仅由 DataB fold 0 确定。

通过条件：

- ordered 相对 static 和 shuffled 各自在至少一个主指标提高 1.5 点，另一个不下降超过 1 点；
- ordered 输入乱序后至少一个主指标下降 1 点；
- 2/3 seed 方向一致；
- Real Recall 相对 static 不下降超过 3 点；
- 特征覆盖率不低于 99%。

## Gate 2：是否与强 Qwen 互补

沿用历史强 Qwen 的 ViF 硬答案和已补算 Real/Fake token confidence，不重新生成 CoT。融合公式预先固定：

~~~text
fused_logit = qwen_logit + 0.25 * expert_logit
~~~

对 static、ordered-shuffled-input 和 shuffled-trained 做相同融合控制，不在 ViF 上训练路由器。

通过条件：

- Gate 1 已通过；
- ordered 固定融合相对 Qwen 至少一个主指标提高 1.5 点；
- group bootstrap 95% 区间下界大于 0；
- ordered 融合相对每个控制至少有一个主指标多 1 点；
- Real Recall 下降不超过 1 点；
- Qwen confidence 覆盖率与历史答案复现率均不低于 99%。

## 解释边界

- Gate 1 通过只说明正确顺序提供可泛化判别增量，不说明 MLLM 已利用它。
- Gate 2 通过说明该增量可转化为最终二分类收益，仍不是最终测试结论。
- Gate 1 失败说明当前冻结 DINO 表征和 DataB 监督不足以支持时序主线。
- Gate 1 通过、Gate 2 失败说明专家信号与 Qwen 不互补，不能靠更复杂融合自动解决。
