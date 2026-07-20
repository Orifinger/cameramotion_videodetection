# 源内容去偏的原生尺度 AIGC 视频取证方案（2026-07-21）

## 1. 研究决策

本项目从 2026-07-21 起重置论文主问题，不再要求 camera motion 成为方法贡献。重置的是研究命题、主模型职责和评测协议，不是把已验证的数据与工程资产全部作废。

新的核心问题是：**现有 MLLM 检测器在高质量 I2V/局部编辑视频上容易把保留下来的真实源内容当成真实性证据，能否通过训练时同源 Real/Fake 配对监督和原生尺度局部取证，学到在单视频推理时仍可用的生成痕迹？**

暂定中文方法名为“源内容去偏的原生尺度取证检测”，英文工作名为 **Source-Paired Native-Scale Forensics（SPNF）**。英文名只用于代码与论文占位，不作为未经实验支持的正式方法名。

## 2. 为什么重新选择这个问题

### 2.1 项目内部证据

- 强 DataB detection Qwen 在 ViF-Bench 的 319 个有效错误中，281 个是 Fake→Real。
- 六类 I2V/编辑型生成器只占 33.0% 的 Fake，却贡献 71.2% 的 Fake→Real；漏检率约为其他生成器的 5 倍。
- Fake→Real 的 CoT 更接近同源 Real 的验证模板，且模型置信度仍高度饱和，说明问题不是简单调阈值或低置信路由。
- camera 文本、正确/打乱 camera、RAFT/DINO 残差、相机 VQA、硬路由和置信度融合均没有形成稳定最终 Real/Fake 增量。

这些结果共同支持“源内容捷径”是比 camera 更直接、也更可证伪的研究切入点。完整证据见 `docs/vifbench_residual_error_analysis_20260720.md`。

### 2.2 外部工作依据

- [Aligned Datasets Improve Detection of Latent Diffusion-Generated Images（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ead108421b202494d01b5060d12aa34-Abstract-Conference.html)表明，语义对齐的 Real/Fake 数据能减少内容、分辨率和文件格式捷径。
- [CoCoVideo / CoCoDetect（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_CoCoVideo_The_High-Quality_Commercial-Model-Based_Contrastive_Benchmark_for_AI-Generated_Video_Detection_CVPR_2026_paper.html)提供高质量商业模型的语义对齐视频对，并证明配对对比监督适用于 AIGC 视频检测。
- [Preserving Forgery Artifacts at Native Scale（ICLR 2026）](https://arxiv.org/abs/2604.04634)表明固定 resize/crop 会破坏高频生成痕迹，并公开了 native-scale Qwen2.5-ViT 检测权重。
- [Seeing What Matters（NeurIPS 2025）](https://openreview.net/forum?id=dOGXKBL7IE)与 [D3（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.html)分别支持取证导向增强和差异学习，但也要求同时检查 ID 与 OOD，不能只优化单个 benchmark。

本项目不是简单复现上述任一工作，而是检验：**训练时使用同源配对和局部掩码去除源内容捷径，是否能蒸馏出推理时只需单视频的原生尺度局部证据。**

## 3. 明确不做什么

- camera motion 不再作为输入条件、路由依据或主要贡献；最多作为事后难度分层变量。
- 不把 Qwen3-VL 的自由生成 CoT 当作真假标签或取证真值。
- 不先大规模重标 DataB，也不在方法信号出现前调用公司 API 扩充大量视频。
- 不从零预训练视频 backbone，不在第一轮同时引入 RL、MoE、多代理或复杂多损失调度。
- ViF-Bench 已参与大量方向选择，只作为开发集；不再把它包装成独立最终测试。

## 4. 数据资产重新分工

| 数据 | 保留内容 | 新角色 | 不再承担的角色 |
|---|---|---|---|
| DataB，6766 条 detection SFT | 帧、可靠 Real/Fake 来源标签、生成器信息 | 通用分类 replay 与已知生成器训练 | 自动 CoT 不作为局部取证真值 |
| DataA，1080 个 Real/Fake case | 同源配对、编辑 mask/bbox、来源 family | 配对排序与局部证据监督 | 不作为通用全生成检测主测试集 |
| CoCoVideo-26K | 高质量语义对齐 Real/Fake 对、生成器信息 | 跨生成器配对训练与留出验证 | 不允许 train/test generator 混用 |
| ViF-Bench | 既有逐样本结果与生成器划分 | 开发诊断、I2V 困难子集 | 独立最终测试 |
| GenBuster benchmark | 帧与来源标签 | 冻结后的主要外部测试 | 方法选择和阈值搜索 |

所有数据先建立统一 manifest，记录真实帧数、原始分辨率、codec、生成器、source/pair ID、路径和哈希。任何代码都不得假定固定 16 帧。同源视频、近重复视频和同一 source ID 必须在 split 中保持组隔离。

## 5. 模型职责重新划分

### 5.1 主检测器

最终 Real/Fake 指标首先由专用视觉取证模型负责。第一轮候选为：

1. native-scale Qwen2.5-ViT 448p 公布权重；
2. native-scale Qwen2.5-ViT 720p 公布权重；
3. CoCoDetect/R3D-18 作为轻量时空对照；
4. 当前 Qwen3-VL detection checkpoint 只作为历史 MLLM 基线。

先用公开预训练权重做能力门，不从零训练。正式骨干由开发门结果决定，而不是预先指定 Qwen3-VL。

### 5.2 训练时配对，推理时单视频

对同源 Real/Fake 对分别独立前向，得到单视频真假分数 `s(r)`、`s(f)`。配对信息只通过训练损失约束 `s(f) > s(r)`，推理时不需要 Real 参考视频：

```text
L = L_cls + lambda_pair * max(0, margin - s(f) + s(r))
            + lambda_loc * L_localization
```

- `L_cls`：DataB、DataA、CoCoVideo 的单视频二分类损失。
- `L_pair`：只用于真实匹配的 DataA/CoCoVideo pair；打乱 pair 是关键负对照，不是训练增强。
- `L_localization`：只在具有可信 mask/bbox 的 DataA fake 上监督局部证据图；对应 Real 的目标为空证据。

第一轮只验证 `L_pair` 是否真正去除源内容捷径。局部证据头和取证增强在配对门通过后再加入，避免一次改动多个因素。

### 5.3 Qwen3-VL 的后续角色

只有主检测器在外部数据上通过后，才把全局低分辨率帧、最高分的原生尺度局部 patch、时间戳和检测分数交给 Qwen3-VL 生成解释。第一版最终真假 verdict 保留为取证检测器输出，防止高置信幻觉覆盖正确证据。解释质量单独评估“证据命中、无依据伪影率和人工偏好”，不能替代二分类指标。

## 6. 分阶段硬验收

### 阶段 0：数据与公开 backbone 可行性门

这一步不训练大模型，回答“现有数据契约是否干净、原生尺度取证模型是否能看到 Qwen 漏掉的 I2V 信号”。

必须完成：

- DataA、DataB、ViF-Bench、GenBuster、CoCoVideo 的 manifest、哈希去重、source/generator 分组审计；
- DataB 与 GenBuster benchmark 的帧级/感知哈希重叠审计；
- 448p、720p native-scale 权重及固定 resize 对照在 ViF 开发集上的逐生成器评测；
- 专门报告 Hunyuan-I2V、Wan-VACE 和六类 I2V/编辑生成器的 Fake recall。

| 判定 | 标准 | 动作 |
|---|---|---|
| 强通过 | 困难 I2V 子集 AUROC ≥ 0.70，或 Fake recall 相对 Qwen 提高 ≥10 点；且 native 输入相对固定 resize 至少有一个稳定优势 | 进入配对训练门 |
| 边界 | AUROC 0.60–0.70，或只有 native/resized 差异而未超过 Qwen | 只做一轮小规模配对训练 |
| 硬失败 | 两个 native 权重均 AUROC ≤0.55，且 native/resized 无差异 | 停止该 backbone，比较 CoCoDetect/D3 类差异模型，不跑全量训练 |

阶段 0 只是方向门，不能作为论文最终结果；公开权重可能与 benchmark 存在训练重叠，必须在报告中注明。

### 阶段 1：正确同源配对是否产生检测增量

三个分支使用同一 backbone、相同单视频样本、batch、步数、增强和 `L_cls`：

| 分支 | 唯一变化 |
|---|---|
| 单视频分类对照 | 不使用 pair rank；用等量重复样本保持更新步数一致 |
| 正确同源配对 | DataA/CoCoVideo 的真实 pair 用 `L_pair` |
| 打乱配对控制 | 在同 generator、同标签结构内打乱 source partner，再用相同 `L_pair` |

通过必须同时满足：

1. 正确配对分别超过单视频对照和打乱配对至少 1.0 个 ViF 跨生成器 Macro Balanced ACC 点；
2. 六类 I2V/编辑生成器 Fake recall 至少提高 3 点；
3. Real recall 相对单视频对照下降不超过 1 点；
4. 按 source ID 分组 bootstrap 的两项差值 95% CI 下界均大于 0；
5. 在 CoCoVideo 未见生成器留出集上方向一致。

没有同时通过时，不增加局部头、取证增强或 RL；先分析是数据对不齐、backbone 无信号，还是 pair loss 只记忆生成器。

### 阶段 2：局部证据与取证增强

阶段 1 通过后，比较“正确配对”与“正确配对 + DataA mask 局部证据头”。再单独加入 wavelet、压缩或 resize 等取证导向增强，每次只增加一个因素。

通过标准：

- 通用检测主指标不低于阶段 1 最优模型；
- DataA mask 的区域 AP/IoU 明显超过无局部监督对照；
- resize、重编码、JPEG/视频压缩和轻度模糊下的平均鲁棒性提高；
- 局部证据在 Real 上不过度激活，并在打乱 mask 控制下消失。

### 阶段 3：冻结方法后的外部测试

模型、超参数和阈值在看 GenBuster benchmark 标签前冻结。最终至少报告：

- GenBuster benchmark 主结果与逐生成器结果；
- CoCoVideo 未见生成器、ViF-Bench 开发结果；
- ID/OOD、T2V/I2V/编辑型、分辨率和 codec 分层；
- source-group bootstrap 95% CI；
- 与 Qwen3-VL、CoCoDetect、native-scale 原模型和当前主流方法的等协议比较。

若只在 ViF 提升、GenBuster 和未见生成器不提升，论文主张判定为未成立。

### 阶段 4：证据约束解释

只在检测主线成立后进行。使用检测器证据 patch 构造 grounded explanation SFT；必要时再考虑短程 RL。奖励必须包含最终答案正确、证据区域一致和无依据伪影惩罚，且不能只奖励格式或冗长 CoT。

## 7. 论文贡献的最小闭环

若实验通过，论文只主张三件可直接验证的事：

1. 现有 MLLM 视频检测器在源内容保持型 I2V/编辑视频上存在系统性内容捷径；
2. 训练时同源配对能把模型从语义内容推向生成取证信号，并在单视频推理中改善未见生成器；
3. 原生尺度与局部证据监督进一步保留、定位该信号，并为语言解释提供外部证据。

Camera 可作为困难度分析的一列，但除非新的匹配/打乱因果控制重新通过，不进入标题、摘要或方法贡献。

## 8. 实施顺序与资源

1. **先做本地/CPU 数据审计代码**：统一 manifest、真实帧数统计、source split、哈希泄漏审计。
2. **服务器做公开权重能力门**：两台 16×96G 服务器可分别跑 448p 与 720p；结果为小 JSON/CSV，持久化到 NAS。
3. **能力门通过后再下载 CoCoVideo 并跑三分支小训练**：两台服务器并行正确配对与打乱配对，单视频对照在先结束的服务器补跑。
4. **三分支门通过后才产生正式 checkpoint**：大 checkpoint 写入 `/tmp`，完成审计后用一条 `ossutil64 cp -r` 上传；manifest、配置和指标留 NAS。

阶段 0 和阶段 1 预计是第一轮完整决策周期。此前不需要重新生成 DataA、不需要重新标全量 CoT，也不需要安装 RL 环境。

## 9. 当前状态与立即下一步

- 状态：`方案已立项，阶段 0 待实现`。
- 第一项工程任务：编写统一数据 manifest 与泄漏审计工具，并锁定公开 448p/720p 权重的下载清单和推理接口。
- 第一项科学产出：得到“native vs resize × generator family × Qwen correct/error”逐样本表，而不是先跑新的五轮 SFT。
- 任何阶段失败都保留完整负结果和原因；失败不会被下一种复杂训练自动覆盖。