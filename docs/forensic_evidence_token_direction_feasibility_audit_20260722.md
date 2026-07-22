# 取证证据增强 MLLM 视频检测：方向可行性审计

更新时间：2026-07-22  
证据截止：2026-07-22  
审计对象：**用原生尺度或局部取证专家提取视觉证据，将证据与 Qwen3-VL 一类 MLLM 直接耦合，并以最终 Real/Fake 判定为主目标；在缺少局部标注时，考虑用视频级标签和 MIL 学习局部证据。**

## 1. 最终裁决

| 裁决层级 | 结论 | 含义 |
|---|---|---|
| 科学问题 | **通过** | 通用 MLLM 的语义视觉表征确实可能漏掉低层、原生尺度和时间取证信号；专用取证分支有明确存在价值。 |
| 工程机制 | **通过** | 专家特征可以被压缩为视觉 token、查询 token 或证据 token，并与语言模型联合优化；这不是凭空设想。 |
| 弱监督局部学习 | **有条件通过** | 视频级标签配合 MIL 可以学到局部异常，但现有直接证据主要来自视频异常、图像 deepfake 和时序篡改定位，不能直接保证在通用 AIGC 局部编辑视频上成功。 |
| 当前数据准备度 | **未通过** | 已抽查的 Omni-Fake 视频分片没有公开局部掩码或时间边界，且 `tampered` 与 GenBuster 的 `real/full_synthetic` 来自不同生成与编码链路，存在严重来源捷径风险。 |
| 当前方法新颖性 | **未通过** | “取证专家 + MLLM”“全局/局部证据”“证据 token 注入”“分阶段 SFT/RL”均已有高度接近工作，不能作为单独创新点。 |
| 是否现在写代码训练 | **否** | 本方向只允许进入第二层“创新边界与受控数据协议设计”，尚不允许直接启动完整训练。 |

**总判断：有条件通过方向层，拒绝直接冻结现有方法。**

可继续保留的核心命题是：

> 面向非人脸限定的通用 AIGC 视频，研究如何让生成范围不同的取证证据（全生成的全局痕迹与局部编辑的稀疏痕迹）直接服务最终 Real/Fake 判定，并检验该证据是否在未见生成器上提供超过专用检测器、MLLM 和简单后融合的独立增量。

不能继续使用的宽泛命题是：

> 首次把取证专家特征或取证 token 注入 MLLM。

这个表述在 2026 年已经不成立。

## 2. 审计规则

每个前提按以下标准裁决：

- **通过**：至少有两项独立原始工作支持，其中至少一项与视频、取证或 MLLM 直接相关；且项目已有结果没有反证该命题。
- **有条件通过**：机制在邻近任务成立，但目标任务、监督形式或数据分布仍有关键差异。
- **未通过**：存在直接反证、关键数据契约不成立，或所谓创新已被近邻工作覆盖。

文献中的“方法报告有效”只用于证明可行性，不等于已经证明在本项目数据上有效。所有从邻近任务迁移到 AIGC 视频的判断均明确标为推断。

## 3. 四个核心前提

### 前提一：原生尺度和专用取证证据对 AIGC 视频检测有用

**结论：通过。**

- ICLR 2026 的 [Preserving Forgery Artifacts](https://arxiv.org/abs/2604.04634) 直接指出，固定 resize/crop 会丢失细微高频痕迹，并用可变空间分辨率和时间长度的 Qwen2.5-VL ViT 在多个 AIGC 视频基准上取得更好结果。这是对“原生尺度有检测价值”的直接视频证据。
- NeurIPS 2025 的 [Seeing What Matters](https://openreview.net/forum?id=dOGXKBL7IE) 表明，面向低层取证信号设计的增强可以改善跨生成器泛化，支持“专用取证归纳偏置不能完全由普通语义训练替代”。
- CVPR 2026 的 [FVBench](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_FVBench_Benchmarking_Deepfake_Video_Detection_Capability_of_Large_Multimodal_Models_CVPR_2026_paper.html) 同时覆盖 Real、AI-edited 和 fully generated，说明局部编辑与全生成不是单一同质 Fake 分布。

边界：这些工作支持“取证证据存在”和“生成范围不同”，但没有证明一个统一的局部/全局 token 结构必然优于专用检测器。

### 前提二：通用 MLLM 的默认视觉 token 不足以承担低层和时间取证

**结论：通过。**

- CVPR Findings 2026 的 [Beyond Static Artifacts](https://openaccess.thecvf.com/content/CVPR2026F/html/Gu_Beyond_Static_Artifacts_A_Forensic_Benchmark_for_Video_Deepfake_Reasoning_CVPRF_2026_paper.html) 发现 VLM 更容易识别静态伪影，却忽略时间不一致；专门的时间取证指令数据可改善域内及跨数据集检测。
- [ForensicsTok](https://arxiv.org/abs/2606.24538) 明确把标准 MLLM 缺少取证先验作为问题，并通过多尺度取证专家特征融合改善图像篡改定位。
- Qwen3-VL 官方[技术报告](https://arxiv.org/abs/2511.21631)证明其具备 interleaved-MRoPE、DeepStack 和文本时间对齐，因而它能承载时空输入；但报告没有提供低层媒体取证专家或相应目标。由此推断：Qwen3-VL 是可用宿主，不是现成取证器。

边界：这不等于 Qwen3-VL “没有时序能力”，也不能用 zero-shot 失败否定微调；正确结论只是默认表征缺少已验证的取证归纳偏置。

### 前提三：专家视觉证据能够被 MLLM 真正利用

**结论：工程可行性通过，作为创新点未通过。**

- ICCV 2025 的 [VideoOrion](https://openaccess.thecvf.com/content/ICCV2025/html/Feng_VideoOrion_Tokenizing_Object_Dynamics_in_Videos_ICCV_2025_paper.html) 用检测、分割、跟踪专家提取对象动态并编码为 object tokens，与上下文 token 一起送入 LLM。
- CVPR 2025 的 [Perception Tokens](https://openaccess.thecvf.com/content/CVPR2025/html/Bigverdi_Perception_Tokens_Enhance_Visual_Reasoning_in_Multimodal_Language_Models_CVPR_2025_paper.html) 将深度图和框等中间视觉表征 token 化，并证明它们可改善下游视觉推理。
- ICCV 2025 的 [D2VLM](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Factorized_Learning_for_Temporally_Grounded_Video-Language_Models_ICCV_2025_paper.html) 使用 evidence tokens 将“先定位证据、再回答”显式因子化。
- 更直接地，[ForensicsTok](https://arxiv.org/abs/2606.24538) 已将多尺度取证专家特征注入 MLLM；[VIGIL](https://arxiv.org/abs/2603.21526) 已使用分部位外部取证证据、阶段门控注入和渐进训练；[The Regularizing Power of Language-Training Deepfake Detectors](https://arxiv.org/abs/2605.31192) 已采用冻结专用检测器、LoRA MLLM、二元对齐和后续 RL。

因此，后续实验必须证明的是**一种目标任务特有的耦合机制带来的增量**，而不是再次证明“MLLM 能读取专家特征”。

### 前提四：没有掩码和配对时，视频级标签足以学习局部证据

**结论：有条件通过。**

- CVPR 2024 的 [Prompt-Enhanced MIL](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html) 证明仅用视频级标签可以学习片段级异常，但也明确指出二元 MIL 容易受异常多样性和上下文边界影响。
- CVPR 2024 的 [Contrastive MIL for DeepFake Classification and Localization](https://openaccess.thecvf.com/content/CVPR2024/html/Hong_Contrastive_Learning_for_DeepFake_Classification_and_Localization_via_Multi-Label_Ranking_CVPR_2024_paper.html) 将图像视为 patch bag，在局部人脸篡改中联合完成真假分类和局部证据学习。
- WACV 2024 的 [Weakly-Supervised Deepfake Localization](https://openaccess.thecvf.com/content/WACV2024/html/Tantaru_Weakly-Supervised_Deepfake_Localization_in_Diffusion-Generated_Images_WACV_2024_paper.html) 说明弱监督局部化可以成立，但其结果对数据集和生成器错配比对监督强弱更敏感。
- CVPR 2026 的 [TLMA](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_TLMA_Mitigating_the_Impact_of_Weakly_Labeled_Information_for_Video_CVPR_2026_paper.html) 与 [GEM-TFL](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_GEM-TFL_Bridging_Weak_and_Full_Supervision_for_Forgery_Localization_through_CVPR_2026_paper.pdf) 进一步说明：弱标签可用于时间异常/篡改定位，但必须处理正常片段污染、训练推理目标不一致和 top-k 聚合问题。

这些证据只支持“值得做受控最小验证”。它们没有直接证明无配对、无掩码的通用 AIGC 局部编辑视频可以稳定学出空间和时间证据。

## 4. 最近工作冲突

| 最近工作 | 已覆盖内容 | 本项目不能再主张 |
|---|---|---|
| [ForensicsTok](https://arxiv.org/abs/2606.24538)，arXiv 2026 | 多尺度取证专家注入 MLLM；取证 token；篡改定位 | 首次取证专家 token 化或首次专家特征注入 MLLM |
| [VIGIL](https://arxiv.org/abs/2603.21526)，arXiv 2026 | 局部部位规划、外部证据注入、阶段门控、SFT/RL | 首次局部证据驱动 MLLM 推理或首次阶段式取证训练 |
| [The Regularizing Power of Language-Training Deepfake Detectors](https://arxiv.org/abs/2605.31192)，arXiv 2026 | 冻结专用检测器 + LoRA MLLM + 二元对齐 + RL | 首次专用检测器与 MLLM 对齐，或首次仅靠二元标签做解释性 RL |
| [VLAForge](https://arxiv.org/abs/2603.24454)，CVPR 2026 | 视频人脸 deepfake 的细粒度/全局取证查询、局部图和 VLM 语义耦合 | 首次全局与局部取证分支增强 VLM 视频检测 |
| [VideoOrion](https://openaccess.thecvf.com/content/ICCV2025/html/Feng_VideoOrion_Tokenizing_Object_Dynamics_in_Videos_ICCV_2025_paper.html)，ICCV 2025 | 专家检测-分割-跟踪结果编码为视频 object tokens | 首次把视频专家信息编码成 LLM 可读 token |

剩余可能的论文空间不是一个模块名，而是一个尚待验证的组合问题：

1. 非人脸限定、面向通用 T2V/I2V/局部编辑的 AIGC 视频；
2. 同一最终 Real/Fake 目标下，按生成范围分配全局与稀疏局部证据；
3. 局部编辑分支主要依赖弱监督，而非完整 mask；
4. 在未见生成器和统一编码控制下，端到端证据耦合稳定超过专用专家、MLLM 和简单后融合；
5. 用随机、打乱、置零证据 token 证明增量确实来自逐样本视觉证据。

这五点必须在第二层近邻工作审计中共同成立，才能形成独立方法。

## 5. 当前数据与模型是否支持

### 5.1 Omni-Fake 本地发布审计

已本地检查：

- `E:\newgaibeishi\train-00000-of-00120.parquet`
- `E:\newgaibeishi\test-00000-of-00040.parquet`

两个分片各 500 条，分别为 `PartialEdit` 和 `PartialEdit_OOD` 的 `tampered` 视频。可见字段只有 `video`、`label`、`generator`、`filename`、`split`；没有 pair ID、空间 mask、bbox 或时间边界。分片视频均为约 5.04 秒、1024x1024、HEVC/`hev1`。两个抽查分片之间未发现文件名或字节级重复。

因此当前只能确认：

- 纯视频子集可用于视频级 `tampered` 监督；不需要音频，也不需要配对。
- 论文声称的 joint detection-localization-explanation 协议不能被直接等同为当前 parquet 已公开定位标注；至少这两个视频分片没有。
- Omni-Fake 的 `real/full_synthetic` 来源与 `PartialEdit` 链路不同。在统一重编码、分辨率/时长匹配、来源分组和 real-vs-real 控制完成前，三分类或 scope head 很可能学习 codec、来源或生成流水线，而不是生成范围。该风险也符合 [VidAudit](https://arxiv.org/abs/2606.31004) 对视频检测捷径的警告。

### 5.2 数据分工的当前上限

| 数据 | 现在可以承担 | 现在不能承担 |
|---|---|---|
| GenBuster / DataB | Real/Fake 主训练、全生成证据、生成器分组 | 不能提供局部编辑位置真值 |
| Omni-Fake 视频 | `tampered` 视频级弱监督、OOD 局部编辑测试 | 不能默认具有 mask；未经去偏不能直接训练 scope 分类器 |
| DataA | 已知局部编辑区域的开发诊断、MIL 局部证据验收 | 自动 CoT 不作为可靠真值；不代表通用全生成分布 |
| ViF-Bench / GenBuster benchmark | 最终 Real/Fake 外部评测与逐生成器分析 | 不参与调阈值或反复选择方法后再称为完全未见测试 |

### 5.3 Qwen3-VL 的适配判断

Qwen3-VL-8B 可以作为 MLLM 宿主，但不能再次采用“独立学一个辅助能力，期待它自动迁移到 detection”的训练逻辑。取证证据必须进入由 Real/Fake 主损失直接监督的融合路径，并至少比较：

1. 专用取证专家单独；
2. Qwen3-VL 单独；
3. 两者分数后融合；
4. 专家证据 token 融合；
5. 随机、样本间打乱、置零证据 token。

如果 token 融合不能超过专家单独和简单后融合，就不能声称 MLLM 学会了证据推理。

## 6. 进入第二层前的硬条件

第二层只做“精确方法与数据协议设计”，满足以下条件后才允许写完整训练代码：

1. 写出一句不与 ForensicsTok、VIGIL、VLAForge 和语言正则化工作重叠的可证伪贡献。
2. 明确主指标始终是外部数据上的 Real/Fake ACC、Balanced ACC、F1/AUROC；定位与解释是辅助验收。
3. 完成 codec、分辨率、时长、来源和生成器分组协议，保证 `real/full_synthetic/tampered` 不可被元数据捷径轻易区分。
4. 明确变长有序帧合同，不默认每个样本 16 帧；原生尺度与统一 resize 必须是受控消融。
5. 确定最小训练信号：主二分类损失必须直接约束融合结果；MIL 只负责候选局部证据，不单独决定真假。
6. 预先锁定专家单独、MLLM 单独、后融合、token 融合和 token 因果控制五组基线。

## 7. 停止条件

出现任一情况就停止该版本，不用靠增加 CoT、DPO 或 GRPO 挽救：

- 第二层检索发现“通用 AIGC 视频 + scope-aware 全局/局部弱监督证据 + MLLM 直接检测耦合”已被近邻工作完整覆盖；
- 统一编码后，`tampered` 与 `real/full_synthetic` 的可分性大幅消失，说明主要信号来自数据来源；
- MIL 局部分数在 DataA 留出 mask 上不优于随机/全局热图；
- token 融合不优于专用专家或简单后融合；
- 提升只存在于一个已反复查看的 benchmark，未见生成器或第二外部集不保持方向；
- Real recall 明显下降，所谓提升只来自过度预测 Fake。

## 8. 下一步

下一步不是训练，而是进行第二层审计，输出一份**精确方法蓝图**：

1. 对最接近的五项工作做结构级对照，找到尚未覆盖且与现有数据匹配的唯一贡献；
2. 决定 scope 是模型内部的软门控变量，还是只作为损失/采样条件，避免被来源捷径监督；
3. 锁定专用取证专家、token 形成方式、注入层、主损失和最小因果消融；
4. 给出只需一轮小规模训练即可否定整条路线的 Gate 0/1。

在这份第二层蓝图通过前，不重新标大规模 CoT，不下载新的超大模型，不启动完整 Qwen3-VL SFT 或 RL。
