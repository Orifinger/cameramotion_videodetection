# 通用 AIGC 视频取证：开源数据支持审计

更新时间：2026-07-22  
审计问题：**不依赖邮件审批或作者未来发布，现有可下载数据是否足以支撑一篇以视频 Real/Fake 为主指标、同时利用全生成与局部编辑证据的 MLLM 方法论文？**

## 1. 裁决

| 能力层 | 裁决 | 依据 |
|---|---|---|
| 全生成视频 Real/Fake 训练 | **通过** | GenBuster-200K 已公开；项目也已有可用 DataB。GenVidBench 可作为可选扩展。 |
| 现代生成式局部编辑训练 | **有条件通过** | OpenVE-3M 已公开原视频、编辑视频和编辑指令三元组，并含 Local Change/Remove/Add；但没有现成取证 mask，且需先验证分片获取和配对质量。 |
| 时间定位监督 | **通过** | ActivityForensics 已公开 5,389 条视频及篡改时间段；ViF-CoT-4K 也包含时间证据。 |
| 通用空间定位监督 | **有条件通过** | ViF-CoT-4K 和项目 DataA 可提供小规模框/区域监督；OpenVE 可由对齐 pair 构造弱掩码，但没有大规模公开真值 mask。 |
| 高质量解释性 CoT | **不足** | ViF-CoT-4K 可作小规模辅助，其他主力数据没有可靠解释真值；不能把 CoT 设为主要监督来源。 |
| 多生成器外部评测 | **通过** | ViF-Bench、MintVid、GenBuster benchmark 已可用；Omni-Fake-OOD 可作补充，但存在类别和来源偏置。 |
| 完全依赖当前公开数据完成论文 | **有条件通过** | 前提是方法不要求每个训练视频都有 dense mask 或人工 CoT，并通过 OpenVE 配对可用性与编码捷径审计。 |

**总裁决：现有开源数据足以支持“视频特有”的方法论文，但不支持大规模全监督时空定位论文。**

图像方法 ForensicsTok、VIGIL 只能证明“取证证据可注入 MLLM”已有先例；视频仍有三个实质空间：

1. 同时处理全生成的全局证据与局部编辑的稀疏证据；
2. 用有序视频证据而非独立图像证据完成最终 Real/Fake；
3. 在没有大规模 mask/CoT 时，利用原视频与编辑视频配对学习局部证据。

但论文贡献不能表述为“首次将取证 token 注入 MLLM”，而应落到**视频生成范围与时间证据如何受最终二分类目标约束**。

## 2. 实际可用数据矩阵

“公开”在本表中表示当前已有直接下载入口；要求填写许可、发邮件并等待审批的不算立即可用。

| 数据 | 当前发布状态 | 可用监督 | 建议职责 | 关键限制 |
|---|---|---|---|---|
| [GenBuster-200K](https://huggingface.co/datasets/l8cv/GenBuster-200K) | **直接可下**，约 114 GB，MIT | 视频级 Real/Fake、生成器 | 全生成主训练；closed benchmark 外测 | 无局部编辑、时间段、空间位置和解释 |
| [GenVidBench](https://github.com/genvidbench/GenVidBench) | **已发布** HF 入口，仓库声明 6.78M；部分来源需 VidProM | 视频级 Real/Fake、生成器/来源 | 可选的大规模全生成预训练或未见生成器扩展 | 规模巨大；应先核对实际文件与许可，不作为第一轮依赖 |
| [OpenVE-3M](https://huggingface.co/datasets/Lewandofski/OpenVE-3M) | **直接可下**，HF 约 2.23 TB，CC BY-NC 4.0 | `original_video`、`video`、编辑 `prompt`；Local Change/Remove/Add 等类别 | **局部编辑主训练源**：pair contrast、弱局部证据、编辑范围先验 | 没有取证 mask；必须验证原/编辑时间对齐、重编码捷径和按样本获取成本 |
| [ActivityForensics](https://huggingface.co/datasets/ActivityForensics/ActivityForensics) | **直接可下**，5,389 条，约 120 GB；研究非商用 | 视频与篡改时间段，add/delete 类型 | 时间证据分支训练/验收；局部编辑外部诊断 | 活动级片段编辑，不提供通用空间 mask，也不是全生成检测集 |
| [ViF-CoT-4K](https://huggingface.co/datasets/JoeLeelyf/ViF-CoT-4K) | **直接可下**，19.7 GB，CC BY 4.0 | Real/Fake、伪影类型、时间、bbox、解释、语义配对 | 小规模 grounding/解释辅助和框级验收 | 约 4K；不能承担全部检测训练，也不能代表所有局部编辑 |
| [ViF-Bench](https://huggingface.co/datasets/JoeLeelyf/ViF-Bench) | **直接可下**，8.83 GB | Real/Fake、生成器 | 主要外部 Real/Fake 测试之一 | 已被本项目反复查看，后续须另保留第二外部集 |
| [MintVid / VideoVeritas](https://github.com/EricTan7/VideoVeritas) | **ModelScope 已发布**，约 3K、9 个生成器 | 视频级 Real/Fake/内容分组 | 独立外部测试 | 无公开 dense 定位真值；规模较小 |
| [VideoSham](https://github.com/adobe-research/VideoSham-dataset) | **直接可下**，352 real + 352 paired edit | 原/编辑 pair、编辑类型、开始/结束时间、描述 | 配对与时间定位的小型控制集 | 专业传统编辑，不是现代扩散式 AIGC 主分布 |
| [ForgeryNet](https://yinanhe.github.io/projects/forgerynet.html) | **有公开下载入口**，221,247 视频 | 视频真假与时间段；图像有空间 mask | 可选的时间定位预训练/负控制 | 人脸限定且较旧，不能支撑通用 AIGC 主张 |
| [Omni-Fake-SET](https://huggingface.co/datasets/JamalLee/Omni-Fake-SET) | **直接可下**，视频 260K，CC BY 4.0 | 视频级 `real/full_synthetic/tampered` | 补充三类测试或去偏后的弱训练 | 视频 parquet 无 mask/时间段/pair；三类来自不同数据流水线，来源捷径风险高 |
| [Omni-Fake-OOD](https://huggingface.co/datasets/JamalLee/Omni-Fake-OOD) | **直接可下**，视频 22K | 同上；1K real、1K full、20K tampered | 补充 OOD 测试 | 极不平衡，且 full/real 与 tampered 来源不同，不能单独作为主结论 |
| [CoCoVideo-26K](https://github.com/DonoToT/CoCoVideo) | **需许可 PDF + 邮件审批** | 商业模型、语义对齐 real/fake pair | 可并行申请的高质量外测 | 不可作为项目时间线的必要依赖 |
| [FVBench](https://github.com/IntMeGroup/FVBench) | **当前不可用**：仓库仅一条简短 README，无数据/代码 release | 论文声称覆盖 Real/AI-edited/full | 暂不纳入 | 论文“available”不等于当前实际可下载 |

项目自有 DataA 仍有价值，但它是 **VACE 单一编辑流水线上的开发/验收集**，不应作为通用结论的唯一测试；DataB 则继续只承担全生成视频训练，不能伪装成局部监督。

## 3. 数据能支撑的最成熟任务定义

> **面向通用 AIGC 视频的生成范围感知取证检测：利用全生成视频的全局监督和原视频/局部编辑视频 pair 的弱局部监督，学习直接受 Real/Fake 主损失约束的视频证据 token；定位与解释用于验证证据是否合理，不取代最终二分类指标。**

推荐的数据分工如下：

| 模块 | 主数据 | 学习信号 |
|---|---|---|
| 全局生成证据 | GenBuster/DataB；可选 GenVidBench 子集 | 视频级 Real/Fake、生成器分组 |
| 局部编辑证据 | OpenVE 的 Local Change/Remove/Add 子集 | 同内容原/编辑 pair 对比；对齐差分产生弱区域，不把伪 mask 当真值 |
| 时间证据 | ActivityForensics | 篡改区间与视频级真假联合约束 |
| 小规模 grounding | ViF-CoT-4K + DataA | bbox/时间命中率，只作为辅助验收和少量监督 |
| 最终外部评测 | ViF-Bench + MintVid + GenBuster benchmark | Real/Fake ACC、Balanced ACC、F1、AUROC；逐生成器报告 |
| 补充 scope/OOD | Omni-Fake-OOD | 三类与二类结果均报告，并明确来源/不平衡限制 |

这个组合不需要重新自动标注大规模 CoT，也不需要让所有数据共享同一种标注格式。统一的是模型最终目标和证据接口，不是强行统一原始监督。

## 4. 现有数据明确不支持什么

以下主张现在不能做：

- “在大规模通用 AIGC 视频上进行全监督空间 mask + 时间段联合定位”；
- “所有训练视频都有高质量、可验证的解释性 CoT”；
- “直接把 Omni-Fake 三类混合训练即可学到生成范围”；
- “仅在 DataA 或反复查看过的 ViF-Bench 提升即可证明泛化”；
- “OpenVE 的原/编辑像素差天然就是篡改真值 mask”。

因此 dense localization 和自然语言解释只能是辅助任务/定性结果，论文主指标仍是未见生成器上的 Real/Fake。

## 5. 写模型代码前唯一必要的数据 Gate

先对 OpenVE 做 **500 对局部编辑可用性审计**，不训练 Qwen3-VL：

1. 从 Local Change、Local Remove、Local Add 各取可访问样本，确认原视频、编辑视频和 prompt 能逐条配对；
2. 检查帧数、FPS、分辨率、时长和时间对齐，统计有效 pair 比例；
3. 将原/编辑双方统一重编码后，再测简单 metadata/codec 分类器，避免把编码链路当作 Fake；
4. 计算配准后的差分集中度，并人工抽查 100 对，确认变化以局部为主而非全帧重绘；
5. 按原始视频、编辑类别和生成流水线分组，保证后续 train/test 不共享 source；
6. 同时验证 ActivityForensics 的时间区间可以稳定解析并映射到所用帧序列。

通过条件不是追求一个漂亮检测分数，而是确认：**pair 真存在、编辑大体局部、弱证据不是编码捷径、划分可以防泄漏。**

- Gate 通过：进入“全局视频证据 + paired 局部证据 + 最终 Real/Fake 直接耦合”的方法设计。
- Gate 未通过：停止空间局部化主张，保留全生成检测与 ActivityForensics 时间证据路线；不要靠自动 CoT 或 RL 掩盖数据契约失败。

## 6. 最终建议

当前不需要完全重造数据，也不应该把 Omni-Fake 当作唯一救命数据。最有价值的新信息是 **OpenVE-3M 提供了大规模原/编辑 pair**，它正好补上此前 DataA 单生成器、规模小的问题；ActivityForensics 则补上独立时间定位验收。

所以数据层结论是：

> **可以继续做视频，而且比照搬 ForensicsTok/VIGIL 到图像更有空间；但应把局部 pair 弱监督和时间证据作为视频特有的核心，把 dense mask 与 CoT 降为辅助。方法是否最终成立，取决于 OpenVE 的 500-pair Gate，而不是再做一次大规模 SFT。**

