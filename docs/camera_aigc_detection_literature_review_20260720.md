# 相机条件化 AIGC 视频检测：文献调研与方案收敛

更新日期：2026-07-20  
调研范围：以 2024--2026 年 AIGC 视频检测、视频 MLLM、相机运动/位姿、多任务优化和条件异常检测工作为主，辅以少量直接支撑方法的经典论文。  
文献状态：正式会议/期刊与 arXiv 预印本分开标注；本文不把预印本写成已录用论文。

## 1. 先给结论

当前最成熟、最可执行、也最能解释已有实验结果的主线，不是继续把 camera labels/caption 拼进 prompt，也不是继续训练独立 Camera VQA，更不是把估计出的全局相机运动从光流里硬减掉，而是：

> **相机条件化时序正常性专家（Camera-Conditioned Temporal Normality Expert, CTNE）+ Qwen3-VL 检测分支的受控概率融合。**

它把相机运动定义为“决定正常视频运动分布的上下文变量”，而不是直接真假证据。模型学习真实视频在给定相机轨迹下的时序表征分布 `p(real temporal evidence | camera trajectory)`；当物体、纹理或语义轨迹与当前相机运动不相容时，条件似然下降，形成生成视频异常分数。随后只用一个小型、正则化的校准器把这个分数与原 Qwen3-VL 检测模型的真假 logit 融合。

这条路线的关键优点是：

1. **相机变量与最终 Real/Fake 分数直接耦合。** 不再依赖“先学会 Camera VQA，能力也许会自行迁移”的弱假设。
2. **不把 camera motion 当作 Fake 证据。** 模型估计的是 `p(y | c, real)`，不建模 `p(c)`；静止、摇摄或复杂运动本身都不应被判假。
3. **与现有失败实验不是同一方法。** 已失败的相机补偿残差使用 `observed motion - estimated camera motion`；CTNE 保留原始交互，只让正常性边界随 camera context 改变。
4. **第一门很便宜且可证伪。** 先比较正确 camera、打乱 camera 和无 camera 条件；若正确条件不能稳定超过两者，就停止 camera 主线，不再烧 Qwen3-VL 全量训练。
5. **与 Qwen3-VL 的分工清楚。** 小专家处理 MLLM 不擅长的低层时序/几何统计，Qwen3-VL 保留语义判断与解释能力；这比让一个 8B 模型同时承担所有取证任务更稳。

这不是“保证提点”的数学证明。可证明的是条件建模在上下文解释了正常变化时能够降低类内方差；camera 是否确实提供可用增量，必须由正确/打乱/无条件三组对照决定。

## 2. 文献证据如何解释现有结果

| 已有项目结果 | 文献给出的解释 | 对方案的约束 |
|---|---|---|
| 正确 Camera VQA 可学，且 opposite/no-frame 控制明显下降 | CameraBench 与 Cambrian-P 都说明 camera 能力可以注入；但 Cambrian-P 的收益来自内部 pose supervision 改造表征，不是把 camera 文本放到推理输入 | “camera 可学”不能替代“camera 改善检测”的独立验收 |
| Camera VQA + detection replay 仍低于 detection-only | MMPareto、VideoVeritas 都报告辅助任务与主任务存在梯度/长度/难度冲突，简单 batch mixing 不保证正迁移 | 不再把独立辅助任务的最终分数当作主方法；先测直接条件关系 |
| labels+caption 训练和推理严格一致仍使 ViF-Bench 下降 | CameraBench 发现 VLM 对精细几何 primitive 较弱；Qwen3-VL 没有显式 SE(3) 状态。文本描述进入语言 token，并不等价于几何表征 | 主方法不向 user prompt 注入 camera 文本 |
| 正确路由低于无路由/错误路由 | 硬路由只按 motion bucket 选择模型，不能表达“同一相机运动下什么时序变化才正常” | camera 必须进入连续条件分布或连续门控，不能只作三分类开关 |
| RAFT/DINO 正确几何残差低于 raw flow 和 wrong geometry | 生成异常不等于纯前景运动；硬补偿会同时删除有用的相机-物体耦合、视差、遮挡和材质变化 | 不做确定性相减；改为条件概率建模 |
| DataA 局部编辑与 DataB 全生成混训伤害 ViF | FVBench、ActivityForensics 和 Omni-Fake 都把局部编辑与全生成视作不同问题；VidAudit 强调来源/长度/codec 等混杂控制 | DataA 用作局部定位与解释开发集，不作为通用真假主训练分布 |
| 原检测模型本身已有较强 ViF 指标 | CoCoDetect 与 VidAudit 支持保留强基线，再融合正交低层专家，而不是反复全量覆盖式续训 | 第一版冻结 Qwen 检测模型，只训练小专家和校准器 |

### 2.1 项目实测锚点

下表只用于约束新方案，不替代 `docs/camera_conditioned_experiment_log.md` 中的完整实验记录。不同历史 prompt 的 ViF 数值不能互相当作严格控制；这里优先列同一实验内部的受控比较。

| 受控比较 | 关键结果 | 已建立的结论 |
|---|---|---|
| DataB 显式 labels+caption，训练与 ViF 推理均提供同格式 camera context | no-camera Balanced ACC 约 79.09%，with-camera 约 76.42%；camera 仅在 2/19 个生成器子集获胜 | 不是“推理漏给 camera”；显式文本条件本身没有带来检测增量 |
| 等量 detection replay / 正确 Camera VQA / 翻转 Camera VQA | Camera Macro AP 为 71.80% / 86.28% / 40.15%；正确监督的 opposite/no-frame 控制明显下降 | camera 能力可学且依赖画面，不是固定答案先验 |
| 同三分支在无 camera 文本检测任务上 | DataA Balanced ACC 63.43% / 60.03% / 60.03%；ViF 为 77.18% / 76.81% / 77.22% | 学到 camera 不等于迁移到 detection；正确标签没有形成可辨别的正迁移 |
| ViF 三分类 hard route | 原模型 79.18%，shared 76.30%，正确 route 74.50%，错误 route 78.03% | 粗 camera bucket 不是可靠路由依据，且“正确 route”反而最差 |
| DINO/RAFT 几何 residual gate | appearance AUROC/BAcc 56.20/53.60；raw motion 59.13/58.51；correct geometry 56.88/55.21；wrong geometry 58.35/55.44 | 正确几何相减没有超过 raw 或 wrong control，不能继续把硬补偿当主方法 |

## 3. 分类参考文献

优先级说明：`P0` 直接决定当前方法；`P1` 可直接借鉴模块、训练或评测；`P2` 用于背景、对照或扩展。

### 3.1 AIGC 视频检测、基准与解释

| 编号 | 优先级 | 论文与状态 | 与本项目直接相关的结论 |
|---|---|---|---|
| A01 | P0 | [VideoVeritas: AI-Generated Video Detection via Perception Pretext Reinforcement Learning](https://arxiv.org/abs/2602.08828), arXiv 2026 | Qwen3-VL 上的 perception pretext RL；作者报告 phase-level 比直接 batch integration 更有效，粗糙 artifact grounding 还会退化。说明辅助能力必须经过主任务验收。 |
| A02 | P0 | [Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Skyra_AI-Generated_Video_Detection_via_Grounded_Artifact_Reasoning_CVPR_2026_paper.html), CVPR 2026 | 以时空定位 CoT 支撑检测和解释，是本项目 detection checkpoint 与 ViF-Bench 的最近邻工作。 |
| A03 | P0 | [CoCoVideo: The High-Quality Commercial-Model-Based Contrastive Benchmark for AI-Generated Video Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_CoCoVideo_The_High-Quality_Commercial-Model-Based_Contrastive_Benchmark_for_AI-Generated_Video_Detection_CVPR_2026_paper.html), CVPR 2026 | CoCoDetect 用 R3D-18 时空专家处理大多数样本，仅将低置信样本送给 MLLM；支持“专门检测器 + MLLM”的职责分离。 |
| A04 | P0 | [Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods](https://openaccess.thecvf.com/content/CVPR2026/html/Hayun_Training-free_Detection_of_Generated_Videos_via_Spatial-Temporal_Likelihoods_CVPR_2026_paper.html), CVPR 2026 | STALL 用真实数据统计构造空间-时间似然，支持不用追逐每个 fake generator、转而学习 real normality。 |
| A05 | P0 | [Auditing Generalization in AI-Generated Video Detection: A Six-Control Protocol and the VidAudit Toolkit](https://arxiv.org/abs/2606.31004), arXiv 2026 | 长度三特征可产生 0.998 的虚假 LOGO AUC；规范重编码、泄漏审计、real-vs-real、同协议、置信区间和跨数据集六项控制必须进入正式评测。 |
| A06 | P0 | [AI-Generated Video Detection via Perceptual Straightening](https://arxiv.org/abs/2507.00583), NeurIPS 2025 | ReStraV 用 DINOv2 表征轨迹的曲率和步长检测生成视频，证明冻结视觉表征的时序几何可形成轻量专家。 |
| A07 | P0 | [D3: Training-Free AI-Generated Video Detection Using Second-Order Features](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_D3_Training-Free_AI-Generated_Video_Detection_Using_Second-Order_Features_ICCV_2025_paper.html), ICCV 2025 | 二阶中心差分比一阶运动更有跨生成器潜力；适合作为 CTNE 的内容特征，而不是 camera condition。 |
| A08 | P0 | [Training-free Detection of Text-to-video Generations via Over-coherence](https://openaccess.thecvf.com/content/WACV2026/html/Brokman_Training-free_Detection_of_Text-to-video_Generations_via_Over-coherence_WACV_2026_paper.html), WACV 2026 | 生成视频可能呈现异常的时间“过度一致”，支持使用全局自相似/轨迹统计而不是只找局部抖动。 |
| A09 | P0 | [Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](https://papers.neurips.cc/paper_files/paper/2025/hash/ff7c9c90030f1eb3cbf5c81b6fbd9a05-Abstract-Conference.html), NeurIPS 2025 | NSG-VD 直接建模自然时空动力学与分布偏移，支持“物理/正常性专家”路线。 |
| A10 | P1 | [Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale](https://arxiv.org/abs/2604.04634), ICLR 2026 camera-ready | 固定 resize/crop 会丢失高频伪造痕迹；最终系统需报告 native-scale 或至少做分辨率控制。 |
| A11 | P1 | [CAM-VFD: Cross-Attention Multimodal Video Forgery Detection](https://arxiv.org/abs/2605.17133), arXiv 2026 | 通过 appearance、motion、depth 的 cross-attention 建模跨模态矛盾；支持“关系”优于单独 cue 或硬相减，但目前仅预印本。 |
| A12 | P1 | [CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection](https://arxiv.org/abs/2605.00630), arXiv 2026 | 研究视觉-文本对齐的时间稳定性；可作为后续跨 substrate 特征，但 caption 生成成本与偏差需单独控制。 |
| A13 | P1 | [ATSS: Detecting AI-Generated Videos via Anomalous Temporal Self-Similarity](https://arxiv.org/abs/2604.04029), arXiv 2026 | 用视觉、文本和跨模态自相似矩阵刻画 anchor-driven 轨迹；支持 CTNE 纳入 self-similarity 特征。 |
| A14 | P1 | [Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation](https://openreview.net/forum?id=dOGXKBL7IE), NeurIPS 2025 | 强调针对取证信号设计增强，而非让模型依赖内容/数据集捷径。 |
| A15 | P1 | [UNITE: Towards a Universal Synthetic Video Detector from Face or Background](https://openaccess.thecvf.com/content/CVPR2025/html/Kundu_Towards_a_Universal_Synthetic_Video_Detector_From_Face_or_Background_CVPR_2025_paper.html), CVPR 2025 | 使用通用视觉表征并抑制注意力集中到单一区域，说明检测不应只依赖人脸或背景。 |
| A16 | P1 | [Your One-Stop Solution for AI-Generated Video Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Your_One-Stop_Solution_for_AI-Generated_Video_Detection_CVPR_2026_paper.html), CVPR 2026 | AIGVDBench 覆盖 31 个生成模型和 44 万余视频，强调跨模型、跨质量与评测协议的重要性。 |
| A17 | P1 | [FVBench: Benchmarking Deepfake Video Detection Capability of Large Multimodal Models](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_FVBench_Benchmarking_Deepfake_Video_Detection_Capability_of_Large_Multimodal_Models_CVPR_2026_paper.html), CVPR 2026 | 同时含 real、AI-edited 和 fully generated，证明局部编辑与全生成不能用一个模糊 fake 分布解释。 |
| A18 | P1 | [ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](https://openaccess.thecvf.com/content/CVPR2026/html/Bao_ActivityForensics_A_Comprehensive_Benchmark_for_Localizing_Manipulated_Activity_in_Videos_CVPR_2026_paper.html), CVPR 2026 | 直接支撑 DataA 更适合做局部编辑定位/解释评测，而不是替代通用全生成 benchmark。 |
| A19 | P1 | [Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Omni-Fake_Benchmarking_Unified_Multimodal_Social_Media_Deepfake_Detection_CVPR_2026_paper.html), CVPR 2026 | 明确区分 real、partially manipulated、fully synthetic，并联合定位和解释；可借鉴任务定义。 |
| A20 | P1 | [Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](https://openaccess.thecvf.com/content/CVPR2026F/html/Gu_Beyond_Static_Artifacts_A_Forensic_Benchmark_for_Video_Deepfake_Reasoning_CVPRF_2026_paper.html), CVPR Findings 2026 | 指出 VLM 易依赖静态伪影而忽视时间异常，支持保留独立时序专家。 |
| A21 | P1 | [From Detector Evidence to Language: Explainable Deepfake Video Detection](https://openaccess.thecvf.com/content/CVPR2026W/FoundGen-Bio/html/Panahi_From_Detector_Evidence_to_Language_Explainable_Deepfake_Video_Detection_CVPRW_2026_paper.html), CVPR Workshop 2026 | 先由短时间窗 detector 定位证据，再让 VLM 解释；这比要求 CoT 自己发现低层异常更可靠。 |
| A22 | P1 | [GenVidBench: A Challenging Benchmark for Detecting AI-Generated Video](https://arxiv.org/abs/2501.11340), arXiv 2025 | 大规模生成视频基准；同时也是 VidAudit 揭示 clip-length/来源偏置的案例，使用时必须补控制。 |
| A23 | P1 | [BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation](https://arxiv.org/abs/2505.12620), arXiv 2025 | MLLM 检测与解释路线的重要基线；说明语言解释可以是输出层，但不自动等于可靠取证证据。 |
| A24 | P1 | [BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM](https://arxiv.org/abs/2507.14632), arXiv 2025 | 纯 RL/统一检测的重要比较对象；需关注其浅层语义捷径与跨数据集稳定性。 |
| A25 | P1 | [VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](https://openreview.net/forum?id=gXjOsBcXIR), ICLR 2026 | RL 驱动检测与解释；适合作为后续 RL 对照，不应在没有可验证 camera-detection reward 前直接照搬。 |
| A26 | P2 | [DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark](https://arxiv.org/abs/2405.19707), arXiv 2024 | 早期大规模 AIGV detection 和时空建模代表，适合作为历史基线。 |
| A27 | P2 | [Detecting AI-Generated Video via Frame Consistency](https://arxiv.org/abs/2402.02085), arXiv 2024 | 从帧间一致性切入，说明时间 cue 是长期主线；但早期生成模型上的结论需在现代 benchmark 复核。 |
| A28 | P2 | [AIGVDet: A Benchmark for AI-Generated Video Detection](https://arxiv.org/abs/2403.16638), arXiv 2024 | 早期 AIGV 检测数据与基线，主要用于领域演进背景。 |
| A29 | P2 | [Beyond Deepfake Images: Detecting AI-Generated Videos](https://openaccess.thecvf.com/content/CVPR2024W/WMF/html/Vahdati_Beyond_Deepfake_Images_Detecting_AI-Generated_Videos_CVPRW_2024_paper.html), CVPR Workshop 2024 | 展示图像 detector 直接迁移到视频的局限，支持专门时间建模。 |
| A30 | P2 | [Can Multimodal Large Language Models Work as Deepfake Detectors?](https://arxiv.org/abs/2503.20084), arXiv 2025 | 系统考察通用 MLLM 的 deepfake 能力；可用于说明 zero-shot 差不代表微调路线不成立，但也不能证明辅助任务会迁移。 |

### 3.2 相机运动、位姿与视频 MLLM

| 编号 | 优先级 | 论文与状态 | 与本项目直接相关的结论 |
|---|---|---|---|
| B01 | P0 | [Towards Understanding Camera Motions in Any Video](https://arxiv.org/abs/2504.15376), arXiv 2025 | CameraBench 发现 SfM 弱于语义 primitive、VLM 弱于精确几何 primitive；单一 labels/caption 不是完整 camera state。 |
| B02 | P0 | [Cambrian-P: Pose-Grounded Video Understanding](https://arxiv.org/abs/2605.22819), arXiv 2026 | 每帧 pose token + pose head + 联合 loss；收益来自训练期 pose supervision，推理去掉 pose token 基本不掉点。它是未来“内部化 camera 表征”的最强参考。 |
| B03 | P0 | [Geometry-Guided Camera Motion Understanding in VideoLLMs](https://openaccess.thecvf.com/content/CVPR2026W/PVUW/html/Feng_Geometry-Guided_Camera_Motion_Understanding_in_VideoLLMs_CVPRW_2026_paper.html), CVPR Workshop 2026 | probing 显示深层 Qwen2.5-VL ViT 中 camera cue 较弱；3D foundation model 提取几何再结构化注入可改善 camera QA，但尚未证明检测迁移。 |
| B04 | P1 | [CaMo: Camera Motion Understanding through Parameter Prediction and Motion Simulation](https://arxiv.org/abs/2605.20165), arXiv 2026 | 将 camera motion 拆成连续参数/轨迹而不是只输出类别；支持 CTNE 使用连续 camera context。 |
| B05 | P1 | [Seeing without Pixels: Perception from Camera Trajectories](https://openaccess.thecvf.com/content/CVPR2026/html/Xue_Seeing_without_Pixels_Perception_from_Camera_Trajectories_CVPR_2026_paper.html), CVPR 2026 | camera trajectory 单独也携带场景/行为信息；既支持其作为上下文，也警告 camera-only shortcut 必须审计。 |
| B06 | P1 | [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651), CVPR 2025 | 可一次前向估计多帧 camera、depth、point map；若 RAFT 条件门通过，可升级为更稳定的 pseudo-pose 来源。 |
| B07 | P1 | [CamFlow: Estimating 2D Camera Motion with Hybrid Motion Basis](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Estimating_2D_Camera_Motion_with_Hybrid_Motion_Basis_ICCV_2025_paper.html), ICCV 2025 | 专门估计 2D camera motion，比单 homography 更灵活；是连续 condition extractor 的候选替换件。 |
| B08 | P1 | [Segment Any Motion in Videos](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Segment_Any_Motion_in_Videos_CVPR_2025_paper.html), CVPR 2025 | 将全局与局部运动分离并做 motion segmentation；适合后续局部异常定位，但不是第一轮二分类门。 |
| B09 | P1 | [VideoOrion: Tokenizing Object Dynamics in Videos](https://openaccess.thecvf.com/content/ICCV2025/html/Feng_VideoOrion_Tokenizing_Object_Dynamics_in_Videos_ICCV_2025_paper.html), ICCV 2025 | 显式动态 token 比纯语言描述更接近模型内部表征；支持后续 anomaly token/trajectory token。 |
| B10 | P1 | [Efficient Motion-Aware Video MLLM](https://arxiv.org/abs/2503.13016), arXiv 2025 | 将运动线索以轻量方式接入 Video MLLM，说明无需全面重训大模型即可增加 motion branch。 |
| B11 | P1 | [ReMoRa: Multimodal Large Language Model based on Refined Motion Representation](https://openaccess.thecvf.com/content/CVPR2026/html/Yashima_ReMoRa_Multimodal_Large_Language_Model_based_on_Refined_Motion_Representation_CVPR_2026_paper.html), CVPR 2026 | 直接优化 motion representation，而不是只添加 motion 文本；可作为第二阶段内部融合参考。 |
| B12 | P1 | [Flow4Agent: Long-Horizon Motion Understanding with Optical Flow](https://arxiv.org/abs/2510.05836), arXiv 2025 | 光流作为外部运动 substrate 接入 MLLM 的参考，但本项目必须使用正确/打乱 camera 条件排除只靠 raw flow。 |
| B13 | P1 | [VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](https://openaccess.thecvf.com/content/CVPR2025/html/Munasinghe_VideoGLaMM__A_Large_Multimodal_Model_for_Pixel-Level_Visual_Grounding_CVPR_2025_paper.html), CVPR 2025 | 视频像素级 grounding 说明解释可以由 detector evidence 引导，而不必完全依赖自由 CoT。 |
| B14 | P1 | [VTimeLLM: Empower LLM to Grasp Video Moments](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VTimeLLM_Empower_LLM_to_Grasp_Video_Moments_CVPR_2024_paper.html), CVPR 2024 | 时间定位训练的重要基线；适合后续把 CTNE 的高异常 transition 映射回时间窗。 |
| B15 | P1 | [The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Gong_The_Devil_is_in_Temporal_Token_High_Quality_Video_Reasoning_CVPR_2025_paper.html), CVPR 2025 | 视频推理依赖合适的 temporal token 设计；只把 caption 接到 user prompt 末尾并非等价方案。 |
| B16 | P2 | [Temporal Grounding Bridge: Bridging Video and Text for Temporal Grounding](https://openreview.net/forum?id=6pZwHdkTJY), EMNLP 2024 | 提供时序 grounding 的跨模态桥接思路，主要用于后续解释/定位扩展。 |

### 3.3 Qwen3-VL、多任务优化、持续学习与强化学习

| 编号 | 优先级 | 论文与状态 | 与本项目直接相关的结论 |
|---|---|---|---|
| C01 | P0 | [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631), arXiv 2025 | interleaved-MRoPE、DeepStack 和文字时间戳加强时空建模，但没有显式 camera pose state/head；有序帧能力不等于几何可辨识性。 |
| C02 | P0 | [MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance](https://proceedings.mlr.press/v235/wei24d.html), ICML 2024 | 辅助单模态目标与多模态目标存在梯度冲突；共同下降方向比固定 loss 权重更有原则。 |
| C03 | P0 | [AdaDARE-gamma: Balancing Stability and Plasticity in Multi-modal LLMs through Efficient Adapter Merging](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_AdaDARE-gamma_Balancing_Stability_and_Plasticity_in_Multi-modal_LLMs_through_Efficient_CVPR_2025_paper.html), CVPR 2025 | 支持把 camera 与 detection 视作稳定性-可塑性问题，但 adapter merge 只能缓解遗忘，不能创造任务间因果联系。 |
| C04 | P1 | [Model Tailor: Mitigating Catastrophic Forgetting in Multi-modal Large Language Models](https://arxiv.org/abs/2402.12048), arXiv 2024 | 解释了专门微调后通用能力或旧接口下降；对应项目中 Camera Yes/No 接管 detection 输出协议。 |
| C05 | P1 | [Dynamic Mixture of Curriculum LoRA Experts](https://arxiv.org/abs/2506.11672), ICML 2025 | 多 LoRA expert 与 curriculum 可减少任务干扰，但目前不应在 camera signal 未成立前增加结构复杂度。 |
| C06 | P1 | [Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multimodal_Continual_Instruction_Tuning_with_Dynamic_Gradient_Guidance_CVPR_2026_paper.html), CVPR 2026 | 动态梯度约束适合第二阶段 pose/detection 联合训练；第一阶段晚融合无需承担该风险。 |
| C07 | P1 | [CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html), CVPR 2025 | MoE 可隔离任务参数，但会增加训练与消融负担，仅作为长期扩展。 |
| C08 | P1 | [Robust Multimodal Large Language Models Against Modality Conflict](https://proceedings.mlr.press/v267/zhang25dq.html), ICML 2025 | 当文本 camera context 与视觉弱表征冲突时，MLLM 可能偏向错误模态；解释显式 caption 条件为何会稀释检测。 |
| C09 | P0 | [When Thinking Drifts: Visual Evidence Reward for Multimodal Reasoning](https://openreview.net/forum?id=qDm3fpLYDW), NeurIPS 2025 | CoT 可能覆盖原本正确的视觉直觉并产生幻觉；若做 RL，奖励必须锚定可验证视觉证据。 |
| C10 | P1 | [Perceptual-Evidence Anchored Reinforced Learning for Multimodal Reasoning](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Perceptual-Evidence_Anchored_Reinforced_Learning_for_Multimodal_Reasoning_CVPR_2026_paper.html), CVPR 2026 | 支持用 detector/grounding 证据约束推理，而非只奖励最终字符串。 |
| C11 | P1 | [Improving Vision-Language Models with Perception-Centric Process Reward Models](https://openaccess.thecvf.com/content/CVPR2026/html/Min_Improving_Vision-language_Models_with_Perception-centric_Process_Reward_Models_CVPR_2026_paper.html), CVPR 2026 | 过程奖励应衡量中间感知是否正确；只有 CTNE 产生可信时空分数后才有可用 camera-detection process reward。 |
| C12 | P1 | [VIDEOP2R: Video Understanding from Perception to Reasoning](https://openaccess.thecvf.com/content/CVPR2026F/html/Jiang_VIDEOP2R_Video_Understanding_from_Perception_to_Reasoning_CVPRF_2026_paper.html), CVPR Findings 2026 | 先感知后推理的分阶段范式，与“低层专家先证伪、MLLM 后解释”一致。 |
| C13 | P1 | [AVATAR: Reinforcement Learning to See, Hear and Reason Over Video](https://openaccess.thecvf.com/content/CVPR2026/html/Kulkarni_AVATAR_Reinforcement_Learning_to_See_Hear_and_Reason_Over_Video_CVPR_2026_paper.html), CVPR 2026 | 视频 GRPO 会遇到同组奖励相同导致 advantage 消失；对应项目早期 reward variance 不足问题。 |
| C14 | P1 | [Incentivizing Versatile Video Reasoning in MLLMs via Data-Efficient Reinforcement Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Incentivizing_Versatile_Video_Reasoning_in_MLLMs_via_Data-Efficient_Reinforcement_Learning_CVPR_2026_paper.html), CVPR 2026 | 数据高效 RL 可作为后续内部化阶段参考，但不能替代第一层 camera 增量信号检验。 |
| C15 | P2 | [UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_UFVideo_Towards_Unified_Fine-Grained_Video_Cooperative_Understanding_with_Large_Language_CVPR_2026_paper.html), CVPR 2026 | 多任务视频细粒度理解的统一框架，适合作为长期 architecture 参考。 |
| C16 | P2 | [Chain-of-Frames: Advancing Video Understanding in Multimodal LLMs via Frame-Aware Reasoning](https://openaccess.thecvf.com/content/CVPR2026/html/Ghazanfari_Chain-of-Frames_Advancing_Video_Understanding_in_Multimodal_LLMs_via_Frame-Aware_Reasoning_CVPR_2026_paper.html), CVPR 2026 | frame-aware reasoning 可用于把异常分数转成解释线索，不作为第一轮二分类核心。 |

### 3.4 条件异常检测、泛化与融合

| 编号 | 优先级 | 论文与状态 | 与本项目直接相关的结论 |
|---|---|---|---|
| D01 | P0 | [Contextual Learning for Anomaly Detection in Tabular Data](https://openreview.net/forum?id=PmqZslRENW), TMLR 2026 | 正式定义 context-conditional anomaly detection，利用 `Var(Y)=E Var(Y|C)+Var(E[Y|C])` 解释条件化为何可缩小正常分布边界。CTNE 的直接理论支撑。 |
| D02 | P0 | [RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_RC-NF_Robot-Conditioned_Normalizing_Flow_for_Real-Time_Anomaly_Detection_in_Robotic_CVPR_2026_paper.html), CVPR 2026 | 用 task/robot state 条件化 object trajectory 的 normalizing flow，只用正常样本训练；方法结构与 camera-conditioned temporal normality 高度同构。 |
| D03 | P0 | [CFLOW-AD: Real-Time Unsupervised Anomaly Detection With Localization via Conditional Normalizing Flows](https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html), WACV 2022 | 冻结预训练 encoder + conditional flow 的成熟实现范式，并能产生局部 anomaly map。 |
| D04 | P1 | [Noise Flow: Noise Modeling With Conditional Normalizing Flows](https://openaccess.thecvf.com/content_ICCV_2019/html/Abdelhamed_Noise_Flow_Noise_Modeling_With_Conditional_Normalizing_Flows_ICCV_2019_paper.html), ICCV 2019 | 经典例子：用 camera/gain 条件解释正常噪声变化，而不是把 camera 本身当异常；与本项目的因果角色最相似。 |
| D05 | P1 | [FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing](https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html), CVPR 2025 | detector 容易依赖特定频段；要求增广、局部/全局一致性和跨域控制。 |
| D06 | P1 | [FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FakeRadar_Probing_Forgery_Outliers_to_Detect_Unknown_Deepfake_Videos_ICCV_2025_paper.html), ICCV 2025 | 在真实、已知 fake 和未知 outlier 间建模边界，支持最终加入 open-set/outlier 对照。 |
| D07 | P1 | [Generalizing Deepfake Video Detection with Plug-and-Play: Video-Level Blending and Spatiotemporal Adapter Tuning](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Generalizing_Deepfake_Video_Detection_with_Plug-and-Play_Video-Level_Blending_and_Spatiotemporal_CVPR_2025_paper.html), CVPR 2025 | 轻量时空 adapter 与 hard negative 能增强泛化；支持冻结大 backbone、训练小模块。 |
| D08 | P1 | [D^3: Scaling Up Deepfake Detection by Learning from Discrepancy](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.html), CVPR 2025 | 多 generator 训练会在 ID 拟合和 OOD 泛化间冲突；正式结果必须同时报告 ID/OOD，不能只追 ViF 单点。 |

共收录 **70 篇**：其中 68 篇为 2024--2026 年工作，另含 2 篇直接支撑 conditional flow/异常建模的经典工作。核心方案主要由 `A01--A09`、`B01--B03`、`C01--C03/C09` 和 `D01--D04` 决定。

## 4. 对 Qwen3-VL-8B-Instruct 的具体判断

### 4.1 它已经具备什么

Qwen3-VL 使用 interleaved-MRoPE 保留图像/视频 token 的时空位置关系，DeepStack 将多层 ViT 特征送入语言模型，并使用文字时间戳加强视频时间对齐。因此，有序多帧输入并不是“模型看不见顺序”，也不能用 zero-shot 检测差直接断言模型完全没有视频能力。

### 4.2 它缺少什么

标准 Qwen3-VL 没有显式 camera pose token、SE(3) 轨迹、pose regression head 或 camera-conditioned likelihood objective。CameraBench 和 2026 年 geometry-guided probing 都表明，通用 VLM 对语义 camera primitive 尚可，但精确几何 cue 较弱。当前训练又冻结 vision tower 和 multimodal projector，Camera VQA LoRA 主要是在语言侧重新读取已有视觉表征；这足以学会 Yes/No 标签，却不保证视觉特征会被重塑成对检测有用的 camera-content relation。

这正好解释了当前看似矛盾的结果：

- Camera VQA 的视觉依赖门通过，说明已有特征中存在可读的 camera 信息；
- 正确 Camera VQA 不优于 detection-only，说明“可读”并未自动变成 Real/Fake 判别边界；
- caption 条件下降，说明把额外语言 token 放入 prompt 会改变注意力和输出分布，却没有提供模型缺失的几何归纳偏置；
- hard routing 下降，说明粗 bucket 不能替代连续 camera-content 相容性。

### 4.3 对后续微调的约束

1. 第一阶段冻结 Qwen3-VL，避免再次破坏已经较强的检测基线。
2. camera context 首先用数值几何/运动描述，不用自然语言 caption。
3. 只有 CTNE 正确条件显著优于无条件和打乱条件后，才值得进入 Cambrian-P 风格的内部 pose supervision。
4. 若进入内部化阶段，不能继续完全冻结 vision tower；至少训练新增 pose/anomaly token、projector/head，并评估解冻最高若干 ViT block。检测 replay 与 pose-only batch 采用交错训练，而不是固定 1:1 混在同一输出协议中。

## 5. 候选路线比较

| 候选路线 | 文献成熟度 | 项目证据 | 工程成本 | 当前决定 |
|---|---:|---:|---:|---|
| labels/caption 拼 user prompt 后 SFT | 中 | 严格训推一致仍下降 | 低 | 停止 |
| Camera VQA + detection replay | 中高 | camera 能力可学，但 detection 低于 control | 低 | 只保留为能力诊断，不作主方法 |
| static/minor/complex 硬路由 | 中 | 正确路由低于无路由和错误路由 | 低 | 停止 |
| RAFT 全局补偿后局部残差 | 中 | correct geometry 低于 raw/wrong | 中 | 停止，不再调参 |
| Cambrian-P 风格 pose token + pose head | 高，但面向空间/VQA | 尚无 camera-detection 因果信号；原论文需复杂采样并使用 64 H200 | 高 | 第二阶段候选，不先做 |
| **CTNE 条件正常性 + Qwen 概率融合** | **高：likelihood、conditional NF、expert fusion 均有成熟先例** | **尚未被现有实验否定；能直接回答 camera 是否有增量** | **中低** | **唯一主线** |

## 6. 收敛后的完整方法

### 6.1 变量与目标

对每个视频按 JSON 中实际列出的有序帧定义。样本 `i` 有 `n_i` 个有效帧，产生 `T_i=n_i-1` 个相邻 transition；不得默认或静默截成 16 帧：

- `C_i={c_it}_{t=1}^{T_i}`：变长连续 camera context。第一版从已有 RAFT 全局流/鲁棒几何估计中提取每个 transition 的平移、旋转、尺度、shear、inlier ratio 和拟合误差；CameraBench labels 只用于分层和语义审计。
- `Y_i={y_it}_{t=1}^{T_i}`：变长时序取证表征。第一版复用 DINOv2 帧特征轨迹、一步/二步差分、局部曲率、自相似统计和未补偿 RAFT motion 统计。
- `Z_i`：最终 `Real/Fake` 标签。

变长输入合同如下：

1. 正式实验默认 `max_frames=0`，表示使用 JSON 中所有可解码帧；RAFT/DINO 按小批次流式处理，不能为了显存静默截断。
2. `n_i<3` 时无法形成统一的二阶时序证据，样本应被标记为 `ctne_unavailable` 并报告覆盖率；融合阶段对这类样本回退到原 Qwen 分数，不用补零伪造证据。
3. 极端长样本若因工程原因必须限帧，只能采用预先固定的均匀采样规则，并把原帧数、使用帧数和索引写入 manifest；主实验与所有控制必须使用完全相同的索引。
4. 训练损失先在每个视频内部按 transition 求平均，再在视频间平均，避免帧多的样本获得更高权重。

只用训练集真实视频学习：

```text
p_theta(y_it | c_it, Z_i=Real)
```

使用 conditional normalizing flow 得到逐 transition 条件负对数似然，并按视频等权训练：

```text
L_i = (1/T_i) * sum_t[-log p_theta(y_it | c_it, Real)]
```

视频异常分数使用长度归一化的均值与尾部分位数，而不是 NLL 总和；二者权重只在训练域 validation 固定，所有外部集和控制共用：

```text
s_cam(X_i) = lambda * mean_t(NLL_it) + (1-lambda) * quantile_0.9_t(NLL_it)
```

理想检测器需要 fake 分布的似然比，但生成器持续变化。用真实正常性的一类模型代替 `p_fake`，可以减少对特定 fake generator 的依赖。因为不学习 `p(C)`，模型不应仅凭“摇摄/静止比例”判假。

### 6.2 为什么条件化可能有效

若 camera context 解释了真实视频时序表征的一部分合法变化，则：

```text
Var(Y) = E_C[Var(Y | C)] + Var_C(E[Y | C])
```

无条件 detector 必须同时容纳类内变化和不同 camera context 之间的变化；条件 detector 只需在当前 `C=c` 的正常切片中判断偏离。因此，小的生成异常可能更容易突出。

这只是带条件的充分性假设。若 `C` 不能解释 `Y`，模型会忽略 condition；若 fake 与 real 在 `p(Y|C)` 上没有差异，camera 路线就应失败。正确/打乱/无条件三组对照正是对此的直接检验。

### 6.3 与已失败 residual 方法的本质区别

已失败方法计算近似：

```text
residual = observed_flow - estimated_camera_flow
```

它隐含假设 camera 与 object motion 可以线性、无损地分离。真实视频中的视差、遮挡、非刚体、滚动快门和物体-相机交互都会破坏这个假设；相减还可能删除检测线索。

CTNE 不相减任何观测量，而是学习：

```text
what Y should look like when camera context is C
```

它允许同一种 object dynamics 在不同 camera motion 下拥有不同正常分布，也保留 camera-content interaction。

### 6.4 与 Qwen3-VL 的融合

第一版不重新全量 SFT Qwen。固定原始 detection checkpoint，在同一训练外验证集上得到：

- `l_qwen`：Real/Fake 的校准 logit 或 answer-token log-odds；
- `s_cond`：CTNE 条件异常分数；
- 可选 `s_uncond`：不使用 camera 的同容量正常性分数。

用 L2 正则 logistic calibrator 或单调小门控拟合：

```text
logit(Fake) = a * l_qwen + b * s_cond + d
```

核心 camera 消融不是“融合后是否比 Qwen 高”，而是：

```text
Qwen + matched-camera conditional expert
    > Qwen + unconditional expert
    > Qwen baseline

matched-camera expert > shuffled-camera expert
```

若只有 `Qwen + unconditional` 提升而 matched camera 不再增加，说明有用的是普通时序取证，不是 camera；论文必须诚实撤掉 camera 主张。

### 6.5 解释性 CoT

先用每个 transition/时间窗的条件 NLL 分解选出 top-k 异常时间窗，再把这些窗口及简短结构化证据交给 Qwen3-VL 生成解释。解释分支不参与第一层真假 gate，也不改变主指标。这样可避免当前自动 CoT 质量反过来污染 detector，并符合“detector evidence -> language”路线。

## 7. 分阶段验收

### Gate 1：camera 是否对时序正常性有独立增量

这个门只训练小型条件专家，不训练 Qwen3-VL。

训练与验证：

1. 先审计每个数据源的帧数分布、解码失败、重复路径和 `ctne_unavailable` 覆盖率；后续不写死 16 帧。
2. 训练只使用 DataB/GenBuster 训练部分中的真实视频；按视频身份和真实来源拆分，不能按帧或 transition 随机拆分。
3. `Y`、`C` 的 scaler、PCA 和 flow 只在训练 fold 拟合。
4. 阈值和视频级 NLL 聚合权重只在 held-out training-domain validation 选择；ViF-Bench 和 GenBuster benchmark 不调参。
5. DataA 不进入主训练；只在 Gate 1 通过后用于局部编辑/解释诊断。
6. matched、unconditional、shuffled 等控制必须复用逐样本完全相同的帧索引和有效 transition mask。

必须同时跑六个控制：

| 条件 | 目的 |
|---|---|
| 无条件 normality | 判断普通时序 cue 的贡献 |
| 正确连续 camera condition | 主方法 |
| 样本间打乱 camera condition | 检验模型是否真正利用 camera-content 配对 |
| camera-only classifier | 检查来源/运动分布捷径 |
| raw-motion discriminative probe | 对齐已完成几何 residual gate |
| correct residual / wrong residual | 明确新方法不是重复旧实验 |

通过标准：

- 正确条件在外部集的 pooled AUROC 与跨 generator macro Balanced ACC 中，至少一个相对无条件提高 `>=1.0` 个百分点，另一个不下降超过 `0.5` 点；
- 正确条件同时优于 shuffled condition，paired bootstrap 95% CI 下界大于 0；
- camera-only 在 canonical/matched 控制下不得形成可疑高分；必须报告 real-vs-real 来源区分 floor；
- 至少在两个 motion bucket 和过半数有充分样本的 generator 上方向一致，不能由单一来源撑起；
- 多随机种子报告均值/方差，不能只保留最佳 seed。

若 Gate 1 未通过，停止 camera 论文主线；转为 VidAudit 风格的 cross-substrate temporal expert，不再做 pose token、RL 或更多 camera prompt。

### Gate 2：camera 增量能否改善最终 Real/Fake

Gate 1 通过后才做。固定同一 Qwen 检测 checkpoint 和 prompt，比较：

1. 原 Qwen 检测模型；
2. Qwen + unconditional temporal expert；
3. Qwen + matched-camera CTNE；
4. Qwen + shuffled-camera CTNE；
5. CTNE 单独输出。

通过标准：

- 在 ViF-Bench 与 GenBuster benchmark 至少一个数据集上，`Qwen + CTNE` 相对原 Qwen 的 macro Balanced ACC 或 Fake F1 提高 `>=1.0` 点，并在另一个数据集不下降超过 `0.5` 点；
- `Qwen + CTNE` 必须优于 `Qwen + unconditional`，否则不能把提升归因于 camera；
- Real Recall 不下降超过 1 点，避免只通过多报 Fake 提高 Fake Recall；
- 使用 paired bootstrap CI、固定 prompt hash，并确保每个模型对同一样本使用完全相同的变长有序帧列表和 preprocessing；逐 generator 报告，不只给总体均值。

### Gate 3：可选的 Qwen 内部化

只有 Gate 1 和 Gate 2 都通过、且晚融合已经证明 camera 有用后再做：

- 参考 Cambrian-P，在每帧视觉 token 后增加 camera/pose token 与轻量 pose head；
- 增加 CTNE anomaly token 或蒸馏 loss，使 pose 表征与最终 detection logit 建立显式联系；
- pose-only、detection-only、joint batch 交错训练，并使用 gradient cosine/PCGrad/MMPareto 监控冲突；
- 推理不提供 caption，可消融去掉 pose token，判断收益来自训练表征还是 inference condition；
- 再考虑 evidence-grounded RL，而不是先用只奖励 `Yes/No` 或 `Real/Fake` 的 GRPO。

Gate 3 是增强版，不是当前最小可行论文必须项。若 Gate 2 已在两个外部 benchmark 稳定提点，条件专家 + MLLM 融合本身已形成完整方法。

## 8. 数据角色与偏置处理

| 数据 | 最合适的角色 | 不应承担的角色 |
|---|---|---|
| DataB detection | Qwen 检测起点、训练域 detection/calibration | 其内部训练样本准确率不能当 held-out 结果 |
| DataB/GenBuster real train | CTNE 正常分布训练 | 不与 benchmark real 重复或近重复 |
| ViF-Bench | 外部全生成 OOD 检测评测 | 项目已多次查看，不能包装成完全未触碰的最终 test |
| GenBuster benchmark | 第二个外部全生成评测 | 不使用其 benchmark 标签调融合权重 |
| DataA | 局部编辑时间/空间定位、成对解释诊断 | 不与全生成样本粗暴混为同一个主训练分布 |
| CameraBench labels/caption | camera extractor 语义审计、分桶和定性解释 | 不直接作为最终 detector 的 user-prompt 文本 |

DataA 的同源 Real/Fake pair 具有几乎相同 camera motion，这恰好说明 camera marginal 本身不应区分二者。它能检验的是：在相同 camera context 下，局部生成区域是否产生异常的 `Y|C`。因此 DataA 应在 Gate 1/2 通过后用于定位与解释，不应用来证明通用全生成检测提升。

## 9. 实施优先级

1. **先完成 Gate 1。** 复用已存在的 DINOv2、RAFT 和各样本实际有序帧，不下载新大模型，不训练 Qwen。
2. Gate 1 通过后，提取原 detection checkpoint 的稳定 Real/Fake logit，训练小校准器并跑 Gate 2。
3. Gate 2 通过后，补 DataA 局部时间窗/区域解释和统计显著性；这是最小完整论文版本。
4. 有充足时间再下载 VGGT、做 continuous pose condition 替换 RAFT condition，并尝试 Cambrian-P 风格内部化。
5. RL 放在最后；只有异常时间窗、camera consistency 和最终 answer 都有可验证 reward 时才启动。

存储约定：Gate 1 的临时特征先放 `/tmp`；只有通过并确定会被 Gate 2 复用后，才作为正式大文件上传 OSS。split、manifest、scaler 配置、指标 JSON/CSV 和 bootstrap 汇总是小型正式产物，应持久化到 NAS。

## 10. 最终判断

Camera 方向仍然可以做，但论文命题必须从“给 MLLM 增加 camera 知识”改成：

> **相机运动解释了真实视频中一部分合法的时序变化；在给定相机轨迹后建模正常时序分布，可以更敏感地发现不符合该上下文的生成异常。**

这个命题比 camera caption、Camera VQA 或 hard routing 更直接，也能被正确/打乱/无条件实验严格证伪。它同时给当前所有负结果一个统一解释，而不需要否认那些实验：camera 表征确实可学，但只有进入最终异常评分的条件关系，才可能改善 Real/Fake。

当前建议是把 CTNE 作为唯一主方案，先做 Gate 1，不并行启动 pose token、GRPO、更多 caption SFT 或 DataA 混训。Gate 1 通过后再进入 Qwen 融合；Gate 1 不通过，则停止 camera 作为主贡献，保留其作为失败消融并转向不带 camera 主张的跨 substrate 时序检测。
