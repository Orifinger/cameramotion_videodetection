# Camera-label SFT 前置学习失败根因分析

## 分析结论

本轮结果不能推出“MLLM 学不会相机运动”，也不能推出“相机运动对 AIGC 视频检测无用”。它只证明当前配方没有学到可验证的相机运动能力：从 detection SFT checkpoint 出发，用约 750 个视频、每视频一个完整标签集目标、语言层 LoRA rank 32 和学习率 `1e-5` 训练四轮，模型主要学习了标签先验并塌缩到 `complex-motion`。

CameraBench 与本轮不是同一个训练任务。CameraBench 的核心监督是逐 primitive 的二元 VQA、显式正负样本和候选文本评分；本轮则要求模型一次自由生成平均约 5 个标签的完整集合。前者把多标签问题分解成大量平衡的简单决策，后者在 token-level CE 下没有为缺失标签提供显式负监督，最容易通过输出高频标签组合降低 loss。

因此，当前应停止的是“一视频一条完整 camera-label list 的低学习率 LoRA 前置学习”，而不是整个 camera-aware detection 方向。下一步若继续，必须先按 CameraBench 的真实任务形式完成一次相机能力复现，再讨论如何迁移到检测。

## CameraBench 实际做了什么

官方论文与发布模型显示：

- 训练集有 1,402 个独立视频，论文称由同一套技能生成约 230K video-QA pairs 和 1,402 captions。
- 官方 7B 主结果使用 Qwen2.5-VL full fine-tuning，冻结视觉塔和多模态 projector，使用原生 video 输入、8 FPS、学习率 `2e-5`，表格配置为 6 epochs。
- 官方 LoRA 消融使用 rank 64、学习率 `2e-4`；论文明确指出五轮后 full fine-tuning 优于 LoRA。
- 主分类评测不是生成完整标签集，而是针对每个 motion primitive 提问 Yes/No，用 `P(Yes)` 计算逐类别 Average Precision。
- VQA 测试将同一问题配成答案相反的两个视频；Q-Acc 只有两个都答对才得分，用来抑制不看视频的答案先验。
- 论文的 FPS 消融显示 8 FPS 稳定优于 2/4 FPS。对 7B LoRA，平均 AP 从 2 FPS 的 51.3 提高到 8 FPS 的 56.7；full SFT 从 56.8 提高到 59.3。

本地获批的 CameraBench 训练文件进一步说明了监督密度：

| 文件 | 记录数 | 独立视频 | 每视频平均记录 | 关键性质 |
|---|---:|---:|---:|---|
| `balanced_vqa.json` | 38,672 | 1,402 | 27.58 | Yes/No 各 19,336；每个视频自身的 Yes/No 数也相等 |
| `captionset.json` | 35,050 | 1,402 | 25 | 同一人工 caption 配 25 个不同 user prompts |
| `imb_raw.json` | 157,552 | 1,261 | 124.94 | 148 类问题的原始长尾 VQA，No 占多数 |

发布的 7B preview checkpoint 的 trainer state 为 3,670 optimizer steps、约 9.98 epochs、全局 batch 256，反推每轮约 94K 处理记录。该 preview 与论文最终表格的 6-epoch 配置属于不同发布版本，但二者都远大于本轮每轮约 750 条完整标签记录。

## 与本轮配方的逐项对比

| 维度 | 本轮 | CameraBench | 对结果的影响 |
|---|---|---|---|
| 起点 | 经过固定检测 prompt 和长 CoT 专项 SFT 的 Qwen3-VL-8B | 通用 Qwen2.5-VL-Instruct | 本轮可能存在任务专化或遗忘；尚未用通用起点做因果对照 |
| 训练目标 | 一次生成 33 类 taxonomy 中约 5 个标签 | 逐 primitive Yes/No、VQA、caption 多任务 | 本轮缺少逐标签负监督，组合空间大，容易学先验 |
| 样本密度 | 每视频 1 条，约 750 条/epoch | 数十至上百条问答/视频 | CameraBench 对每个视觉概念提供多次、可分解梯度 |
| 类别平衡 | Gold dev 为 complex 213、no-motion 61、minor 47 | balanced VQA 显式构造等量 Yes/No | 本轮普通 CE 强烈偏向 complex-motion |
| 参数更新 | LoRA rank 32，LR `1e-5` | full LR `2e-5`；LoRA rank 64、LR `2e-4` | 本轮 LoRA 学习率比官方 LoRA 最优值低 20 倍，容量也更小 |
| 时间输入 | 16 张离散图片，约等于平均 2–3 FPS | 原生 video，8 FPS | 本轮时序密度和 video-specific 表征更弱 |
| 评测 | 自由生成整集、exact/micro/macro/bucket | 候选 primitive 的 Yes 概率与 AP；配对 VQA Acc/Q-Acc | 两者数字不能直接比较；CameraBench 没证明整集生成已解决 |
| 训练规模 | 48/96/144/192 steps，约 1–4 epochs | 数千 optimizer steps、6–10 epochs 版本 | 本轮更新次数和任务实例曝光量低一个到多个数量级 |

## 训练与推理是否一致

### 本轮内部一致的部分

- 干净四轮实验的 train/dev 都使用相同 canonical system/user prompt。
- 都使用同样的 16 帧图片组织、`image_max_pixels=262144` 和相同 taxonomy。
- Correct 与 shuffled 分支的视频、prompt、训练步数和 LoRA 容量一致。
- 修复后从初始 detection checkpoint 干净重训，assistant 结束标记也进入 loss；格式有效率恢复到约 99%。
- 评测按标签集合计算，不要求输出标签顺序与训练顺序完全相同。

因此，最终多数类塌缩不是由明显的 train/inference prompt 不一致、adapter 未加载或输出终止 bug造成的。

### 与 CameraBench 能力定义不一致的部分

- 本轮虽然 train/inference 自洽，但训练和评测的任务都是“完整标签集生成”，并不等价于 CameraBench 的 binary primitive scoring/VQA。
- 本轮把视频拆成多张图片；CameraBench 使用原生 video 和 FPS 采样。两者内部各自一致，但视觉输入机制不同。
- 本轮 shuffled-training 使用固定语义置换，会改变标签名边缘分布；它是等算力错误监督控制，却不能单独排除标签先验。后续 balanced accuracy 已揭示这一点。

## 为什么会得到当前曲线

### 1. 主要原因：目标函数鼓励高频标签模板

多标签 list SFT 只对实际出现的输出 token 计算正向语言建模损失。一个标签没有出现时，没有独立的 BCE 式负损失告诉模型“这个标签为假”。训练集中 `regular-speed`、`complex-motion`、`no-shaking` 等标签频率很高，模型只需掌握格式和高频组合就能快速降低 loss。

这与结果吻合：格式率从约 97% 升至 99%，micro-F1 达到约 50%，但 macro-F1 只有约 22%，Exact set 只有约 3%；模型学会了格式和常见标签，而没有学会长尾 primitive。

### 2. 类别不平衡把 motion bucket 推向 complex-motion

开发集 bucket 分布为 complex `213/321`、no-motion `61/321`、minor `47/321`。Correct 模型四轮分别将 283、279、266、273 条预测为 complex，balanced accuracy 只有 33.25%–35.98%，接近三分类随机水平 33.33%。这不是简单“少训一轮”：第三轮到第四轮已经回落。

CameraBench 的 balanced VQA 恰好针对这一问题：把每个概念拆成 Yes/No，并为同一问题配相反答案的视频，使不看视频的先验策略无法获得高 Q-Acc。

### 3. LoRA 配置相对官方配方明显偏弱

本轮采用 rank 32、LR `1e-5`；CameraBench 的 LoRA 消融采用 rank 64、LR `2e-4`，主结果使用 full fine-tuning。因而本轮不能用来证明“LoRA 学不会 camera”，只能证明当前低学习率、小容量 LoRA 加整集目标没有学会。

### 4. 时序输入不足是次要但真实的损失

CameraBench 平均视频 5.7 秒，8 FPS 大约提供 46 帧；本轮固定 16 帧大约相当于 2.8 FPS。论文消融表明低 FPS 会稳定降低 AP。它可能加剧细微平移、旋转、zoom 与 object motion 的混淆，但仅凭 FPS 差异不足以解释几乎纯多数类的输出，主要问题仍是监督形式和优化配置。

### 5. Detection checkpoint 起点可能造成任务专化，但尚未被单独证明

本轮从经历五轮 detection/CoT SFT 的 checkpoint 出发，而 CameraBench 从通用 Instruct 模型出发。基础模型对新 camera JSON prompt 的格式有效率为零，说明输出接口完全陌生，但不能单凭这一点证明视觉 camera 知识已遗忘。需要用同一 binary VQA 数据分别微调通用 Instruct 和 detection checkpoint，才能把起点影响与目标设计分开。

### 6. 标签质量不是当前首要嫌疑

DataA camera labels 与 captions 来自 CameraBench 的人工标注体系；本地旧版 real 数据中有 1,067 条有效 camera records、418 种标签组合，平均每条约 4.94 个标签。数据不是单一标签模板，caption 与 no/minor/complex 语义也基本一致。标签仍可能有边界歧义，但现有证据更支持“监督被压缩和类别不平衡”，而不是大规模错标。

## 当前实验能与不能说明什么

### 能说明

1. 单一 prompt 本身不是主要故障；train/inference prompt 一致后仍然塌缩。
2. 直接生成完整 camera label set 是不合适的最小 pretext，尤其不适合小数据、低学习率 LoRA。
3. 只比较 correct-label training 与语义置换 training 会受标签先验影响；必须报告 balanced metrics 或同模型 shuffled-video 控制。
4. 增加到四轮没有解决该配方，继续相同训练没有合理依据。

### 不能说明

1. 不能说明 Qwen3-VL 或 MLLM 学不会 camera motion。
2. 不能说明 CameraBench 数据对我们的模型无效，因为我们没有使用其核心 balanced VQA 形式和官方量级。
3. 不能说明 camera motion 无法帮助 AIGC 检测，因为 camera 能力本身尚未按合理任务学成，更没有进入有效的联合迁移实验。
4. 不能把我们的 macro-F1/Exact set 与 CameraBench 的 AP 直接对比。

## 从失败中得到的可用方向

最值得保留的思路不是继续扩写 camera 文本，而是把 camera supervision 变成显式、平衡、可校准的辅助决策：

1. 将每个 camera primitive 转成独立 Yes/No 任务，构造同问题正负视频平衡的数据；输出只需一个 token，避免整集组合和格式损失。
2. 使用 candidate-level `P(Yes)` 形成 camera score vector，而不是把自然语言 caption 当作推理时外部上下文。
3. 在检测训练中持续保留 camera VQA auxiliary loss，避免“先学 camera、再被 detection 覆盖”；correct camera auxiliary 与 shuffled-video/label control 才是因果比较。
4. 对同源 real/fake pair，可约束两者 camera score vector 一致，同时让检测分数区分真假。这利用了 DataA 的真实结构：局部编辑改变伪影，但不应改变全局相机运动。
5. 若 camera score vector 本身不能在 held-out balanced VQA 上超过 shuffled-video/no-video，就不进入检测迁移。

## 建议的下一步因果排查顺序

这不是立即启动的新实验，仅是从本轮失败中收敛出的最小诊断顺序：

1. **任务复现门**：用当前 train/dev 身份，把 camera labels 展开为平衡 binary VQA；使用原生 real 视频与 8 FPS，按 AP、balanced Acc 和 paired Q-Acc 评测。
2. **视觉依赖门**：同一模型比较正确视频、shuffled-video 和 no-video；正确视频必须显著更好。
3. **起点对照**：通用 Qwen3-VL-Instruct 与 detection checkpoint 使用完全相同的 VQA LoRA 配方。只有 detection 起点失败时，才归因于专项 SFT 遗忘。
4. **容量对照**：低成本先采用 CameraBench 论文的 LoRA rank 64、LR `2e-4`；若任务可学但距离官方结果明显，再考虑 full LM，而不是先调 prompt。
5. **检测迁移门**：camera VQA 学成后，联合训练 camera auxiliary、DataA detection 与 DataB replay；检测推理不提供 GT camera 文本。Correct auxiliary 必须同时超过 no-camera auxiliary 和 shuffled-video/label auxiliary。

若第一、二门通过，说明之前是任务设计错误，camera 方向仍可继续；若使用合理 binary VQA、8 FPS、官方级 LoRA 配置仍无法学成，再排查 Qwen3-VL/检测起点或数据映射问题。只有这些排查完成后，才有依据说该方向本身不可行。

## 依据

- CameraBench 论文：<https://arxiv.org/pdf/2504.15376>
- CameraBench 官方仓库：<https://github.com/sy77777en/CameraBench>
- 官方 7B 发布模型与训练元数据：<https://huggingface.co/chancharikm/qwen2.5-vl-7b-cam-motion>
- 本地获批训练数据：`E:/newgaibeishi/camerabench_train_2/cam_motion/`
- 本轮结果：`/tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_clean_4epoch_curve.json`
