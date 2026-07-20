# ViF-Bench 强检测模型残余错误分析（2026-07-20）

## 分析问题

本分析不重新训练或推理模型，而是对强 DataB detection Qwen 在 ViF-Bench 上的逐样本结果做对齐审计，回答三个问题：

1. 319 个残余错误集中在哪里；
2. 当前时序/相机专家为什么有 oracle 救错空间，却无法转化为最终 Real/Fake 增量；
3. camera 是否仍存在可验证的作用，以及下一步最小训练门应如何隔离 generator 与 camera 贡献。

## 输入与契约

- 置信度融合逐样本结果：`res/vifbench_qwen_confidence_fusion/v1/eval/vifbench_confidence_fusion_items.csv`
- 时序/相机专家逐样本结果：`res/camera_discriminative_gate/v1/eval/vifbench/camera_discriminative_gate_items.csv`
- 历史 Qwen response：`Qwen3-VL-v4vif_2766busterall_trainall-vifbench.json`
- CameraBench 预测相机标签：`datab_cameramotion_labels_v2.jsonl`
- 3156 条有效置信度记录全部与专家及 camera 数据对齐；4 条因答案 token 合约无效未进入分析。
- ViF-Bench 标签参与本分析与 OOF 校准，因此所有结果均为开发诊断，不是独立最终测试收益。

## 1. 错误总体结构

| GT 类别 | 样本数 | 错误数 | 召回率 |
|---|---:|---:|---:|
| Real | 165 | 38（Real→Fake） | 76.97% |
| Fake | 2991 | 281（Fake→Real） | 90.61% |
| 合计 | 3156 | 319 | - |

错误明显不对称：88.1% 的错误是 Fake→Real 漏检。

## 2. 错误集中于源内容保持型生成器

| 生成器 | Fake 样本 | Fake→Real | Fake Recall |
|---|---:|---:|---:|
| HunyuanVideo-I2V | 165 | 66 | 60.00% |
| Wan2.1-VACE-1.3B-T | 165 | 50 | 69.70% |
| Wan2.2-TI2V-5B-I | 165 | 27 | 83.64% |
| Wan2.2-I2V-14B | 165 | 23 | 86.06% |
| LTX-Video-13B-I | 164 | 21 | 87.20% |
| SkyReels-V2-I2V-14B-540P | 164 | 13 | 92.07% |

这六类 I2V/编辑型生成器只占 988/2991（33.0%）Fake，却贡献 200/281（71.2%）漏检，漏检率为 20.24%；其余生成器漏检率为 4.04%，相差约 5 倍。仅 HunyuanVideo-I2V 与 Wan-VACE 就以 330 条样本贡献 116 条漏检。

错误并非由少数坏源视频独占：136/179 个同源 base 至少出现一个错误，错误最多的前 20 个 base 仅覆盖 35.4% 错误。Real 误报 base 上的 Fake 漏检率反而更低（4.82% 对 10.75%），说明 Real→Fake 与 Fake→Real 不是同一种“内容本身很难”的单一故障。

## 3. Camera motion 是难度轴，不是当前可用真假证据

| GT | Camera bucket | 样本数 | 错误率 |
|---|---|---:|---:|
| Real | static/no-motion | 60 | 31.67% |
| Real | minor-motion | 58 | 24.14% |
| Real | complex-motion | 47 | 10.64% |
| Fake | static/no-motion | 1107 | 6.87% |
| Fake | minor-motion | 1188 | 8.33% |
| Fake | complex-motion | 696 | 15.23% |

Fake 的复杂运动更容易掩盖生成异常，而 Real 的静态/低速画面更容易触发模型对手部、接触和面部区域的过度怀疑。按 Fake 生成器分层后，相机标签与漏检仍有关联：`pan-right`、`unsteady`、`fast-speed`、`complex-motion` 的合并 odds ratio 分别约为 3.03、2.23、2.12、1.98。

这些是相关性，不是 camera 自身的真假证据。标签由 CameraBench 模型预测，也可能编码画面内容和生成器特征，因此必须通过 matched/shuffled 和独立测试控制。

## 4. Qwen 置信度不足以可靠路由

- 错误样本 `|Fake-Real logit margin|` 中位数为 19.63，正确样本为 20.50；二者虽有统计差异，但都高度饱和。
- 最低置信度 10% 样本仅覆盖 76/319（23.8%）错误。
- 316/319（99.1%）错误的绝对 margin 仍不低于 15。

因此这些不是通常意义上的低置信度错误。仅依靠 answer-token margin 很难识别应该交给专家的样本。

## 5. 专家 oracle 空间无法转化为实际增量

| 专家 | 与 Qwen 分歧 | 救错 | 误伤 | 分歧中救错比例 |
|---|---:|---:|---:|---:|
| 正确相机交互 | 1456 | 142 | 1314 | 9.75% |
| 无相机时序证据 | 1603 | 122 | 1481 | 7.61% |
| 打乱相机交互 | 1561 | 144 | 1417 | 9.22% |
| 仅相机专家 | 1039 | 180 | 859 | 17.32% |

打乱相机甚至比正确相机多覆盖 2 个 Qwen 错误。完美 oracle 确实能提高上限，但现实中没有可辨识的 gate：专家分歧时，82.7%–92.4% 是 Qwen 原本正确。五折 OOF 最终只有正确相机与打乱相机各翻转一个样本，而且都是误伤同一个正确 Fake。

## 6. CoT 表明存在两种相反的输出模板

- 127 个正确 Real 均无伪影 `<type>` 标签。
- 38 个 Real→Fake 全部输出伪影标签，主要是 Contact Region Artifact（22）、Hand Anatomy Error（20）和 Facial Landmark Distortion（10）。
- 2710 个正确 Fake 中 2709 个输出伪影标签。
- 281 个 Fake→Real 中只有 1 个输出伪影标签。

在具有同源 Real 的 2976 条 Fake 中，Fake→Real CoT 与同源 Real CoT 的 TF-IDF 余弦相似度中位数为 0.151，正确 Fake 为 0.084（Mann–Whitney 单侧 `p=1.48e-30`）。这只是文本行为诊断，不是视觉因果证明，但与 I2V/编辑模型保持源内容、模型直接进入 Real 验证模板的解释一致。

## 7. 直接 camera 校准仍未达到方法门槛

在相同原视频分组五折 OOF 下额外检查了直接使用 CameraBench 标签校准 Qwen margin：

| 条件 | 跨生成器 Macro Balanced ACC | AUROC | 相对 margin-only |
|---|---:|---:|---:|
| margin-only | 83.907% | 83.991% | - |
| margin + coarse bucket（固定阈值） | 83.907% | 85.304% | 主指标 0 |
| margin + coarse bucket（嵌套阈值） | 83.940% | 85.304% | +0.032 点 |
| margin + fine labels | 81.667% | 84.417% | -2.240 点 |

粗粒度 camera bucket 改善了排序，但固定阈值不改变任何答案；嵌套阈值只救回 2/319，bootstrap 95% CI 为 `[0, +0.081]` 个百分点，下界仍为 0。细粒度标签产生 19 次救错和 33 次误伤，并且没有稳定超过 20 次同类别、同生成器内打乱控制。

因此 camera-difficulty 现象真实存在，但当前信号强度不足以支撑推理时 camera 路由、校准或专家融合。

## 结论

1. 当前主要瓶颈不是一般时序理解，而是高质量、源内容保持型 I2V/编辑视频的 Fake→Real 漏检。
2. Camera motion 调节检测难度，但当前没有形成 sample-specific 的真假增量；它更适合作为训练数据分层变量，而不是推理证据。
3. 当前 RAFT/DINO 专家、camera 文本注入、camera 交互专家、硬路由和置信度融合已经形成闭环反证，不应继续围绕相同特征调参。
4. 仅凭错误 CoT 无法知道漏检视频中真正缺失的视觉取证线索。选择新专家前，需要对 top I2V/编辑生成器的 Fake→Real、同源 Real 和正确 Fake 做小规模视觉审计。

## 最小下一步

若仍希望保留 camera 故事，下一门应把 camera 从“输入条件”改为“训练时难度分层”，并采用等样本、等步骤的 2×2 控制：

| 分支 | I2V/编辑型 hard replay | Camera-difficulty 分层 |
|---|---|---|
| 随机 replay | 否 | 否 |
| 生成器 hard replay | 是 | 否 |
| camera hard replay | 否 | 是 |
| 生成器 + camera hard replay | 是 | 是 |

Camera 分支只有在超过生成器 hard replay 至少 0.5 个跨生成器 Macro Balanced ACC 点、matched 权重优于 shuffled 权重且 bootstrap 95% CI 下界大于 0 时，才有资格成为方法贡献。ViF-Bench 只能用于开发选择；通过后必须在未参与选择的 GenBuster benchmark 验收。

在启动这轮训练前，应先抽取约 60–80 个分层残余错误视频进行人工视觉审计，确认值得增加的是局部细节、频域/纹理、时序一致性还是源条件差分专家。
