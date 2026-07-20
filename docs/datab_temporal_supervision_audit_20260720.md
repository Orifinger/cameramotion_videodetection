# DataB 时序监督内容审计

- 输入：`D:\1codex\camera\cameramotion_videodetection\ourdata\dataB\v4vif_2766busterall_trainall.json`
- SHA256：`06af06923c537a297e6b7a55620ce5dfe6041379965f0684bfba961b81b579bb`
- 样本数：6766（Real 3383 / Fake 3383）
- 帧数分布：{'11': 1, '16': 6748, '17': 17}

## 口径

`<t>[start, end]</t>` 是 system prompt 强制要求的输出格式，不能单独证明模型进行了跨帧推理。
本审计分别统计保守的时序伪影类别、显式跨帧语言，以及时间标签覆盖范围。

## 核心结果

| 指标 | 数值 |
| --- | ---: |
| 具有保守时序伪影类别的 Fake 样本 | 139 / 3383 (4.11%) |
| 时序类别在全部伪影标签中的占比 | 148 / 5250 (2.82%) |
| 仅含非时序类别的 Fake 样本 | 3244 (95.89%) |
| 含显式跨帧语言的全部样本 | 3701 (54.70%) |
| 含显式跨帧语言的 Fake 样本 | 966 (28.55%) |
| 含显式跨帧语言的 Real 样本 | 2735 (80.85%) |
| 覆盖至少 80% 视频长度的时间标签 | 8543 / 11769 (72.59%) |
| 精确覆盖整段视频的时间标签 | 8083 / 11769 (68.68%) |

## 伪影类别分布

| 类别 | 出现次数 |
| --- | ---: |
| Contact Region Artifact | 1654 |
| Hand Anatomy Error | 1015 |
| Object Deformation | 501 |
| Facial Landmark Distortion | 414 |
| Boundary Fusion | 381 |
| Physical Interaction Error | 341 |
| Malformed Text | 279 |
| Occlusion Error | 168 |
| Motion Discontinuity | 123 |
| Object Part Inconsistency | 100 |
| Face Boundary Fusion | 93 |
| Material Inconsistency | 83 |

## 结论边界

该审计衡量训练文本中是否存在显式时序监督，不衡量 Qwen3-VL 是否真正利用帧顺序，
也不衡量这些时序描述是否能提高 Real/Fake 检测。后两者需要输入干预和逐样本残余错误分析。
