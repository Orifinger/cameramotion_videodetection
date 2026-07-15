# 三分类相机运动硬路由检测专家验证

## 这轮实验测什么

这轮先不把硬路由当最终论文方法，而把它当成一个低成本机制门：在检测 prompt、输入帧和训练数据总集合不变时，按模型从画面预测的 `无相机运动 / 轻微运动 / 复杂运动` 选择对应 detection LoRA，是否优于一个看全部训练数据的共享 detection LoRA，并显著优于循环选错专家的路由控制。

检测推理始终不接收 camera caption、camera label 或其他相机文字。相机模型与检测模型读取 ViF-Bench index 指向的同一 16 帧；路由代码与既有 `ViFBench.py` 一样读取 `timestamps.txt` 并按 `1.png ... N.png` 组帧，主实验不依赖不完整的原视频集合。

## 必须复制到服务器的代码

从 GitHub 更新以下文件，并覆盖到 `/input/workflow_58770161/workspace/test/cameramotion_det` 的同名位置：

- `tools/build_camera_hard_route_gate.py`
- `tools/install_camera_hard_route_gate.py`
- `configs/camera_hard_route_gate/train_template.yaml`
- `configs/camera_hard_route_gate/train_router.yaml`
- `configs/camera_hard_route_gate/train_smoke.yaml`
- `scripts/camera_hard_route_gate/__init__.py`
- `scripts/camera_hard_route_gate/route_manifest.py`
- `scripts/camera_hard_route_gate/run.sh`
- `scripts/camera_detection_retention/run_vifbench.sh`

## 输出存储

- 一次性训练 adapter、合并模型、路由逐样本 logit 和 ViF 逐样本预测：`/tmp/1res/camera_hard_route_gate/v1`。
- 正式小型数据摘要、split、route manifest 和评测 JSON：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_hard_route_gate/v1`。
- 第一轮校准未通过时，不上传 OSS。
- 硬路由 ViF 门通过后，四个训练 adapter 才属于可复用大文件，再执行：

```bash
ossutil64 cp -r /tmp/1res/camera_hard_route_gate/v1/train/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_hard_route_gate/v1/train/
```

## 第一优先级：只做路由校准

这里先训练一个很小的三桶 router，不训练 detection 专家。旧二元相机 adapter 的标签构建曾忽略独立 `static` 标签，不能直接把 `static` 强行解释为其 `no-motion` 正类；新 router 明确把 `static/no-motion` 合并为第一桶，并只用 DataA train 的 real 帧训练。随后在 held-out DataA 的 real/fake 两套帧上检查三分类和路由一致性。

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/camera_hard_route_gate/run.sh

STAGE=preflight bash "$RUN"
STAGE=build bash "$RUN"
STAGE=smoke bash "$RUN"
STAGE=train_router bash "$RUN"
STAGE=calibrate_dataa_route bash "$RUN"

cat /tmp/1res/camera_hard_route_gate/v1/routes/dataa_route_summary.json
```

进入专家训练的预注册条件：

- score coverage 为 100%；
- 三分类 accuracy 至少 60%；
- macro recall 至少 55%；
- 三个桶的 recall 均至少 40%；
- 同一 DataA real/fake pair 的预测路由一致率至少 80%。

未达到时，先看 confusion 与分数分布，不训练四个 detection LoRA。阈值调整只需复用现有 score，不重跑 GPU：

2026-07-15 实际结果中，三分类总体 accuracy 为 73.46%、macro recall 为 58.64%、pair consistency 为 92.59%，但 `minor-motion` recall 仅 4.90%，所以三分类门已确定未通过。不要执行本文件后面的三专家 `train_all`。先运行无需训练和 GPU 的二路复核：

```bash
STAGE=audit_dataa_binary_route bash "$RUN"
cat /tmp/1res/camera_hard_route_gate/v1/routes/dataa_binary_route_summary.json
```

二路映射固定为 `no-motion` 对 `motion = minor-motion + complex-motion`，不比较多种映射后挑最好结果。二路门要求 coverage 100%、accuracy 和 Balanced ACC 均至少 75%、两类 recall 均至少 70%、real/fake pair consistency 至少 90%。通过只允许继续实现“共享模型 + 静止专家 + 有运动专家”，不能追认三分类门成功，也不能直接运行当前三专家训练命令。

```bash
MIN_ROUTE_PROBABILITY=0.45 MIN_ROUTE_MARGIN=0.08 \
STAGE=aggregate_dataa_route bash "$RUN"
```

这些数值只是命令示例，不是预设最终阈值；最终阈值只能根据 held-out DataA 的 coverage/accuracy 权衡确定，然后原样用于 ViF-Bench。

若 DataA 路由门通过，router 将在后续 ViF manifest 中复用，先用一个直接命令保存到 OSS：

```bash
ossutil64 cp -r /tmp/1res/camera_hard_route_gate/v1/train/router/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_hard_route_gate/v1/train/router/
```

## 路由校准通过后：训练共享模型与三个专家

**当前状态：禁止执行本节。** 2026-07-15 三分类路由门已因 `minor-motion` recall 4.90% 未通过；本节仅保留原三分类协议记录。只有二路检测数据和训练入口另行实现并通过审计后，才执行新的二专家命令，不能复用下面的 `train_all`。

router 从同一个 detection checkpoint 开始，LoRA rank 16、alpha 32、学习率 `1e-4`、3 epochs，只学习三个 coarse Yes/No 问题。每个问题内部 Yes/No 等量，三个问题之间的记录数也严格相同，避免 complex question 因出现更频繁而获得额外先验。四个 detection 分支同样从该原始 detection checkpoint 开始，不挂载 router，只训练 detection 输出；训练 prompt 中没有 camera 文本。共享分支的数据是三个专家数据的精确不重叠并集。detection LoRA 默认 rank 16、alpha 32、学习率 `5e-5`、2 epochs，冻结视觉塔和多模态 projector。

```bash
STAGE=smoke bash "$RUN"
STAGE=train_all KEEP_ALIVE_AFTER_RUN=1 bash "$RUN"
```

也可分开执行：

```bash
STAGE=train_shared bash "$RUN"
STAGE=train_no_motion bash "$RUN"
STAGE=train_minor_motion bash "$RUN"
STAGE=train_complex_motion bash "$RUN"
```

## 生成 ViF-Bench 三分类 route manifest

这一步不要求 ViF-Bench 有 gold camera 标签，也不使用原视频。脚本对每个 index frame directory 构造三个互斥问题，对 `Yes-No` logits 做三路相对 softmax，保存 top-1、margin、低置信度回退和循环错误路由。

```bash
STAGE=build_vif_route_inputs bash "$RUN"
STAGE=score_vif_route bash "$RUN"

# 必须使用 DataA 校准后冻结的同一阈值；下面仍是示例。
MIN_ROUTE_PROBABILITY=0.45 MIN_ROUTE_MARGIN=0.08 \
STAGE=aggregate_vif_route bash "$RUN"

cat /tmp/1res/camera_hard_route_gate/v1/routes/vifbench_route_summary.json
```

route summary 额外报告 Real 与 Fake 的路由分布差异。若两者差异很大，必须在论文中标记为潜在 benchmark shortcut，不能把 route 本身与真假相关当成“伪影推理”的证据。

## ViF-Bench 专家推理与合成

三个专家的 ViF 推理相互独立，各自只加载自己的合并模型。共享分支会在同一 prompt/index 下同时生成一次原始 detection checkpoint 基线和共享 LoRA 结果，用来消除旧 83.96 与当前严格协议 79.18 的提示协议差异。所有分支必须使用原 detection prompt 和同一 16 帧 index。

```bash
STAGE=vif_shared bash "$RUN"
STAGE=vif_no_motion bash "$RUN"
STAGE=vif_minor_motion bash "$RUN"
STAGE=vif_complex_motion bash "$RUN"
STAGE=compose_vif bash "$RUN"

cat /tmp/1res/camera_hard_route_gate/v1/vifbench/composed/camera_hard_route_gate.json
```

`compose_vif` 不重新推理模型，只对同一批四模型预测做三种选择：

- 共享 detection LoRA；
- 预测相机路由选择三个专家；
- 将每条预测路由循环映射到错误专家。

预注册通过条件是 coverage/格式均至少 99%，预测路由相对原始 base 和共享模型分别在 Balanced ACC 或 Fake F1 至少提高 0.5 点且另一项下降不超过 0.5 点，同时相对循环错误路由至少提高 1.0 点且另一项下降不超过 0.5 点。

通过只说明“相机条件化 specialization 有检测信号”，下一步才实现内部 soft gate/residual adapter；未通过则停止这个路由家族，不继续做同配方 RL。
