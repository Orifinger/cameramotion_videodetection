# 静止/有运动二路硬路由检测专家执行说明

本实验只回答一个问题：冻结的视觉相机 Router 能否通过选择静止/有运动 detection 专家，提高 ViF-Bench 最终 `Real/Fake` 指标。检测推理不接收任何 camera 文本。

## 入口与固定路径

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/camera_binary_route_gate/run.sh
```

默认输入：

- detection 起点：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`
- 已通过的二路门：`/tmp/1res/camera_hard_route_gate/v1/routes/dataa_binary_route_summary.json`
- 冻结 Router：`/tmp/1res/camera_hard_route_gate/v1/train/router`
- 已有三桶检测数据：`/tmp/1res/camera_hard_route_gate/v1/data`
- 新工作目录：`/tmp/1res/camera_binary_route_gate/v1`

## 先做预检、数据构建和两步 smoke

```bash
STAGE=preflight bash "$RUN"
STAGE=build bash "$RUN"

cat /tmp/1res/camera_binary_route_gate/v1/data/camera_binary_route_data_summary.json
cat /tmp/1res/camera_binary_route_gate/v1/data/llamafactory_install_summary.json

STAGE=smoke bash "$RUN"
tail -n 10 /tmp/1res/camera_binary_route_gate/v1/smoke/trainer_log.jsonl
```

构建必须满足：共享数据等于静止与有运动专家的不重叠精确并集；三个分支均 Real/Fake 等量；`motion` 只由原 `minor-motion + complex-motion` 组成；检测 prompt 中没有 camera 文本。任一项失败都不开始正式训练。

## 正式训练

一台 16 GPU 服务器串行训练：

```bash
STAGE=train_all KEEP_ALIVE_AFTER_RUN=1 bash "$RUN"
```

也可在独立服务器分别运行：

```bash
STAGE=train_shared bash "$RUN"
STAGE=train_no_motion bash "$RUN"
STAGE=train_motion bash "$RUN"
```

输出目录：

- 共享模型：`/tmp/1res/camera_binary_route_gate/v1/train/shared`
- 静止专家：`/tmp/1res/camera_binary_route_gate/v1/train/no_motion`
- 有运动专家：`/tmp/1res/camera_binary_route_gate/v1/train/motion`

训练完成后，这三个正式 adapter 将用于完整 ViF 推理，应在容器退出前上传：

```bash
ossutil64 cp -r /tmp/1res/camera_binary_route_gate/v1/train/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_binary_route_gate/v1/train/
```

冻结 Router 也会用于 ViF route 打分，若尚未保存到 OSS：

```bash
ossutil64 cp -r /tmp/1res/camera_hard_route_gate/v1/train/router/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_binary_route_gate/v1/router/
```

## 冻结 Router 并生成 ViF 二路 manifest

```bash
STAGE=build_vif_route bash "$RUN"

cat /tmp/1res/camera_binary_route_gate/v1/routes/vifbench_binary_route_summary.json
```

该阶段对 ViF 检测 index 的同一 16 帧计算原三道相机问题分数，然后使用已经冻结的规则把 top-1 映射成 `no-motion` 或 `motion`。ViF 没有 gold camera 标签，因此只审计覆盖率、两路分布、real/fake route 分布差异和 pair consistency；禁止根据 ViF detection 标签修改映射。

## 三个分支的完整 ViF 推理

顺序执行：

```bash
STAGE=vif_shared bash "$RUN"
STAGE=vif_no_motion bash "$RUN"
STAGE=vif_motion bash "$RUN"
STAGE=compose_vif bash "$RUN"
```

也可以一次串行跑完并在结束后进入 keep-alive：

```bash
STAGE=vif_all KEEP_ALIVE_AFTER_RUN=1 bash "$RUN"
```

共享阶段会在相同协议下同时重跑原始 detection checkpoint；两个专家阶段只推理对应 adapter。离线合成不再加载模型，生成共享、正确路由和交换错误路由三套预测。

最终只需读取：

```bash
cat /tmp/1res/camera_binary_route_gate/v1/vifbench/composed/camera_binary_route_gate.json
```

## 最终门

- 四个条件 coverage 和格式有效率均至少 99%；
- 正确路由相对原始模型和共享模型，Balanced ACC 或 Fake F1 至少提高 0.5 点，另一项下降不超过 0.5 点；
- 正确路由相对交换错误路由至少提高 1.0 点，另一项下降不超过 0.5 点；
- 同时报告逐生成器 Balanced ACC 胜率与 ViF real/fake route 分布差异。

只有全部通过才能说明 camera route 与最终检测发生了可验证耦合。失败时停止硬路由，不通过调整 ViF route、改门槛或追加同方向 RL 追分。
