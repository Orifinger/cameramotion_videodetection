# 原生尺度 DINO 时序专家两层验证

本代码验证两个问题：

1. **时序因果能力门（Gate 1）**：相同 DataB 监督下，正确帧顺序专家是否稳定优于顺序不敏感专家和等容量乱序专家。
2. **Qwen 互补性门（Gate 2）**：若 Gate 1 成立，该专家能否用预注册固定融合改善强 Qwen 的 ViF-Bench Real/Fake 指标。

这不是旧 CTNE/RAFT 相机实验的重复。这里不使用 camera labels、camera caption、RAFT、光流或相机补偿，只使用冻结 DINOv2 patch tokens、真实帧顺序和最终 Real/Fake 监督。

## 数据边界

- DataB 完整使用 6766 条，不删除 GenBuster 原始 train/test 行。
- 帧数保持原样：11 帧 1 条、16 帧 6748 条、17 帧 17 条。
- fold 0 只选择 checkpoint 和阈值，fold 1-4 训练；同组样本不跨 fold。
- ViF-Bench 只作开发门，不在其标签上拟合阈值、路由器或融合权重。
- GenBuster Closed Benchmark 本轮完全不读取。两个 Gate 都通过后才进行一次最终测试。

## 特征与模型

- DINO 权重：/home/admin/dinov2-small
- 使用每条样本列出的全部帧，不默认 16 帧。
- 只做等比例下采样和 14 像素 patch 对齐，不裁剪。
- 每帧保存 CLS token 与 4x4 pooled patch tokens。
- static：跨帧 mean/max/variation 集合聚合。
- ordered：frame token、相邻帧差分和双向 GRU。
- shuffled：与 ordered 等架构、等数据、等优化器，只在每个 epoch 打乱帧序。
- 评测还会把 ordered 模型直接喂乱序帧。

## 快速预检

服务器 A：

~~~bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
STAGE=preflight bash scripts/forensic_temporal_expert_gate/run_server_a.sh
STAGE=build bash scripts/forensic_temporal_expert_gate/run_server_a.sh
STAGE=smoke bash scripts/forensic_temporal_expert_gate/run_server_a.sh
~~~

服务器 B：

~~~bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
STAGE=preflight bash scripts/forensic_temporal_expert_gate/run_server_b.sh
STAGE=build bash scripts/forensic_temporal_expert_gate/run_server_b.sh
STAGE=smoke bash scripts/forensic_temporal_expert_gate/run_server_b.sh
~~~

## 两台服务器并行执行

服务器 A 负责 DataB 特征与训练：

~~~bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
ROOT=/tmp/1res/forensic_temporal_expert_gate/v1/server_a
mkdir -p "$ROOT"

nohup env STAGE=all KEEP_ALIVE_AFTER_RUN=1 \
bash scripts/forensic_temporal_expert_gate/run_server_a.sh \
> "$ROOT/launcher.log" 2>&1 &

echo "server A pid: $!"
~~~

服务器 B 同时提取 ViF 特征；随后只等待 NAS 上的小模型，不依赖 A 的 /tmp：

~~~bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
ROOT=/tmp/1res/forensic_temporal_expert_gate/v1/server_b
mkdir -p "$ROOT"

nohup env STAGE=all KEEP_ALIVE_AFTER_RUN=1 \
bash scripts/forensic_temporal_expert_gate/run_server_b.sh \
> "$ROOT/launcher.log" 2>&1 &

echo "server B pid: $!"
~~~

持续查看：

~~~bash
watch -n 10 'bash scripts/forensic_temporal_expert_gate/progress.sh A'
~~~

~~~bash
watch -n 10 'bash scripts/forensic_temporal_expert_gate/progress.sh B'
~~~

## 核心结果

~~~bash
cat /input/workflow_58770161/workspace/test/cameramotion_det/res/forensic_temporal_expert_gate/v1/eval/forensic_temporal_expert_gate1_summary.json
cat /input/workflow_58770161/workspace/test/cameramotion_det/res/forensic_temporal_expert_gate/v1/eval/forensic_temporal_expert_gate2_summary.json
~~~

Gate 1 要求 ordered 相对 static 和 shuffled 至少一个主指标提升 1.5 点，另一个不下降超过 1 点；乱序输入后至少下降 1 点；2/3 seed 同方向；Real Recall 下降不超过 3 点。

Gate 2 固定使用：

~~~text
fused_logit = qwen_logit + 0.25 * ordered_expert_logit
~~~

它要求至少一个主指标提升 1.5 点、分组 bootstrap 下界大于 0、胜过全部控制融合且 Real Recall 下降不超过 1 点。若 Qwen confidence 分片缺失，结果会写成 conclusion_insufficient，不会拿硬答案冒充连续置信度。

## 存储

- 临时大特征：/tmp/1res/forensic_temporal_expert_gate/v1/
- 持久化小结果：/input/workflow_58770161/workspace/test/cameramotion_det/res/forensic_temporal_expert_gate/v1/

验证失败不上传大特征。通过并确定复用时才执行：

~~~bash
ossutil64 cp -r /tmp/1res/forensic_temporal_expert_gate/v1/server_a/datab_features/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/forensic_temporal_expert/v1/datab_features/
~~~

~~~bash
ossutil64 cp -r /tmp/1res/forensic_temporal_expert_gate/v1/server_b/vifbench_features/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/forensic_temporal_expert/v1/vifbench_features/
~~~
