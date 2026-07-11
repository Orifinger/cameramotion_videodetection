# Gate 1 局部配对 DPO 执行说明

- 日期：2026-07-11
- 状态：代码完成，训练前基线与 DPO smoke 待执行
- 目标：验证模型能否从 DataA 同源 Real/Fake 双顺序配对中学会选择局部编辑视频并定位真实 VACE mask。

## 1. 为什么使用 LlamaFactory

现有偏好数据是 LlamaFactory 支持的 OpenAI/ShareGPT 多模态偏好格式：

```text
messages + images + chosen + rejected
```

已有 LlamaFactory 环境即可运行 Qwen3-VL LoRA-DPO，不需要另装 ms-swift。训练使用 LoRA，reference 是关闭 adapter 后的初始 detection checkpoint；训练完成后合并 adapter，继续使用现有 `infer_dataa.py` 推理。

## 2. 数据和规模

```text
训练：/tmp/1res/counterfactual_gate/data/dataa_counterfactual_dpo_local_only.json
评测：/tmp/1res/counterfactual_gate/data/dataa_counterfactual_eval_local_only.json
初始模型：/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115
```

- 755 个 train case；
- 每个 case 有 real-first 和 fake-first 两种顺序；
- 每种顺序有错误视频选择、错误位置两类 rejected；
- 共 3020 条偏好记录；
- held-out 321 个 case，共 642 条双顺序评测记录。

## 3. 服务器预检

```bash
ROOT=/input/workflow_58770161/workspace/test/cameramotion_det
LF=/input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory

cd "${ROOT}"
git pull origin main

cd "${LF}"
llamafactory-cli version
python -c 'import peft, trl, deepspeed; print("DPO dependencies OK")'
```

不要主动升级 LlamaFactory、Transformers 或 qwen-vl-utils。当前环境已经跑通过 Qwen3-VL SFT，先用 smoke 检查当前版本的多模态 DPO。

## 4. 注册并审计数据

训练脚本会自动执行注册；也可以单独运行：

```bash
cd "${ROOT}"
python tools/install_gate1_llamafactory_data.py \
  --source-json /tmp/1res/counterfactual_gate/data/dataa_counterfactual_dpo_local_only.json \
  --llamafactory-data-dir "${LF}/data" \
  --smoke-samples 64 \
  --seed 20260711 \
  --check-image-files
```

预期输出：3020 条完整数据、64 条四种条件配平的 smoke 数据，且 `all_images_exist=true`。脚本用结构化 JSON 更新 `dataset_info.json`，不会删除原有数据集条目。

## 5. 训练前基线

```bash
cd "${ROOT}"

MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
MODEL_NAME=Qwen3-VL-8B-gate1-pair-baseline \
SAVE_DIR=/tmp/1res/gate1_pair_eval/Qwen3-VL-8B-gate1-pair-baseline \
bash eval/infer_dataa_counterfactual_pair_gate.sh

MODEL_NAME=Qwen3-VL-8B-gate1-pair-baseline \
PRED_JSON=/tmp/1res/gate1_pair_eval/Qwen3-VL-8B-gate1-pair-baseline \
bash eval/run_dataa_counterfactual_pair_gate.sh
```

基线预计不通过 Gate 1，但必须保留它的选择准确率、预测 A 比例、swap consistency 和 bbox IoU，作为 DPO 前对照。

## 6. 两步 DPO smoke

```bash
cd "${ROOT}"

TRAIN_MODE=smoke \
LLAMAFACTORY_ROOT="${LF}" \
MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
OUTPUT_DIR=/tmp/1res/gate1_pair_dpo_smoke \
bash scripts/gate1/run_llamafactory_pair_dpo.sh
```

Smoke 只检查：数据能否解析、16 张图片是否与 `<image>` 对齐、chosen/rejected 是否正确进入 DPO、16 卡能否完成 forward/backward、loss 是否有限且无 OOM/NaN。两步 loss 的涨跌不作为方法结论。

## 7. 完整一轮 DPO

Smoke 通过后执行：

```bash
cd "${ROOT}"

TRAIN_MODE=full \
LLAMAFACTORY_ROOT="${LF}" \
MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
OUTPUT_DIR=/tmp/1res/gate1_pair_dpo_local_only \
bash scripts/gate1/run_llamafactory_pair_dpo.sh
```

默认配置：LoRA rank 16、`pref_beta=0.1`、sigmoid DPO、学习率 `5e-6`、1 epoch、每卡 batch 1、梯度累积 1、16 卡全局 batch 16、视觉像素上限 262144、ZeRO-2。3020 条数据预计约 189 个 optimizer step，`save_steps=95`。

## 8. 合并 LoRA

完整训练结束后，优先使用输出根目录；若其中没有 `adapter_model.safetensors`，将 `ADAPTER_PATH` 改为最后一个 checkpoint：

```bash
cd "${ROOT}"

MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
ADAPTER_PATH=/tmp/1res/gate1_pair_dpo_local_only \
MERGED_MODEL_DIR=/tmp/1res/gate1_pair_dpo_local_only_merged \
LLAMAFACTORY_ROOT="${LF}" \
bash scripts/gate1/merge_llamafactory_pair_dpo.sh
```

## 9. DPO 模型推理与验收

```bash
cd "${ROOT}"

MODEL_PATH=/tmp/1res/gate1_pair_dpo_local_only_merged \
MODEL_NAME=Qwen3-VL-8B-gate1-pair-dpo \
SAVE_DIR=/tmp/1res/gate1_pair_eval/Qwen3-VL-8B-gate1-pair-dpo \
bash eval/infer_dataa_counterfactual_pair_gate.sh

MODEL_NAME=Qwen3-VL-8B-gate1-pair-dpo \
PRED_JSON=/tmp/1res/gate1_pair_eval/Qwen3-VL-8B-gate1-pair-dpo \
FAIL_ON_GATE=1 \
bash eval/run_dataa_counterfactual_pair_gate.sh
```

Gate 1 要求：选择准确率不低于 70%、swap consistency 不低于 85%、预测 A 比例在 45% 至 55%、mean bbox IoU 不低于 0.30、选择与 bbox 格式正确率均不低于 95%。

## 10. 当前停止点

先返回以下两份结果：

```text
训练前 baseline 的 dataa_counterfactual_pair_gate_summary.json
DPO 模型的 dataa_counterfactual_pair_gate_summary.json
```

Gate 1 未通过时不启动 GRPO。Gate 1 通过后才进行普通 DataA/VIF-Bench 检测迁移，并准备等步数 detection replay 对照。
