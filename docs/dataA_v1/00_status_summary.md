# Data A v1 阶段汇总：CameraBench 局部 AIGC 编辑数据构造

> 更新时间：2026-06-29
>
> 本文记录已完成资产、关键统计、工程位置、当前未完成项与后续约束。它是 Data A v1 配对局部编辑数据构造的阶段事实汇总，不把尚未 smoke 的模型能力写成既成事实。

---

## 1. 项目目标与 Data A 定义

目标：构建相机运动感知的视频 AIGC 检测数据与模型。Data A 采用 **paired Real–Fake** 设计：

- **Real**：CameraBench 源视频中的原始片段；
- **Fake**：在同一源视频、同一时间段、同一 focus region 上施加局部 AIGC 编辑得到；
- 尽量保持源场景、相机运动、非编辑区域、视频编码设置一致；
- 强监督包括 mask tube、bbox tube、时间范围、编辑类型、生成器与相机运动标签。

核心原则：**相机运动是局部真实性检查的上下文条件，不是 Fake 的标签捷径。**

---

## 2. 已完成：Qwen3 候选发现 + SAM3 跨帧轨迹构造

### 2.1 运行名称与总体结果

SAM3 运行：

```text
sam3_v4_20260627T175213Z
```

统计：

| 指标 | 数量 |
|---|---:|
| 输入视频数 | 1366 |
| 成功处理视频 | 1197 |
| 无稳定 track 视频 | 151 |
| 失败视频 | 18 |
| 原始 track 总数 | 6375 |
| quality-pass track 总数 | 3132 |
| 至少有 1 条 quality-pass track 的视频 | 911 |
| `physical_instance` | 2906 |
| `editable_surface` | 226 |

### 2.2 quality-pass track 的候选类别分布

| candidate_class | 数量 |
|---|---:|
| bounded_object | 1183 |
| human | 1071 |
| vehicle | 397 |
| handheld_object | 137 |
| sign_or_poster | 128 |
| animal | 118 |
| display_screen | 73 |
| framed_art | 15 |
| paper_book_map | 7 |
| apparel_panel | 3 |

当前 track bank 是“可编辑区域资源池”，不是最终生成样本清单。

---

## 3. 已有资产与持久化位置

项目仓库：

```text
/input/workflow_58770161/workspace/test/cameramotion_det
```

稳定 JSON 清单目录：

```text
res/sam_track_bank/
├── sam3_tracks_all.json
├── sam3_quality_tracks.json
├── sam3_quality_tracks_enriched.json
├── sam3_failures.json
└── sam3_run_summary.json
```

Data A v1 后续以以下文件为主输入：

```text
res/sam_track_bank/sam3_quality_tracks_enriched.json
```

其重要字段包括：

```text
video_id
video_path
candidate_id
track_id
region_family
candidate_class
canonical_concept
display_phrase
sam_prompt
mask_tube_path
bbox_tube_xywh
track_quality_score
```

### 3.1 mask tube

原始 mask tube 位于临时路径：

```text
/tmp/cambench_train/cam_train/object_discovery_sam/track_masks_v1/...
```

每个 `.npz` 应包含：

```text
frame_indices: int32 [N_visible]
masks: uint8 [N_visible, H, W]
```

**注意**：`/tmp` 不是长期可靠位置。批量生成前必须确认 `track_masks_v1` 已完整备份到 OSS，并在后续 manifest 中将临时路径映射为可持久访问路径。当前不能假定 OSS 上传已完成。

---

## 4. 当前冻结的总体数据构造逻辑

### 4.1 source–donor pairing

对于 donor-driven 操作：

```text
目标视频 A：提供 target video、target track、mask tube、原始相机运动和背景
供体视频 B：提供 donor track / donor reference crop
生成结果：Fake_A
```

- `Real_A` 始终是 A 的原始片段；
- `Fake_A` 始终在 A 的局部区域中由生成模型产生；
- B 仅作为生成条件/reference，不与 Fake_A 构成真假 pair；
- 不允许把 B 的像素直接复制粘贴进 A。

### 4.2 配对硬约束

```text
同 content_domain
+ 同 coarse_group
+ object 时同 semantic_bucket 优先
+ surface 时同 surface_subtype 优先
+ target 与 donor 不得来自同一视频
```

示例：

```text
真实场景杯子 ↔ 真实场景杯子
动画人物 ↔ 动画人物
游戏车辆 ↔ 游戏车辆
screen ↔ screen
sign/poster ↔ sign/poster
```

### 4.3 视频视觉域标签

所有视频都适用以下 `content_domain`，它不等于“视频里有没有真人”：

```text
real_live_action
animation_cartoon
game_scene
cg_rendered
mixed
unknown
```

该标签只用于避免跨视觉域 donor pairing，例如避免真人视频中的车与动画车直接配对。

---

## 5. 当前代码资产（已推送至 GitHub main）

仓库：`Orifinger/cameramotion_videodetection`

```text
scripts/init_video_domain_index.py
scripts/label_video_domains_qwen_vllm.py
scripts/build_dataa_v1_catalog_and_pairs.py
scripts/sample_dataa_v1_generation_plan.py
```

作用：

| 脚本 | 功能 |
|---|---|
| `init_video_domain_index.py` | 从现有 quality track 建立视频级视觉域补标骨架 |
| `label_video_domains_qwen_vllm.py` | 直接请求既有 vLLM Qwen3-VL server，补充视频级视觉域标签；不重做对象发现 |
| `build_dataa_v1_catalog_and_pairs.py` | 生成 generator/operation/rule registry、track editability catalog、donor pair pool 与统计 |
| `sample_dataa_v1_generation_plan.py` | 按操作权重、质量、pair score、复用上限采样最终生成执行计划 |

---

## 6. 当前尚未完成的工作

1. 通过既有 vLLM Qwen3-VL server 对有效视频补充 `content_domain/style_domain`；
2. 由脚本生成并检查 `track_editability_catalog_v1.json`；
3. 生成并检查 `donor_pair_pool_v1.json`；
4. 根据统计补充/修正 semantic bucket 词表；
5. 按权重抽样，形成每个 case 已确定 target、operation、donor、route 的 `generation_plan_v1.json`；
6. 对 face route 增加独立 face detection / landmark / parsing track；
7. 对 insertion route 增加 placement-site tube；
8. 确认 mask OSS 备份和路径重写策略；
9. 进入新会话进行 VACE / VideoPainter / PISCO / FaceFusion 的环境、smoke、参考图 materialization 和正式视频生成。

---

## 7. 重要工程约束

- 服务器的清理约束是：**总 GPU 计算利用率长期低于 30% 可能导致作业被清理**；不是 VRAM 使用率必须低于 30%。
- 当前 Qwen3-VL 使用既有 vLLM server，不使用外部 API。
- `--workers` 是客户端请求并发，不是 GPU 数量。TP=16 的单实例可以连续 batch 多个请求；实际吞吐仍受服务端 scheduler、图像 token、`max_num_seqs`、`max_num_batched_tokens`、请求超时和网络/CPU 编码开销约束。
- 用 4 帧、短输出做视频域标注时可尝试 `--workers 40`。先在几十条视频上观察超时、HTTP 5xx、GPU 显存和吞吐；若稳定再跑完。若 server-side `max_num_seqs` 明显小于 40，更多 worker 只会增加排队，不会获得等比例吞吐。

---

## 8. 本阶段完成判据

在开启“03｜局部编辑生成执行”新会话前，本阶段应保存：

```text
video_domain_index_v1_labeled.json
generator_registry_v1.json
operation_registry_v1.json
candidate_operation_rules_v1.json
track_editability_catalog_v1.json
donor_pair_pool_v1.json
pairing_stats_v1.json
generation_plan_v1.json
```

其中 `generation_plan_v1.json` 中每个 case 至少必须明确：

```text
target video / target track
operation
generator route
donor track（若需要 donor）
operation / route 权重来源
状态、随机种子、后续 reference materialization 状态
```
