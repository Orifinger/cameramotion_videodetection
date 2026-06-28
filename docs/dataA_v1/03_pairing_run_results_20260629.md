# Data A v1：domain / catalog / donor pairing 首次运行结果

> 运行日期：2026-06-29
>
> 本文记录 `init_video_domain_index.py`、Qwen3-VL 域补标、catalog/pair-pool 构建，以及首次 VACE stage-1 随机采样的实际输出。首次 15-case plan 是候选草案，不是已冻结的生成计划。

## 1. 视频域索引

```text
video_domain_index_v1.json：911 videos
```

构建 `track_editability_catalog_v1.json` 时没有缺失视频域标签：

```text
videos_missing_domain_labels: 0
missing_domain_track_count: 0
```

track 级视觉域分布：

| content_domain | tracks |
|---|---:|
| real_live_action | 2178 |
| cg_rendered | 561 |
| game_scene | 260 |
| animation_cartoon | 132 |
| unknown | 1 |

`unknown` 默认不进入 donor pairing。

## 2. Catalog 与对象归并

| 指标 | 数量 |
|---|---:|
| raw tracks | 3132 |
| catalog tracks | 3132 |
| eligible tracks | 3014 |
| 当前未进入 v1 主采样的 tracks | 118 |

当前 118 条未进入 v1 主采样的 track 对应 `animal`。这是当前规则的预期行为，不是构建失败。

semantic bucket 中：

```text
generic_unknown: 885
human: 1071
vehicle: 397
screen: 73
chair_stool: 99
table: 73
lamp: 61
box_package: 34
ball: 31
decor_object: 31
small_appliance: 25
toy_plush: 22
phone: 19
backpack_bag: 14
bottle_container: 14
book_notebook: 6
cup_mug: 6
```

`generic_unknown` 不自动参与 donor-driven `object_swap`，但可保留 text-only 的 attribute editing 候选；不需要为第一轮 VACE smoke 立即补完全部 885 条的语义 bucket。

## 3. 已构建的操作与 donor candidate pool

operation attachment 数量：

| operation | attached tracks |
|---|---:|
| object_swap | 1717 |
| object_attribute_edit | 1717 |
| object_removal | 1183 |
| person_appearance_swap | 1071 |
| surface_content_edit | 223 |
| surface_attribute_edit | 204 |

同域、同类 donor candidate pair 数量：

| operation | candidate directed pairs |
|---|---:|
| person_appearance_swap | 12852 |
| object_swap | 9112 |
| surface_content_edit | 2300 |

结论：v1 主线的 object/person/surface donor pairing 候选池已经足够，不需要继续做对象发现或扩大 donor pool 才能开始 VACE smoke。

## 4. 首次 VACE stage-1 随机 sampling 输出

命令生成：

```text
res/dataA_v1/plans/vace14b_stage1_plan.json
```

首次随机 15 case 的实际分布：

| operation | count |
|---|---:|
| object_swap | 9 |
| surface_content_edit | 3 |
| person_appearance_swap | 2 |
| surface_attribute_edit | 1 |
| object_attribute_edit | 0 |

对应 route：

```text
vace14b_object_reference_swap: 9
vace14b_surface_reference_content: 3
vace14b_person_reference_swap: 2
vace14b_surface_attribute_text: 1
```

### 决策

这份 plan 可以保留为随机采样基线，但**不能作为最终 VACE stage-1 smoke plan**：它缺少 `object_attribute_edit`，且对象替换占 9/15，无法覆盖 VACE 的主要输入模式。

建议 VACE stage-1 使用固定 smoke quota：

```text
object_swap              5
person_appearance_swap   3
surface_content_edit     3
object_attribute_edit    2
surface_attribute_edit   2
---------------------------
total                   15
```

该 quota 只用于接口、mask、reference、局部保真和输出编码的 smoke 覆盖；不等于 v1 的最终数据分布。

## 5. 进入生成会话前的唯一剩余门槛

对象发现、类别归并、domain pairing 与 donor pair pool 已完成。当前还不能直接把随机 plan 投入生成：需要在当前阶段完成 15 条 case 的可视化审阅和 plan 冻结。

每条最终 case 至少需要补齐：

```text
- target / donor 的可视化确认
- target clip 的具体时间范围
- target 的 source description
- target edit description / prompt
- donor reference 最佳帧与 crop 策略
- case 是否保留 / 替换 donor / 重采样
- target bbox tube 保存与后续证据 bbox 归一化策略
```

`bbox_tube_xywh` 已随 track metadata 进入 catalog/pair plan。生成时仍以非矩形 mask tube 为主要控制输入；bbox tube 后续用于 ROI、可视化审阅与检测输出的 evidence slot。
