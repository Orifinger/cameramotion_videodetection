# Data A v1 当前计划：生成模型、编辑方式、配对与 JSON 计划

> 更新时间：2026-06-29
>
> 本文是当前 v1 的规则说明。模型是否进入批量生成由 smoke test 决定；未通过 smoke 的 route 保留在 registry，但保持 `enabled=false`。

---

## 1. v1 目标

Data A v1 不是把所有 3132 条 track 全部生成，而是建立一套可复现的：

```text
track bank
→ 可编辑操作 catalog
→ donor candidate pair pool
→ 按权重采样 generation plan
→ 新会话执行局部视频生成与 QA
```

每个 Fake 只允许 **一个主要局部 AIGC 操作**。不使用传统贴图、逐帧 Photoshop、复制粘贴对象或全局重生成。

允许的后处理只有：将 AIGC 模型的局部结果通过时空 alpha mask 回贴，以保证 mask 外的真实背景和相机运动尽量保持不变。

---

## 2. v1 生成模型分工

| 模型 | v1 负责的操作 | 输入条件 | 当前状态 |
|---|---|---|---|
| Wan2.1-VACE-14B | 人物外观替换、物体替换、物体属性编辑、平面内容编辑、平面属性编辑 | target video + SAM3 mask tube + prompt；reference route 还需 donor reference | 主线；先 smoke |
| VideoPainter | 物体移除 / 局部背景补全 | target video + SAM3 mask tube | 保留；先 smoke |
| PISCO-14B | 物体新增 / 实例插入 | placement-site tube + donor object reference | 延后；需要 placement-site pipeline 和 smoke |
| FaceFusion | 人脸身份替换 | target face track + donor face | 延后；需要 face detection / landmark / parsing track 和 donor 合规管理 |

### 2.1 不把模型能力写死

- VACE 是 v1 主模型，但首先只对其适合的操作做最小 smoke；
- VideoPainter、PISCO、FaceFusion 只在相应输入管线就绪并 smoke 通过后才开启；
- 当前 registry 应记录 `enabled`、`smoke_status`、`requires_sam3_mask`、`requires_face_track`、`requires_placement_site`，而不是假设每条 route 都已可批量生产。

---

## 3. 固定的编辑方式

| operation | 主要模型 | 是否直接复用现有 SAM3 track | 是否需要 donor |
|---|---|---:|---:|
| `face_identity_swap` | FaceFusion | 否 | 是 |
| `person_appearance_swap` | VACE-14B | 是，`human` | reference route 需要 |
| `object_swap` | VACE-14B | 是 | reference route 需要 |
| `object_attribute_edit` | VACE-14B | 是 | 通常不需要 |
| `surface_content_edit` | VACE-14B | 是 | reference route 需要 |
| `surface_attribute_edit` | VACE-14B | 是 | 通常不需要 |
| `object_removal` | VideoPainter | 是 | 否 |
| `object_insertion` | PISCO-14B | 否 | 是 |

说明：

- `face_identity_swap` 与 `person_appearance_swap` 不同：前者只针对脸部身份，后者针对人物服装、身体级外观或整体 appearance；
- `object_insertion` 不应复用“对象 track”。它需要跨帧稳定的空白落点/支撑面，即 placement-site tube；
- `object_removal` 不需要 donor，它是利用 source video 背景上下文补全 mask 区域；
- `object_attribute_edit` / `surface_attribute_edit` 可以只用 text prompt，例如颜色、材质、花纹、车身涂装或布料图案变化。

---

## 4. Qwen3 + SAM3 结果如何继续使用

现有 Qwen3 + SAM3 的对象发现不重跑。

第一次 Qwen3-VL 已经回答：

```text
视频里有什么可编辑对象/表面？
```

并经 SAM3 形成跨帧 mask tube 与质量结果。

本阶段只增加两类补充：

```text
1. 视频级：content_domain / style_domain
2. 少量模糊 track：semantic bucket 或编辑风险的定向补标
```

第二次 Qwen3-VL 不重新发现候选对象。它只回答：

```text
这段视频属于真人、动画、游戏还是 CG？
已有对象是否可作为同域 donor pairing 的候选？
```

---

## 5. candidate class → 可编辑操作映射

| Qwen/SAM3 candidate_class | 主要操作 | pairing 规则 |
|---|---|---|
| `human` | `person_appearance_swap` | human ↔ human，且同 visual domain |
| `bounded_object` | `object_swap`、`object_attribute_edit`、低权重 `object_removal` | 同 semantic bucket 优先 |
| `handheld_object` | `object_swap`、`object_attribute_edit` | 同 semantic bucket；手部强接触时后续降权/拒绝 |
| `vehicle` | `object_attribute_edit`、少量 `object_swap` | vehicle ↔ vehicle，且同 visual domain |
| `display_screen` | `surface_content_edit`、`surface_attribute_edit` | screen ↔ screen |
| `sign_or_poster` | `surface_content_edit`、`surface_attribute_edit` | sign/poster ↔ sign/poster |
| `framed_art` | `surface_content_edit` | framed visual ↔ framed visual |
| `paper_book_map` | `surface_content_edit` | paper/map ↔ paper/map |
| `apparel_panel` | `surface_attribute_edit` | apparel surface；不单设配额 |
| `animal` | v1 首轮不作为主采样池 | 仅 future / 人工严格审核后开放 |

### 5.1 对象 semantic bucket

`bounded_object` 与 `handheld_object` 需要把已有 `canonical_concept` 归入 v1 的小型超类：

```text
cup_mug
bottle_container
book_notebook
backpack_bag
ball
toy_plush
box_package
phone
chair_stool
lamp
small_appliance
table
decor_object
vehicle
```

规则：

```text
同 bucket 配对优先
近邻 bucket 只能在人工允许时开放
generic_unknown 不自动进入 donor-based object_swap
```

`generic_unknown` 仍可保留 text-only attribute edit，避免因对象语义不确定而强配 donor。

---

## 6. 视觉域与 donor pairing 规则

### 6.1 视频级 visual domain

```text
real_live_action
animation_cartoon
game_scene
cg_rendered
mixed
unknown
```

该字段适用于有人和无人的视频；它只约束 donor 与 target 是否属于相近视觉域。

### 6.2 基础硬规则

```text
1. target video != donor video
2. content_domain 必须相同；unknown 默认不做 donor pairing
3. human ↔ human
4. face ↔ face（需要独立 face pipeline）
5. object ↔ 同 semantic_bucket object
6. screen ↔ screen；sign/poster ↔ sign/poster；其他 surface 同 subtype 优先
7. 一个 target source video 默认最多产生 1 条正式 Fake
8. 一个 donor track 默认最多服务 3 个正式 case
```

### 6.3 donor 的角色

```text
A = target source video
B = donor source video

Real_A = A 的原始片段
Fake_A = 在 A 的 mask tube 内由生成模型编辑出的片段
B 只提供参考对象 / 人物外观 / 人脸身份 / 平面内容
```

B 不是 Fake_A 的 Real pair，也不能直接把 B 的像素粘贴到 A。

---

## 7. 初始全局操作权重

以下是当前脚本中用于采样的初始分布；它们是 **可调整配置**，不是最终论文配额。

```json
{
  "face_identity_swap": 0.16,
  "person_appearance_swap": 0.14,
  "object_swap": 0.25,
  "object_attribute_edit": 0.10,
  "surface_content_edit": 0.15,
  "surface_attribute_edit": 0.07,
  "object_insertion": 0.07,
  "object_removal": 0.06
}
```

含义：

```text
替换类（face + person + object）为主；
表面内容和属性编辑提供不同类型局部改写；
新增和移除均保留，但不主导数据。
```

当前阶段的实际执行应遵循：

```text
未 smoke 的 route 不参与正式 sampling；
face / insertion 因输入管线尚未具备而先不进入第一版正式 plan；
VACE smoke 优先覆盖 object / person / surface / attribute；
removal 等 VideoPainter smoke 通过后单独开启。
```

---

## 8. 概率采样逻辑

对 track `i` 和它支持的 operation `o`：

```text
w(i, o)
= global_operation_weight(o)
× class_compatibility(i, o)
× track_quality(i)
× editability_gate(i, o)
× risk_penalty(i, o)
```

track 总权重：

```text
track_total_weight(i) = Σ_o w(i, o)
```

但正式采样不能简单在全部 track 上按总权重抽，否则 `human` 和 `bounded_object` 会压过小类。正确顺序：

```text
先按全局 operation 权重选择操作类别
→ 在支持该操作的 target tracks / donor pairs 中按质量与兼容性抽样
→ 在该操作允许的 route 中按 route 权重抽样
→ 应用 target video 最大使用次数、donor 最大复用次数
```

reference route 与 text-only route 的初始比例：

```text
object_swap:
  vace14b_object_reference_swap 0.70
  vace14b_object_text_swap      0.30

person_appearance_swap:
  vace14b_person_reference_swap    0.75
  vace14b_person_text_appearance   0.25

surface_content_edit:
  vace14b_surface_reference_content 0.80
  vace14b_surface_text_content      0.20

attribute edit:
  text route       0.60
  reference route  0.40
```

---

## 9. 需要产出的 JSON

### 9.1 视频域索引

```text
video_domain_index_v1.json
video_domain_index_v1_labeled.json
```

### 9.2 三个 registry

```text
generator_registry_v1.json
operation_registry_v1.json
candidate_operation_rules_v1.json
```

### 9.3 每条 track 的编辑能力 catalog

```text
track_editability_catalog_v1.json
```

每条 track 需要写入：

```text
content_domain
style_domain
semantic_bucket / surface_subtype
editable_operations
operation_weight
route_candidates
track_total_weight
risk / eligibility 状态
```

### 9.4 donor 候选池

```text
donor_pair_pool_v1.json
```

一个 target 可保留多个兼容 donor。最终 sampling 才选择其中一个。

### 9.5 最终执行计划

```text
generation_plan_v1.json
```

每条 case 必须固定：

```text
case_id
target video / target track
operation
generator route
donor track（如果需要）
reference-frame strategy
随机种子
status
```

`reference crop` 的实际抽取应在 plan 固定后做：从 donor mask tube 选择对象面积大、遮挡少、贴边少、画面清晰的帧，再生成去背景/白底 reference。

---

## 10. 当前阶段执行顺序

```text
Step 1  生成 video_domain_index 骨架
Step 2  用既有 vLLM Qwen3-VL server 补视频级视觉域
Step 3  构建 registry、track editability catalog、donor pair pool
Step 4  查看 pairing_stats，补 semantic bucket 规则；必要时只对模糊 track 定向 Qwen3 补标
Step 5  固定可进入本轮 smoke 的 route
Step 6  按权重生成 generation_plan_v1.json，并人工 review 计划内容
Step 7  保存 plan，开启新 03 会话进入 reference materialization、模型安装/smoke、视频生成和 QA
```

本会话在 Step 6 完成后结束；视频生成本身不在本会话继续展开。
