# 交接计划：从 Data A v1 编辑计划到 03｜局部视频生成执行

> 使用时机：本会话已完成 `generation_plan_v1.json` 的人工审阅与冻结后。
>
> 新 03 会话负责模型环境、reference materialization、局部视频生成、回贴、QA 和 accepted/rejected 管理；不再重做候选发现、视觉域分类、pairing 或全局采样策略。

---

## 1. 新会话的输入必须已具备

在开启新 03 会话前，以下文件应已经生成并保存：

```text
res/dataA_v1/registries/
├── video_domain_index_v1_labeled.json
├── generator_registry_v1.json
├── operation_registry_v1.json
├── candidate_operation_rules_v1.json
├── track_editability_catalog_v1.json
├── donor_pair_pool_v1.json
└── pairing_stats_v1.json

res/dataA_v1/plans/
└── generation_plan_v1.json
```

此外必须确认：

```text
1. 所有 plan 中使用的 mask tube 均能通过 OSS 或持久路径访问；
2. 每个 case 的 target video 路径可读取；
3. 各 route 的 enabled/smoke_status 已在 generator registry 中更新；
4. 尚未通过 smoke 的 route 不在正式 batch plan 内。
```

---

## 2. 当前已经完成、不要在新会话重做的工作

```text
Qwen3 候选对象/表面发现
SAM3 跨帧实例轨迹和 mask tube
视频级 content_domain / style_domain 补标
candidate_class → editable operation 规则
semantic bucket / surface subtype 归类
source–donor candidate pairing
全局 operation 权重与 route 权重
正式 generation plan 抽样与人工审核
```

新 03 会话以 frozen `generation_plan_v1.json` 为唯一 case 输入。任何要修改操作比例、配对规则或换 donor 的需求，应回到本阶段重新生成/版本化 plan，而不是在生成代码中临时改逻辑。

---

## 3. 新 03 会话的职责

### 3.1 先做每个 route 的最小 smoke

推荐顺序：

```text
1. VACE-14B
   - object_swap
   - person_appearance_swap
   - surface_content_edit
   - object_attribute_edit / surface_attribute_edit

2. VideoPainter
   - object_removal

3. FaceFusion
   - face_identity_swap（前提：face track / face parsing / donor 合规都已具备）

4. PISCO-14B
   - object_insertion（前提：placement-site tube 已具备）
```

每个 route 先做 2–3 个 case，不要直接批量。

### 3.2 donor reference materialization

对 `*_reference_*` route：

```text
读取 donor video 与 donor mask tube
→ 从 visible frames 中选对象面积大、遮挡低、贴边少、图像清晰的参考帧
→ crop + mask 去背景 / 白底化
→ 保存 donor reference image / alpha mask
→ 更新 case manifest 的 reference_materialization
```

规则：

```text
参考图不是传统粘贴素材；
它只是 VACE / PISCO / FaceFusion 的生成条件；
最终 Fake 仍应由生成模型在 target A 的时空区域内产生。
```

### 3.3 统一 case input package

每条 case 生成：

```text
source clip
mask_raw
mask_edit
mask_gen
mask_alpha
target metadata
donor reference（若需要）
prompt / target description
model route config
```

mask 约定：

```text
M_raw   = SAM3 原始 mask
M_edit  = 根据操作的适度膨胀/清理 mask
M_gen   = 传给编辑模型的控制 mask
M_alpha = 最终 soft composite mask
```

最终回贴：

```text
Fake = M_alpha * Generated + (1 - M_alpha) * Real
```

目的不是伪造非 AIGC 内容，而是严格保持编辑区外的真实背景、相机运动和编码条件，降低全局重生成泄漏。

### 3.4 输出与目录约定

建议：

```text
res/dataA_v1/
├── assets/
├── references/
├── attempts/
│   └── <case_id>/
│       ├── source_real_raw.mp4
│       ├── input_mask_raw.mp4 / npz
│       ├── donor_reference.png
│       ├── generated_raw.mp4
│       ├── fake_pair_render.mp4
│       ├── case_manifest.json
│       └── logs/
├── accepted/
│   └── <case_id>/
├── rejected/
│   └── <case_id>/
└── qa/
```

每个 accepted pair 应保存：

```text
source_real_raw.mp4
real_pair_render.mp4
fake_pair_render.mp4
mask_raw / mask_edit / mask_gen / mask_alpha
case_manifest.json
model checkpoint / repo commit / seed / prompt / route
QA 结果
```

### 3.5 paired 渲染约束

`Real` 与 `Fake` 必须尽量匹配：

```text
fps
frame count
resolution
clip time range
video codec
bitrate / encoding pipeline
audio policy
```

否则下游检测器可能学到编码差异，而非局部 AIGC 编辑痕迹。

---

## 4. QA 与收录原则

### 4.1 自动 QA（最低要求）

```text
帧数、fps、分辨率一致
输出可解码
编辑区域与计划 mask 有重叠
mask 外像素差异不过大
无大面积全局重绘
无明显黑帧 / 花屏 / 崩帧
```

### 4.2 人工快速审核

检查：

```text
目标编辑是否真的发生
局部对象是否符合目标操作
是否出现全局内容漂移
是否严重破坏相机运动与背景
是否因为 donor/target 风格不匹配而显得极端不自然
是否有可见、可复核的局部生成现象
```

结果至少区分：

```text
accepted
rejected_generation_failure
rejected_global_drift
rejected_wrong_edit
rejected_pairing_mismatch
rejected_low_visibility
needs_manual_review
```

已知编辑 mask 可用于 focus region 和监督；不能因为“这里编辑过”就自动捏造 explanation。解释数据只收录具有可见、可复核现象的 accepted Fake。

---

## 5. 新会话的首条交接指令

将以下内容复制到新会话即可：

```text
项目：相机运动感知局部视频 AIGC 检测，当前进入 Data A v1 的“局部视频生成执行”阶段。

已经完成：
- CameraBench 1366 条视频的 Qwen3 候选发现与 SAM3 track bank；quality-pass track 3132 条、覆盖 911 个视频；
- 视频级 content_domain / style_domain、track editability catalog、donor pair pool、generation_plan_v1 已在上一阶段冻结；
- Data A 为 paired Real–Fake：Real 是 target A 原片，Fake_A 是在同一 clip / 同一 mask tube 内完成的局部 AIGC 编辑；donor B 仅作生成 reference；
- 不重做 Qwen3 候选发现、SAM3 跟踪、pairing 或采样。

当前输入：
- res/dataA_v1/plans/generation_plan_v1.json
- res/dataA_v1/registries/*.json
- res/sam_track_bank/sam3_quality_tracks_enriched.json
- 可访问的持久化 mask tube 路径。

请按以下顺序执行：
1. 先读 generation_plan，按 route 分组，检查输入路径与 OSS mask 可用性；
2. 为 VACE-14B 先实现统一 case packager：source clip、mask_raw/edit/gen/alpha、donor reference materialization、prompt、manifest；
3. 只对 VACE 的 object_swap / person_appearance_swap / surface_content_edit / attribute edit 各跑少量 smoke；
4. 通过 QA 后才扩 VACE batch；
5. VideoPainter、FaceFusion、PISCO 分别在其输入支线齐备后再做独立 smoke；
6. 所有生成都保存 attempts、参数、seed、原始输出、回贴输出与 QA 状态；
7. Real/Fake 使用相同渲染与编码流程，避免 codec shortcut。

重点：不要把 donor 像素复制粘贴到 target；donor 只是生成条件。不要做全局重生成或多操作组合编辑。
```

---

## 6. 新会话不应自行改变的约束

```text
- 一条正式 Fake 只含一个主要局部编辑；
- target 与 donor 不来自同一 source video；
- donor pairing 需要同 visual domain，object 还需同 semantic bucket，surface 还需同 subtype；
- 一个 target source video 默认只生成一条正式 Fake；
- 一个 donor track 默认最多复用三次；
- 不通过 smoke 的 route 不能直接批量；
- object insertion 不得用手工粘贴来凑数据；
- face donor 必须有合规来源或使用合成身份；
- mask 外回贴不是传统伪造手段，而是 paired-control 所需的保真步骤。
```
