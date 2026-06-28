# Data A v1：Wan2.1-VACE-14B 局部编辑 Smoke 执行规格

> 状态：设计冻结，尚未开始写 packager / 执行 VACE。
>
> 本文是 Data A v1 当前 VACE-14B smoke 的唯一实现依据。任何后续脚本、manifest、服务器命令均以本文约束为准。

---

## 1. 目标与不变量

构造 CameraBench 上严格配对的局部编辑样本：

```text
Real_A = target 视频 A 的同一 clip
Fake_A = 在 A 的同一 clip、同一局部 mask tube 中由 VACE 生成后回贴得到
Donor_B = 仅提供 reference condition，不能作为 Real pair，也不能直接复制 donor RGB 到 target
```

每个 accepted pair 必须尽量保持：

- target 背景；
- target 相机运动；
- 非编辑区域；
- 片段时间范围；
- 最终 Real / Fake 的 fps、帧数、分辨率、编码策略与音频策略。

每条 Fake 只允许一个主要局部编辑。

### 已冻结、绝不重做的上游资产

- Qwen3-VL 候选发现；
- 视频 `content_domain / style_domain` 补标；
- SAM3 track bank 与 mask tube；
- source–donor pairing pool；
- operation / donor / route 自动采样；
- `res/dataA_v1/plans/vace14b_stage1_quota_plan.json` 的 15 条 quota smoke case。

### 当前 VACE smoke 范围

```text
object_swap              5
person_appearance_swap   3
surface_content_edit     3
object_attribute_edit    2
surface_attribute_edit   2
```

不在此阶段接入：`face_identity_swap`、`object_removal`、`object_insertion`。

---

## 2. 官方 VACE 对接结论

VACE 的直接推理接口由四类条件组成：

```text
src_video
src_mask
src_ref_images (optional)
prompt
```

Data A 对应关系：

```text
src_video       = canonical source_clip.mp4
src_mask        = 从 M_gen 派生的动态 mask video
src_ref_images  = donor_reference.png（仅 donor-driven route）
prompt          = 描述最终画面的 model_prompt
```

### 重要接口现实

1. VACE 实际读取的是 mask video，不直接读取 SAM3 `.npz`。
2. SAM3 `.npz` 始终是唯一 mask 真值；VACE mask video 是输入派生物。
3. `src_ref_images` 是一张或多张 RGB 图像路径；donor alpha 仅用于构造白底 reference 和 QA，不作为 VACE 直接输入。
4. 官方 VACE 会按预设生成规格处理视频；Data A 必须主动生成 canonical paired Real，不能将 native source clip 与 VACE output 直接配对。
5. VACE 的 mask 外区域仍可能漂移；最终 Fake 必须走 soft-alpha compositing。

---

## 3. 依赖与代码布局

### 3.1 第三方依赖

主仓库仅纳入 VACE；Wan2.1 作为环境依赖安装，不复制 Wan2.1 源码。

```text
third_party/
└── VACE/                         # ali-vilab/VACE，固定 upstream commit

requirements/
└── vace_wan21_offline.txt        # 锁定 Python / Wan2.1 / VACE requirements 的离线安装说明
```

- `third_party/VACE` 通过 git submodule 在实际 git 工作目录添加；不以复制源码替代 upstream。
- `Wan2.1` 在独立环境内从固定 commit 构建的离线 wheel 或本地源码包安装。
- 模型权重、pip cache、视频、mask 视频、attempt 结果不提交 Git。

### 3.2 项目新增模块

```text
scripts/dataa_v1/
├── __init__.py
├── common.py                    # 路径、JSON、ffprobe、日志、命令执行基础设施
├── schema.py                    # Plan / track bank → CanonicalCaseSpec
├── path_resolver.py             # /tmp / local / OSS 持久路径映射与审计
├── mask_io.py                   # SAM3 npz 读取、frame index 映射、dense mask 生成
├── clip_selection.py            # 连续可见段、shot-cut 规避、3/4/5 sec 选择
├── canonical_video.py           # canonical Real 规范化与帧/分辨率一致性
├── mask_processing.py           # M_raw / M_edit / M_gen / M_alpha
├── mask_video.py                # M_gen → VACE mask video；反解码一致性验证
├── donor_reference.py           # donor 最优帧、RGBA/白底 RGB crop、alpha 保存
├── prompt_builder.py            # model_prompt / control_prompt
├── mask_visualization.py        # clip-aligned raw / overlay mask 视频
├── vace_command.py              # 仅生成 VACE argv / shell command；不执行
├── manifest.py                  # case_manifest 与 preflight report 写入
├── package_vace_case.py         # 单 case CLI
├── package_vace_plan.py         # 15 case plan CLI（后续启用）
└── validate_case_pack.py        # 独立校验器

configs/dataa_v1/
└── vace14b_packager.yaml        # profile、mask 参数、路径、codec、阈值

tests/dataa_v1/
├── fixtures/                    # 纯合成短视频 / synthetic npz / 小 JSON schema fixture
└── test_*.py
```

已有 `scripts/render_dataa_v1_mask_videos.py` 应优先复用或拆成公共渲染函数；不得维护两套不一致的 overlay/bbox 逻辑。

---

## 4. 两阶段执行架构

### Stage P：Packager（先实现；不执行 VACE）

输入：冻结 plan + track bank + 原视频 + 原始 mask tube。

输出：一个可被 VACE 直接消费、并已通过输入一致性验证的 attempt pack。

```text
plan case
→ schema normalization
→ path preflight
→ clip selection
→ canonical paired Real creation
→ M_raw / M_edit / M_gen / M_alpha
→ VACE dynamic mask video
→ donor reference（如需要）
→ prompt
→ visualization
→ manifest
→ VACE command spec（不执行）
```

### Stage G：Generation + Pair Rendering（后实现）

```text
valid case pack
→ VACE generated_raw.mp4
→ generated output canonicalization / frame validation
→ Fake = M_alpha * Generated + (1 - M_alpha) * Real
→ fake_pair_render.mp4
→ automatic QA
→ accepted / rejected / manual-review state
```

Stage P 与 Stage G 必须分离。第一轮 object_swap smoke 先验证 Stage P，再允许调用 VACE。

---

## 5. CanonicalCaseSpec：计划解析后的唯一内部格式

Plan 的真实 JSON 嵌套格式不应扩散到所有模块。`schema.py` 必须将 plan case 和 track bank 条目标准化为：

```json
{
  "case_id": "vace14b_stage1_0001",
  "operation": "object_swap",
  "generator_route": "vace14b_masktrack_reference_swap",
  "target": {
    "video_id": "...",
    "video_path": "...",
    "track_id": "...",
    "candidate_id": "...",
    "candidate_class": "bounded_object",
    "canonical_concept": "...",
    "display_phrase": "...",
    "region_family": "physical_instance",
    "mask_tube_path": "...",
    "bbox_tube_xywh": "...",
    "content_domain": "...",
    "style_domain": "..."
  },
  "donor": {
    "video_id": "...",
    "video_path": "...",
    "track_id": "...",
    "canonical_concept": "...",
    "display_phrase": "...",
    "mask_tube_path": "...",
    "bbox_tube_xywh": "..."
  },
  "sampling_meta": {},
  "plan_source": "res/dataA_v1/plans/vace14b_stage1_quota_plan.json"
}
```

`donor` 对 text-only route 为 `null`。

任何缺字段、重复 case id、target/donor 同 source video、operation/route 不匹配、track 不存在，必须在 schema preflight 直接失败，不允许悄悄回退或人工补样。

---

## 6. 路径审计与持久化映射

`mask_tube_path` 可能仍指向 `/tmp/...`。必须由 `path_resolver.py` 统一处理，不允许业务模块自行拼路径。

### 输入映射格式

```json
{
  "schema_version": "dataA_v1_path_mapping_v1",
  "rules": [
    {
      "source_prefix": "/tmp/cameramotion_det/res/sam_track_bank/track_masks_v1",
      "persistent_prefix": "oss://<bucket>/<prefix>/track_masks_v1",
      "status": "planned_or_verified"
    }
  ],
  "explicit_overrides": {}
}
```

### preflight 结果

每个 target / donor mask 标记为：

```text
readable_persistent
readable_volatile
mapped_but_unverified
missing
```

规则：

- `missing`：case 停止，状态 `blocked_missing_mask`；
- `readable_volatile`：允许仅做本地 Stage P 调试，但不允许正式 VACE generation；
- `mapped_but_unverified`：不允许正式 generation；
- `readable_persistent`：可进入正式 generation。

此项是 Stage 2 的主产物，但 PathResolver 的接口需在 Stage P 同时实现。

---

## 7. 自动 clip selection

### 目标

对 target track 从最长连续可见段中选择 3、4 或 5 秒 clip，优先 5 秒，且避开镜头切换。

### 算法

1. 读取 `.npz` 中 `frame_indices: int32[N_visible]` 与 `masks:uint8[N_visible,H,W]`。
2. 将连续可见定义为 source-frame gap `<= 1`；最长段不足最低阈值时可配置为 gap `<= 2`，但必须在 manifest 记录容忍规则。
3. 使用原视频 fps 将每段转换为秒数，保留可覆盖 3 秒的段。
4. 枚举每个候选 5 sec、4 sec、3 sec 窗口：
   - mask 连续且非空；
   - 平均面积、最小面积与 bbox 面积满足 operation-specific gate；
   - 在目标 clip 中的有效可见率为 100%；
   - cut score 最小（使用相邻帧 HSV histogram / edge-change；初版只拒绝明确 hard cut）；
   - track 位移与 bbox 波动不超过 route 允许阈值。
5. 依次尝试 5 / 4 / 3 秒，选综合分最高的窗口；没有合法窗口则 `blocked_low_visibility`。
6. 将原始 source frame 起止、时间、采样到 canonical frame 的映射都写入 manifest。

### Canonical VACE profile

VACE profile 可配置，不能硬编码：

```yaml
smoke_480:
  fps: 16
  frame_options: [49, 65, 81]
  landscape_size: [480, 832]   # H, W
  portrait_size: [832, 480]

accept_720:
  fps: 16
  frame_options: [81]
  landscape_size: [720, 1280]
  portrait_size: [1280, 720]
```

- 49 / 65 / 81 帧分别对应 3 / 4 / 5 秒的 inclusive 16 fps 时间轴（`4n+1`）。
- `source_real_raw.mp4` 保留原始剪段，仅做审计。
- `source_clip.mp4` 是真正用于 paired 训练的 Canonical Real：固定 fps、固定帧数、固定 raster、固定裁剪策略。
- `source_clip.mp4` 必须与 target mask video、最终 `fake_pair_render.mp4` 完全对齐。

---

## 8. Mask 四层语义

每个 attempt 必须保存 `npz` 格式的：

```text
M_raw   = SAM3 原始实例 mask，重采样到 canonical frame 的 nearest-neighbor 版本
M_edit  = 对 M_raw 进行轻度清理 / closing / dilation 后的二值 edit mask
M_gen   = 真正传入 VACE 的二值生成 mask
M_alpha = 用于最终 compositing 的 [0,1] soft alpha
```

### 默认处理规则

- `M_raw`：不改语义、不用 MP4 覆盖，逐帧最近邻映射；
- `M_edit`：移除极小连通域，填小洞，轻度空间 closing；
- `M_gen`：默认等于 `dilate(M_edit, radius_px)`，保证边界重绘有充分上下文；
- `M_alpha`：对 `M_gen` 的边缘进行有限距离 softening；alpha 的有效范围不能失控扩张。

所有 kernel / dilation / blur / connected-component 参数写入 manifest。

### Surface 特例

对 `display_screen / sign_or_poster / framed_art / paper_book_map`：

- 保留 `M_raw`；
- 允许在独立 `surface_mode` 中用平面近似或四边形/矩形扩展构造 `M_edit`；
- 任何平面扩展必须保存参数、扩展前后面积比和可视化；
- Stage P 首版先实现通用 mask 模式，surface refinement 通过 config 开关逐步启用，不能隐式替换原始 mask。

---

## 9. VACE mask video

`target_mask_gen_video` 是 VACE 输入，须满足：

```text
same frame count
same fps
same H × W
same clip time axis
same frame order
binary values after thresholding
```

推荐使用 RGB lossless codec（优先 `libx264rgb -crf 0 -pix_fmt rgb24`；根据服务器 FFmpeg 能力配置 fallback）。不得将 H.264 loss 带来的灰边当作监督标签。

### 必须反解码验证

写出 mask video 后：

1. 用与运行环境兼容的 reader 重新读取；
2. 读取第一通道并阈值化；
3. 和 `M_gen.npz` 逐帧比较；
4. 写入：

```json
{
  "frame_count_match": true,
  "shape_match": true,
  "thresholded_iou_mean": 1.0,
  "thresholded_iou_min": 1.0,
  "pixel_equal_after_threshold": true
}
```

不满足即 `blocked_mask_video_mismatch`。

---

## 10. Donor reference 自动实体化

仅 donor-driven routes 使用。

### 候选帧评分

在 donor visible frames 中计算：

```text
score =
  area_score
+ interior_margin_score
+ temporal_stability_score
+ sharpness_score
+ non-degenerate_bbox_score
```

其中：

- area：mask / bbox 面积足够大；
- margin：mask/bbox 不贴画面边缘；
- stability：与相邻 visible frame 的 mask IoU / bbox 位移稳定；
- sharpness：crop 内 Laplacian variance 等无参考清晰度指标；
- bbox：宽高、面积、长宽比合法。

选择最高分帧；无需人工选 donor frame。

### 输出

```text
donor_reference.png        # 白底 RGB crop，真正传 VACE
donor_reference_alpha.png  # donor 原始 / 清理 alpha，仅审计和重建
donor_reference_meta.json  # source frame、bbox、score components、crop padding
```

- donor crop 只用于生成条件；禁止任何 target 合成环节读取 donor RGB；
- VACE 参考图不使用透明 alpha，因此 packager 统一输出白底 RGB；
- crop 保留有限 padding，避免物体被截断。

---

## 11. Prompt 双层设计

VACE model prompt 采用**结果描述**，避免仅给命令式“replace / do not alter”。

每个 case 保存两种文本：

```text
model_prompt   # 真正传 VACE，描述最终画面
control_prompt # 写入 manifest/QA，明确 Data A 保留约束；不依赖它控制模型
```

### model_prompt 模板信息源

- operation；
- target canonical concept / display phrase；
- donor canonical concept / display phrase（如有）；
- content_domain；
- style_domain；
- 可选 scene context（只在已有视频 caption 时使用，禁止虚构）；
- 可选 reference flag。

### operation 模板

- `object_swap`：描述替换后对象在原场景的光照、尺度、透视、运动模糊与阴影；
- `person_appearance_swap`：描述人物最终外观，保留原姿态、动作与镜头运动；
- `surface_content_edit`：描述平面上最终 coherent content / layout，reference 优先；
- `object_attribute_edit`：描述目标材质、颜色、纹理等属性；
- `surface_attribute_edit`：描述平面图案、涂装、颜色或材质。

`control_prompt` 固定强调：只改变 target mask 及合理边界环带；保持背景、相机运动、姿态、场景几何、光照、非编辑区域和时间连续性。

---

## 12. Attempt 目录和 manifest

```text
res/dataA_v1/attempts/<case_id>/
├── source_real_raw.mp4
├── source_clip.mp4
├── target_mask_raw.npz
├── target_mask_edit.npz
├── target_mask_gen.npz
├── target_mask_alpha.npz
├── target_mask_gen.mp4
├── target_mask_raw.mp4
├── target_mask_overlay.mp4
├── donor_reference.png                         # donor route only
├── donor_reference_alpha.png                   # donor route only
├── donor_reference_meta.json                   # donor route only
├── donor_mask_raw.mp4                          # donor route only
├── donor_mask_overlay.mp4                      # donor route only
├── vace_command.json
├── case_manifest.json
├── preflight_report.json
└── logs/
```

Stage G 后增加：

```text
generated_raw.mp4
fake_pair_render.mp4
qa_report.json
```

### case_manifest 必填字段

```text
case_id
stage_status
target: video_id / track_id / resolved path
donor: video_id / track_id / resolved path (optional)
operation / generator route
source clip native + canonical start/end frames/times/fps/resolution
canonical VACE profile
mask source path + M_raw/edit/gen/alpha paths
mask processing parameters
bbox tube / union bbox / normalized [0,1000] bbox
prompt: model_prompt / control_prompt
reference metadata
seed (reserved)
model version / VACE upstream commit / Wan commit / weight revision (reserved)
code commit
generated command (reserved in Stage P)
preflight + QA status
```

---

## 13. Stage P 自动检查与状态

### preflight / packager 状态

```text
planned
blocked_missing_mask
blocked_volatile_mask
blocked_low_visibility
blocked_clip_selection_failure
blocked_mask_video_mismatch
blocked_donor_reference_failure
blocked_schema_error
packed
```

### Stage G 质量状态

```text
accepted
rejected_generation_failure
rejected_global_drift
rejected_wrong_edit
rejected_low_visibility
needs_manual_review
```

Stage P 不得将任何未完成 case 标记为 `accepted`。

---

## 14. 实施顺序

### P0：仓库与环境边界

1. 添加 VACE 作为 `third_party/VACE` pinned submodule；
2. 添加 VACE/Wan2.1 离线环境说明；
3. 补充 `.gitignore`：weights、models、attempt 视频、mask video、logs、wheel cache；
4. 不下载权重、不执行 VACE。

### P1：schema + path preflight

1. 实现 `CanonicalCaseSpec`；
2. 实现 plan / track-bank linter；
3. 实现 `PathResolver` 与 path mapping；
4. 对 15 case 输出路径可访问性 audit，但不复制数据。

### P2：单 case packager

1. 用第一条 `object_swap` case 验证 Stage P 全链路；
2. 输出完整 attempt pack；
3. 运行 mask video 反解码校验；
4. 人工只看 raw/overlay 与 donor reference 是否合理；
5. 不运行 VACE。

### P3：all-15 Stage P

1. 仅对 preflight 的可读且持久 mask 路径打包；
2. 其余 case 保留明确 blocked 原因；
3. 汇总 case manifest / failure table。

### P4：单 case VACE execution smoke

1. 准备离线环境、VACE commit、Wan2.1 package、VACE-14B 权重；
2. 只执行已 pack 的 object_swap 一条；
3. 验证 VACE output 与 source_clip 对齐；
4. 实现 compositing 和 generation QA。

### P5：15-case generation smoke

仅在 P4 验收通过后执行。

---

## 15. 验收门槛

### Stage P 单 case 通过条件

- plan / track schema 无歧义；
- target（和 donor 如需要）mask 可读取，且路径持久性状态明确；
- target clip 为 3–5 秒、连续可见、无明显 hard cut；
- source_clip 与 target_mask_gen video 严格同帧数 / fps / raster；
- mask video 反解码后与 M_gen 阈值化逐帧一致；
- donor reference 自动选出且 alpha / crop / metadata 齐全；
- case manifest 字段完整；
- 不调用 VACE、不产生 Fake。

### Stage G 单 case 通过条件

- VACE 可解码输出；
- output 与 canonical Real 可对齐；
- 回贴后 mask 外差异不出现大面积漂移；
- 编辑发生在请求区域附近；
- 无黑帧、花屏、全局重绘；
- 自动 QA 与人工抽查均未指出严重失败。

---

## 16. 明确禁止事项

- 重做对象发现、SAM3 或 pairing；
- 重新随机采样 plan；
- 以 bbox 替代 SAM3 mask 作为 VACE 主输入；
- 用 donor RGB 像素直接粘贴 target；
- 用传统拼贴替代 AIGC 生成；
- 全局重生成 target；
- 一条 Fake 同时叠加多个主要编辑；
- 将 VACE mask MP4 当成真值替代 `.npz`；
- 提交模型权重、视频、attempt 二进制文件到 Git。
