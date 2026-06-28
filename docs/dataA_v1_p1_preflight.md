# Data A v1 P1：VACE Stage-1 Schema 与 Mask Path Preflight

P1 只审计冻结 plan 的结构、track 引用和 `mask_tube_path` 可访问性。

它**不会**：

- 运行 VACE；
- 下载权重；
- 修改 plan、track bank 或 registry；
- 导出视频、mask video 或 Fake；
- 重新采样 target、donor 或 operation。

## 输入

```text
res/dataA_v1/plans/vace14b_stage1_quota_plan.json
res/sam_track_bank/sam3_quality_tracks_enriched.json
optional: server-local path mapping JSON
```

## 运行

先做无数据自测：

```bash
python scripts/dataa_v1/audit_vace_stage1.py --self-test
```

服务器真实审计示例：

```bash
python scripts/dataa_v1/audit_vace_stage1.py \
  --plan res/dataA_v1/plans/vace14b_stage1_quota_plan.json \
  --track-bank res/sam_track_bank/sam3_quality_tracks_enriched.json \
  --path-mapping /path/to/server/path_mapping.json \
  --output /path/to/persistent/audits/vace14b_stage1_mask_preflight.json \
  --strict
```

建议把报告写到持久盘或 OSS 挂载目录。`res/dataA_v1/reports/` 也可以作为本地运行目录，但它被 `.gitignore` 忽略，避免把服务器特定路径和数据状态提交 Git。

## Path mapping

参见：`configs/dataa_v1/path_mapping.example.json`。

该 mapping 只解决“原 path → 持久 path”的运行时解析问题；不修改原始 track bank，也不假设 OSS 已完成上传。只有当前运行机器实际读到映射目的 `.npz` 时，状态才会成为 `readable_persistent`。

若原 `/tmp/...` 文件还可读、但映射目标不能验证，则会报告 `readable_volatile`，而不是错误地宣称它已持久化。

## 结果状态

- `preflight_passed`：target / donor（如需要）的 mask 均为本机可读、非 `/tmp` 的持久路径，且 `.npz` 含合法 `frame_indices` 与 `masks`。
- `blocked_missing_mask`：路径不存在且没有映射。
- `blocked_volatile_mask`：当前只在 `/tmp` 或其他 volatile location 可读。
- `blocked_mapped_but_unverified`：存在持久化映射，但本机无法验证映射目标。
- `blocked_invalid_mask_npz`：`.npz` 不符合 `frame_indices:[N]` 与 `masks:[N,H,W]` 的可用 tube 契约。
- `blocked_schema_error`：case id、target/donor track、operation 或 pairing 语义不合法。

`--strict` 会在任何 case 非 `preflight_passed` 时返回 exit code 1，适合服务器 batch job 或 CI。

## 报告内容

每条 case 的报告都会包含：

```text
case_id / operation / generator_route
target 与 donor 的 canonical metadata
原始 mask 路径、resolved path、path state、mapping rule
npz 键/shape/visible-frame 合法性检查结果
blockers
canonical_case_spec
```

P2 的 clip selector、mask processor 和 donor-reference builder 只读取 P1 通过的 `canonical_case_spec`，而不会重新解释 plan 的原始 JSON 嵌套结构。
