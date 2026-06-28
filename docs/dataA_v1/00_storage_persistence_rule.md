# Data A v1 持久化存储硬约束

> 设为项目长期约束。任何后续会话在读取或生成 Data A 前，都必须先检查本规则。

## 一、核心规则

**所有仅存于 `/tmp` 的 Data A 资产都视为易失数据。**

镜像关闭、作业重启、节点迁移或临时盘清理后，`/tmp` 下的内容可能消失。因此：

```text
/tmp 中的数据不能作为唯一副本；
不能把 /tmp 路径作为后续 generation plan 的长期依赖；
必须先备份到 OSS 或其他持久化存储，并完成可读性核验。
```

## 二、当前必须持久化的资产

优先级 P0：

```text
/tmp/cambench_train/cam_train/object_discovery_sam/track_masks_v1/
```

该目录包含 SAM3 mask tube 文件。每个 `.npz` 是 Data A 的原始无损空间监督，通常包含：

```text
frame_indices: int32 [N_visible]
masks: uint8 [N_visible, H, W]
```

P1：后续生成阶段写入 `/tmp` 的所有中间与最终资产：

```text
source clip
mask_raw / mask_edit / mask_gen / mask_alpha
donor reference crop / alpha
generated_raw.mp4
fake_pair_render.mp4
case_manifest.json
logs
QA result
```

## 三、推荐 OSS 目标位置

当前约定的 SAM3 mask 备份目标：

```text
oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/sam3/track_masks_v1
```

上传示例：

```bash
ossutil64 cp -r \
  /tmp/cambench_train/cam_train/object_discovery_sam/track_masks_v1 \
  oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/sam3/track_masks_v1
```

上传完成后必须至少核验：

```text
1. OSS 中目录层级与本地 track_masks_v1 一致；
2. `.npz` 文件数量与本地一致；
3. 随机抽取多个 `.npz` 能下载并被 numpy.load 正常读取；
4. 生成 plan 中使用的 target / donor mask 均能从 OSS 映射到可读路径。
```

## 四、manifest 路径规则

长期 manifest 应逐步采用双路径字段：

```json
{
  "mask_tube_path_local": "/tmp/.../track.npz",
  "mask_tube_path_oss": "oss://.../track.npz",
  "mask_tube_path_resolved": "<当前运行环境可读路径>"
}
```

原则：

```text
- local path 仅用于当前运行时加速；
- OSS path 是持久化真源；
- 新镜像 / 新节点启动后必须先从 OSS 恢复或挂载；
- 不得假定旧 /tmp 路径仍存在。
```

## 五、后续会话强制检查项

在 03｜Data A v1 局部视频编辑生成与 Smoke Test 开始时，第一项不是运行 VACE，而是：

```text
检查 generation plan 中每个 target / donor 的 mask 是否已拥有可读的持久化副本。
```

若 OSS 备份尚未完成：

```text
先执行上传与核验；
不要开始依赖该 mask 的大规模生成；
可以只在当前临时环境中做极小 smoke，但结果与 manifest 必须同步持久化。
```

## 六、事实状态记录

截至本文创建时：

```text
- 已知 mask tube 原始路径位于 /tmp；
- 已提出 OSS 上传命令；
- 不应假定上传已经完成，除非有上传日志和核验结果；
- GitHub 文档可以持久保留“必须备份”的规则，但不能替代实际 OSS 数据备份。
```
