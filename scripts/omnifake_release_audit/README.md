# Omni-Fake 发布审计

本工具在正式下载和训练前回答两个独立问题：

1. Omni-Fake video 是否可用于 Real/Fake 主任务；
2. 它是否真实提供了可用于 evidence bottleneck 的配对、mask 和时序定位监督。

`hub` 只检查 Hugging Face 发布文件和许可证；`local` 检查已经下载的 parquet、解压视频、字段覆盖率、抽样解码和 SET/OOD 文件名重叠。

## 使用边界

- 本项目只使用 `video` 子集，训练和评测显式忽略音频流；不下载 `audio` 或 `avth`。
- SET 的 real/full_synthetic 来自 GenBuster-200K train，OOD 的对应部分来自 GenBuster-200K Closed Benchmark；不得把 Omni-Fake 与 GenBuster 当作两个独立来源累计结果。
- 官方 video parquet 字段声明中没有 pair、mask、bbox 或时间区间。本审计会检查真实 parquet，但在字段门通过前只把 Omni-Fake 当作三分类/二分类视频检测数据，不把它当作定位监督。
- 先下载两个小样本：SET `data/Video/train-00000-of-00120.parquet`，OOD `data/Video/test-00000-of-00040.parquet`。不要先下载完整约 120 GB video 发布。

## 输出位置

- 下载的 parquet、视频和 7z：`/tmp/omnifake/`，属于可重新下载的大文件；
- JSON/CSV 审计结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/omnifake_release_audit/v1/`，属于持久化小文件；
- 本审计不会生成需要上传 OSS 的大特征。

## 执行

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/omnifake_release_audit/run.sh

STAGE=preflight bash "$RUN"
```

在可以访问 Hugging Face 的机器上运行：

```bash
STAGE=hub bash "$RUN"
```

把 SET/OOD 下载或解压到 `/tmp/omnifake/Omni-Fake-SET` 和 `/tmp/omnifake/Omni-Fake-OOD` 后运行：

```bash
STAGE=local bash "$RUN"
```

只下载了部分 parquet 或视频时也可以运行，但结果会标记为 `incomplete` 或 `insufficient_or_partial`，不能当成全量发布审计。
