# 取证证据方向：数据准备度更正

日期：2026-07-22

本记录更正 [`forensic_evidence_token_direction_feasibility_audit_20260722.md`](forensic_evidence_token_direction_feasibility_audit_20260722.md) 中基于 Omni-Fake 单独审计得出的“当前数据准备度未通过”。该结论不能泛化为整个公开视频方向缺少数据。

补充核验实际发布内容后的裁决为：

- 全生成 Real/Fake：**通过**，GenBuster/DataB 已满足，GenVidBench 可选扩展；
- 现代局部编辑 pair：**有条件通过**，OpenVE-3M 公开原视频、编辑视频和编辑指令；
- 时间定位：**通过**，ActivityForensics 已公开视频与篡改时间段；
- 大规模通用空间 mask 与高质量 CoT：**仍未通过**，只能使用 ViF-CoT-4K、DataA 和 pair 差分形成小规模或弱监督；
- Omni-Fake：仍仅适合作为去偏后的补充训练/测试，不能直接视为干净的生成范围监督。

因此，数据层总裁决修正为 **有条件通过**。完整数据矩阵、限制和方法设计前的数据 Gate 见 [`open_video_data_support_audit_20260722.md`](open_video_data_support_audit_20260722.md)。原审计关于方法新颖性不足、必须比较专家单独与简单后融合、最终指标必须落在外部 Real/Fake 的其他结论不变。

