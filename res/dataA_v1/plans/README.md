# Data A v1 Execution Plans

Place the server-provided frozen full VACE execution plan here:

```text
res/dataA_v1/plans/frozen_full_vace_execution_plan.json
```

This repository must not synthesize that file from candidate pools or the 15-case quota smoke plan. The frozen plan is source-of-truth metadata and must already contain fixed `case_id`, target, donor when required, `operation`, `generator_route`, and `sampling_meta`.

