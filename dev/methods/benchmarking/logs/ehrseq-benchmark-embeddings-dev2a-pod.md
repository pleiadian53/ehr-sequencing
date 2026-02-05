@TESTING_ROADMAP.md#L148-156  I am currently running this proceses and also another process:  @TESTING_ROADMAP.md#L104-110  I am a bit surprised that they are not using a lot of VRAM. Are they configured to use cuda? they should according to ehr-sequencing/src/ehrsequencing/utils/resource_manager.py

---

Your processes **are using CUDA** and quite heavily:

| Process                                       | GPU Memory | Status  |
| --------------------------------------------- | ---------- | ------- |
| `benchmark_embedding_finetuning.py`           | 14,270 MiB | Running |
| `benchmark_transfer_learning.py` (instance 1) | 9,022 MiB  | Running |
| `benchmark_transfer_learning.py` (instance 2) | 9,022 MiB  | Running |

**Total GPU usage: 32,327 MiB / 46,068 MiB (70%)** with **100% GPU utilization**.

Note: You have **two instances** of `benchmark_transfer_learning.py` running (PIDs 1008025 and 1008232). This might be unintentional - you may have launched it twice. If so, you can kill one:

```bash
kill 1008232  # or 1008025
```

---

