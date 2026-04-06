# 根目录整理记录（Round 2）

本轮将根目录中的样例资源与运行产物进一步归档，避免根目录持续膨胀。

## 已迁移

- `lw1.mp4` -> `assets/samples/lw1.mp4`
- `lw2.mp4` -> `assets/samples/lw2.mp4`
- `lw3.mp4` -> `assets/samples/lw3.mp4`

## 运行日志策略

- 日志输出路径已切换到 `runtime/logs/livetalking.log`（见 `logger.py`）。
- 根目录旧日志 `livetalking.log` 可能因运行进程占用暂未删除，可在服务停止后清理。

## 说明

- 本次迁移未发现代码中对 `lw1.mp4/lw2.mp4/lw3.mp4` 的硬编码引用。
- 迁移目标目录均补充了 `README.md`，用于说明目录职责。

