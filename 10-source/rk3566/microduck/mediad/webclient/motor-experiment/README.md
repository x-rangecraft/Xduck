# 电机实验接口与示例

本文件夹集中保存接口说明和电脑端示例，可整体复制到电脑使用。需要 Python 3.10+，仅依赖标准库。

- [接口说明](motor-experiment.md)：任务上传、启动、状态、停止、结果下载与日志接口。
- [Python 客户端](motor_experiment.py)：调用实验接口的示例脚本。
- [正弦扫描说明](joint-sine-sweep-README.md)：离线任务生成与运行流程。
- [正弦扫描脚本](joint_sine_sweep.py)：在电脑上生成任务文件。

先在本文件夹中运行 `python3 motor_experiment.py --help` 或 `python3 joint_sine_sweep.py --help` 查看参数。生成任务不会连接设备；启动任务前请按接口说明完成现场检查。
