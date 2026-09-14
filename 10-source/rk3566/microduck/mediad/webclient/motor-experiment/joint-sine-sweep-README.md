# 离线指令文件生成示例

`joint_sine_sweep.py` 现在只在电脑上生成任务文件，不再实时发送指令，也不会连接或使能设备。

```sh
python3 joint_sine_sweep.py --config config.json --output sweep.jsonl
python3 motor_experiment.py http://192.168.50.121:8080 upload sweep.jsonl
# 检查上传结果与当前设备姿态后，现场明确启动：
python3 motor_experiment.py http://192.168.50.121:8080 start 返回的任务ID
```

沿用原配置的 joints、scan_joints、ramp_seconds、cycles、settle_position_rad、max_tracking_error_rad、max_temperature_c。每台电机的 q0 必须明确填写，不能为 null；可以先用 capabilities 查询 current_motors 获取当前姿态。增益和零位由实验人员设定，不会自动猜测。

原来的反馈闭环 settle 改成提前确定的固定保持时间 settle_seconds（默认 2 秒）。这符合“提前生成、不根据反馈重新计算”的任务模式；设备仍在本地检查跟踪误差、温度、限位和故障。生成器保留 C2 包络和原幅度/频率组合，不做额外插值。单任务最大 30 分钟，超出应分批生成。

实时 acquire/command/release/feedback 接口及 --execute/--diagnose 用法已经移除。完整流程见 motor-experiment.md。电脑脚本随时退出不影响已经启动的设备任务。
