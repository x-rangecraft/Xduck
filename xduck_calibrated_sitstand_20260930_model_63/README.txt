五姿态候选参数训练的坐下/起立策略；已通过记录的仿真门槛，尚未真机验证。
部署使用 xc_sitstand.onnx + xc_sitstand_policy.py 两个文件，必须成对使用。
API v2，20ms，输入62维，输出14个电机目标偏移；电机顺序、KP/KD及软限位沿用现有消费协议。
训练模型含14关节K=75Nm/rad、D=1，以及五姿态拟合零位和有效IMU旋转。
消费端继续读取原始编码器/IMU，不应再次叠加候选零位或IMU角度修正。
model_63.pt用于UniLab续训；上传策略请选择ONNX与配套Python消费文件。
training_config.yaml包含本机路径，复现实验还需要现有UniLab工程与CAD网格。
