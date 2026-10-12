# 4096刚性100ms起步200更新：恢复成功基准

用户要求在刚性4096环境、100ms起步从零训练200轮。此实验以此前已验证能走的xduck_rigid_airtime_ab_20260929/air100为基准，而非最近叠加参考奖励及低速指令的配置。

模型和项目源码逐文件哈希与成功1024基准一致。保留当前质量8.8933kg、真实默认站姿和硬限位、足底及踝壳接触、足底独立±3°DR、普通原DR、Gaussian std1、动作尺度及所有原奖励。无示范轨迹奖励、无手写反馈。

刚性nq21/nv20/14电机；CUDA/mjwarp/implicitfast；机械20×1ms、电机1kHz、策略50Hz。恢复成功基准的宽速度命令：vx±0.5783、vy±0.4337、yaw±0.6917；20%前进桶最小vx0.4337，15%原地转向桶。空气时间门槛100ms，权重3，需另一脚接触。

调整范围：1024→4096环境，24步rollout仍不变；minibatch4→16，保持每minibatch6144；5epochs、LR0.001 adaptive KL0.01保留。所有原课程触发控制步除以4并向上取整，以样本量对应。此批量调整并不能保证优化轨迹完全相同，应由结果验证。

不恢复旧权重，seed1。完成200次更新停止训练，最终model_199.pt（样本量相当于1024×800次更新）。每50更新保存。训练后采用和成功基准相同seed50021、0.3m/s、四环境×8秒做1ms接触评估与正常1x录像。旧两组保持停止，不自动延长训练或改变门槛。

远端 /home/xingchen/Xduck/xduck_rigid4096_100ms_200_20260930；status.json状态；TRAINING_CONFIG.yaml完整参数；SUCCESSFUL_1024_CONFIG.yaml基准；audit.json差异核对；inputs_sha256.json输入哈希；diagnostics/与videos/最终结果。STOP文件只停止本实验。
