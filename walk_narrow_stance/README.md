# 缩小站距的走：从 0 到最终 ppo_flex100 模型

归档日期：2026-10-12。实验执行于2026-09-30至2026-10-10。用户指定备注：**（缩小站距的走）**。

本目录保存这条行走模型的完整**选用主线**：从随机初始化、累计1200轮课程、1229均速修正、1279柔性适配，到自然站距、多方向PPO和最终柔性适配。包含关键权重、原配置、源代码版本、依赖锁、物理资产、训练用数据、过程日志、验证原始数据和视频。

**最终是 `ppo_flex100/model_50.pt`，实际新增51次PPO更新，不是完成100轮。** `ppo_flex100` 是最初计划100轮的目录名；用户随后要求先50轮，选取最近完整的50号检查点。检查点从0编号。

这是一份旧版UniLab/Warp训练归档。它独立于本分支现有 `walk/stand/xl330` 新版工程，**没有完成对新版1.3.1的训练语义移植或真机部署**。

## 1. 最终模型与快速入口

|内容|位置|
|---|---|
|最终训练checkpoint|[checkpoints/07_flex_model_50.pt](checkpoints/07_flex_model_50.pt)|
|最终均值ONNX|[policy/flex_adapt_50.onnx](policy/flex_adapt_50.onnx)|
|推理消费者：关节顺序、REFERENCE、尺度、KP/KD|[policy/xc_walking_v3_policy.py](policy/xc_walking_v3_policy.py)|
|对称站姿目标与训练/电机ID顺序|[policy/NEW_STANDING_TARGET.json](policy/NEW_STANDING_TARGET.json)|
|界面电机ID角度表|[policy/DEFAULT_POSE_MOTOR_ID.csv](policy/DEFAULT_POSE_MOTOR_ID.csv)|
|实际最终训练配置|[runs/xduck_r2_flex_adapt100_20261010/PPO_FLEX100_CONFIG.yaml](runs/xduck_r2_flex_adapt100_20261010/PPO_FLEX100_CONFIG.yaml)|
|最终8DR闭环结果|[evidence/final_flexible_8dr/FLEX50_CAPABILITY_AUDIT.json](evidence/final_flexible_8dr/FLEX50_CAPABILITY_AUDIT.json)|
|全部8DR前进视频|[evidence/final_flexible_8dr/flex51_forward_8dr_1x.mp4](evidence/final_flexible_8dr/flex51_forward_8dr_1x.mp4)|
|六方向代表视频：case7|[evidence/final_flexible_8dr/flex51_six_commands_1x.mp4](evidence/final_flexible_8dr/flex51_six_commands_1x.mp4)|
|每个文件的大小/SHA-256|[PACKAGE_MANIFEST.json](PACKAGE_MANIFEST.json)|
|所有关键模型原始来源|[checkpoint_sources.json](checkpoint_sources.json)|

最终checkpoint SHA-256：`e959c81f9a362bc6c3c0a4f2cc626716c41d54fafad2ff74debe900e65458ad9`。

最终ONNX SHA-256：`3db03a4558cb06033e9239af4c57bec5bff107bc9d2fed27df127221e564f85d`。

## 2. 权重主线：不能只按文件名猜训练次数

```text
随机初始化（不加载已有checkpoint）
 └─ rigid4096_100ms_200/model_199.pt：完成200次
     └─ rigid4096_curriculum/model_900.pt：累计901次
         └─ rigid4096_220after900/model_1199.pt：续299次，累计1200次
             └─ speed_correction/mean_only/model_1229.pt：续30次，累计1230次
                 └─ meanonly_flex50/model_1279.pt：续50次，累计1280次
                     └─ clean_roll_seed.pt：仅重新初始化左右髋侧摆输出两行
                         └─ clean r3/model_80.pt：新优化器，选取本阶段81次更新
                             └─ direction/model_150.pt：选取本阶段151次更新
                                 └─ backward_bootstrap.pt：800次监督优化，不是PPO轮数
                                     └─ direction r2/model_179.pt：本阶段180次PPO更新
                                         └─ ppo_flex100/model_50.pt：本阶段51次柔性PPO更新
```

从自然站距阶段起，PPO优化器重新创建，actor归一化冻结；这些编号是各新实验的局部编号。它们不是沿着1199/1279文件名继续编号。不同阶段并行环境数不同，不能把更新数当作相同数量的样本。

没有把先前失败的BC、ready_start/transition及后来未采用的方向第三/第四轮权重混入最终主线。部分历史状态用于保留集；动作标签来源在各阶段脚本和清单中明确记录。

## 3. 从0开始的200次：真正随机初始化

目录：`runs/xduck_rigid4096_100ms_200_20260930`，源码版本 `00_fromzero200`。

- 来源是已验证100ms设置的工程/配置，不加载该基线的权重；`resume=false`、`load_run=-1`、seed1。
- 4096环境、24控制步/更新，即98,304个样本；16个minibatch、每批6144样本、5 epochs。
- 初始LR0.001，自适应KL0.01；摆脚时间下限100ms，不启用后来参考曲线项。
- 真正刚性模型：删除48被动坐标，nq21/nv20/14电机；策略50Hz，电机1kHz，每控制周期20个1ms机械步。
- 原1024环境的非DR课程步数除4并向上取整，详见该目录 `audit.json`；不是把学习率乘4。
- 200次结束选择 `model_199.pt`，SHA见 `checkpoint_sources.json`。

原训练入口（在恢复后的该阶段 `project/` 中）：

```bash
uv sync --locked
PYTHONPATH="$PWD/src" MUJOCO_GL=egl uv run --no-sync python scripts/train_rsl_rl.py --config-name rigid4096_100ms_200
```

原始 `setup.py` 和 `supervisor.py` 用于审计过程，**不要在已恢复目录直接执行setup.py**；它会尝试新建/复制目录。重新训练时改用新的输出目录。

## 4. 200→累计1200：课程与900处分支

源码版本 `01_curriculum900`、`02_curriculum1200`。恢复权重、归一化、优化器以及环境课程进度。

|累计更新阶段|摆脚门槛|DR倍率|参考权重|步频权重|
|---|---:|---:|---:|---:|
|0–299|100ms|1.0|0|0|
|300–599|140ms|1.5|0.45|0.15|
|600–899|180ms|2.0|0.45|0.15|
|900以后（最终选用分支）|220ms|2.5|0.45|0.15|

原180ms分支继续到约1055次但未作为主线终点；最终从其 `model_900.pt`（已完成901次）另开220ms分支，续299次到 `model_1199.pt`。不是从第900次再加300次，更不是从1055继续到1200。

指令保留前后、侧走、斜走、纯转及走转宽范围；300之后15%样本覆盖0.15–0.20m/s的直行命令，其余保留宽范围。参考高度用于移动方向，脚掌俯仰只模仿有参考覆盖的直行。步频参考同脚周期0.563286s，权重0.15。

这一时期脚底角随课程倍率从±3°放大到±7.5°；观测噪声level1→2.5，外部速度推扰存在。不要将其与后期固定±3°、noise level1、无外部推扰的配置混同。

入口均为 `scripts/train_rsl_rl.py --config-name rigid4096_curriculum`，分别使用各自原工程快照。后段配置的 `algo.load_run` 指向901次检查点、`max_iterations=299`。

1200次实验并未全面达到速度/步频/全向目标；原始结果及分支说明见 [docs/ORIGINAL_1200_ARCHIVE.md](docs/ORIGINAL_1200_ARCHIVE.md) 和对应run中的RESULTS.md。后续改进不能倒推早期已成功。

## 5. 1200→1229→1279：均速修正和第一次柔性适配

### 均速修正30次

`03_mean1230`，从1199恢复。原实验有control、mean_only、mean_var三组各30次，**主线选择mean_only**。

- 将瞬时线速度项改为0.5秒窗口均值跟踪，标准差0.08；10次更新渐入。
- mean_var另外有1秒速度方差项（std0.02、权重0.2），属于对照组，不是所选模型。
- 主线结果 `mean_only/model_1229.pt` 对应累计1230次。
- 原入口 `scripts/train_rsl_rl.py --config-name mean_only`，全部源脚本和三组配置保留。

### 第一次恢复柔性50次，到1279

`04_flex1279`，从1229继续50次，原入口 `--config-name meanonly_flex50`。

- 真正48个被动柔性坐标，nq69/nv68/14电机；机械dt62.5µs，每20ms策略周期320机械步。
- 刚度与阻尼每坐标原值90%–110%；脚底安装pitch固定±3°，不再乘课程2.5到±7.5°。
- 4096环境；继承mean_only奖励和其他DR2.5，将旧奖励课程冻结在源检查点值。
- 最终 `model_1279.pt` 对应累计1280次，SHA `04595e1eb5736b6cac43f25613711156ed4ff02daf78684c860156aba3bbce41`。

## 6. 1279→自然站距：清洁起点和重新PPO

`05_natural_stance`。2026-10-10用户要求从前述起步BC改造前的1279重新开始，接受约212mm自然站距，不再追求183mm及Microduck强制比例对齐。

- `make_clean_roll_seed.py`：仅将actor最后层髋侧摆两行（1、10）的权重置0、bias对应物理角+0.08/-0.08；其他actor字段、归一化及critic继承1279。
- 新站姿由固定PD真实保持并采集站稳状态，不运行Stand599。
- 新优化器；冻结actor归一化，探索std0.25；取消旧官方参考/步频及mean速度替换项，保留自然宽度、进度、真实足端几何摆动/落脚等目标。
- r1/r2试验未作为权重父链；r3仍从 `clean_roll_seed.pt` 开始，256环境×24步，LR5e-5。虽计划跑更多更新，**选择model80，即本阶段81次更新**。
- 步幅改为同侧真实落脚点距离，不能把腾空位移和全周期步幅混用。

先前短时右支撑伸展教师/加权岭回归和0.6–1.2s双网络交接，属于起步研究与监督数据来源。它们不是最终flex模型的运行时补丁；见ENGINEERING_HISTORY的22–23节。

该阶段原入口：

```bash
uv run --no-sync --project project python run_walk_ppo.py --iterations 300 --parent clean_roll_seed.pt --config PPO_CLEAN_R3_CONFIG.yaml --logdir ppo_clean_r3_run --max-std .25
```

选用端点仅为81次，之后未选用的更新不能混入主线模型。配置中的计划轮数与CLI实际轮数均保留原始记录。

## 7. 自然站距→多方向第二轮：151 PPO + 后退监督 + 180 PPO

`06_direction_r2`，64/128等测试条件不能当作连续DR穷举。

1. 第一轮：从clean r3/model80，六类Twist指令；128×24、LR5e-5、冻结归一化/探索std；选择 `model_150.pt`，即151次更新。
2. 后退物理教师：组合原始干净1279的前后摆腿输出与当前侧摆/头部输出，在自由根节点真实物理仿真中筛选可持续后退且自然站距的轨迹。倒序周期/简单镜像失稳样本没有训练。
3. `bootstrap_backward.py` 做800步监督初始化，并保留认可起步/前进及成功转向样本；不是800轮PPO。
4. 第二轮：从 `backward_bootstrap.pt`，新的PPO优化器，180次更新，选 `model_179.pt`；探索std显式0.32，LR5e-5。

训练目标实际线速度约为输入Twist的0.65倍，+0.3对应约0.195m/s；输入值并不等于实测速度。后退进度/落脚投影/停滞检查改为沿指令的带符号量，纯转不再强求向前步幅。训练含4–6秒后的方向切换。

原命令：

```bash
uv run --no-sync --project project python run_walk_ppo.py --iterations 300 --parent accepted_parent.pt --config PPO_DIRECTION_CONFIG.yaml --logdir ppo_direction_run --max-std .35
# 实际std仍为父模型0.25；max-std只是上限，选取151更新的model150。
uv run --no-sync --project project python bootstrap_backward.py
uv run --no-sync --project project python run_walk_ppo.py --iterations 180 --parent backward_bootstrap.pt --config PPO_DIRECTION_R2_CONFIG.yaml --logdir ppo_direction_r2_run --max-std .32 --exploration-std .32
```

后来的未采用第三/第四轮只用于对照。本归档恢复工作区时用 `direction_speed_r2_restored.py` 和四池R2 anchor实现覆盖第二轮工程中后来改写的owner。这个恢复是已知代码语义的重建，**不声称恢复后的R2源码字节等于训练当日未保存的原文件**。

## 8. 对称默认站姿

旧站姿取自某次站立输出再调整，左右有平衡补偿。对称化以镜像均值为基础，将髋偏航、头部偏航/侧摆归零，保留约212mm站距并做平脚俯仰耦合。

GUI电机ID1–14顺序，单位rad：

```text
[0, 0.080, 0.394, -0.061, -0.455,
 0, -0.080, -0.394, 0.061, 0.455,
 0.019, 0.060, 0, 0]
```

训练14关节顺序（左右腿之间放颈/头）：

```text
[0, 0.080, 0.394, -0.061, -0.455,
 0.019, 0.060, 0, 0,
 0, -0.080, -0.394, 0.061, 0.455]
```

名义+16DR的固定PD站姿17/17站稳，物理脚间距约212.3–212.5mm。**这只是站姿目标，不是新的站立策略。** 行走的观测减基准/动作加基准REFERENCE继续使用原值（髋/踝±0.384），不能直接换成新的站姿数组，否则旧权重输入输出的含义会改变。

## 9. 最后一次柔性适配：ppo_flex100，选取51更新

`07_flex_final`，只从第二轮model179加载actor/critic；不加载第三/第四轮权重。

- 新独立 `DirectionFlex` owner基于StartupFlex；没有绕过原RigidStartCourse仅支持刚性的断言。
- 真实48柔性自由度，名义刚度100/500/800/1500，名义阻尼3；各坐标刚度与阻尼独立±10%。
- 踝部±3°沿用脚底安装pitch几何随机化，**不是±3°电机编码器零位偏置**。
- 真正柔性模型中采64组参数，每组3/4秒两个平衡时刻，共128状态；完整保存69维qpos、68维qvel及非零被动平衡变形和相配物理参数，不能拿刚体21维状态直接填柔性模型。
- 保留集动作标签全部由原第二轮自身重新计算，旧教师动作不再直接作为最后适配标签。
- 64×24、LR3e-5、冻结actor归一化，探索std0.25，新优化器；原计划100次，用户中途改为先50次。
- 最近完整检查点model50实际完成51次；停止后不继续训练。原第二轮保留以便对照。

实际运行入口已单独归档为 `run_walk_ppo_training_used.py`；后来加入更多诊断的 `run_walk_ppo.py` 不能倒推为已运行过的版本。

```bash
uv run --no-sync --project project python run_walk_ppo_training_used.py --iterations 100 --parent parent_r2.pt --config PPO_FLEX100_CONFIG.yaml --logdir ppo_flex100 --max-std .25 --exploration-std .25
# 本次已中途停止。若新建试验严格需要50次，可用--iterations 50，最后编号49。
```

## 10. 最终DR范围、输入输出与验证

|项目|最终启用范围|
|---|---|
|48柔性坐标刚度/阻尼|名义值0.9–1.1倍|
|左右脚底安装pitch|±3°|
|脚部/踝壳接触摩擦|0.25–1.75|
|armature|0.75–1.25倍|
|躯干质量/惯量|0.875–1.125倍|
|躯干/头部COM各轴|±15.675mm|
|IMU安装旋转|0–6°|
|均匀观测噪声|level1；q±.001rad，dq±.17293rad/s，gyro±.02075rad/s，gravity±.01|
|IMU/关节速度延迟|0或20ms / 20ms|
|未启用|电机参数、KP/KD、关节摩擦DR及外部推扰|

通用质量/COM开关false不表示自定义provider未随机化；以实际数组验证为准。柔性状态恢复检查已确认全部随机物理字段与站稳库匹配、48个被动平衡坐标未被重置成0。

部署契约：单个均值actor，float32[1,61]→float32[1,14]，50Hz；61项为gyro3、gravity3、相对关节位置14、延迟速度14、原始动作历史14、命令13。critic76只用于训练。没有运行时世界速度/高度/足力起步助手。髋侧摆动作尺度0.0366667，颈0.0966667，其余0.12；精确值、限位和KP/KD均在消费者代码。

### 8组新柔性DR的前后配对闭环

每模型六指令×8组，共96轨迹；先真实固定PD站稳4秒，再走12秒。参数、均匀噪声和IMU延迟抽样成对匹配；真实观测Torch/ONNX误差<2e-5，目标映射逐帧核对，Stand599推理0次，无运行中qpos编辑。新模型48轨迹均平衡通过。

|指令|原第二轮在相同柔性场景：持续/粗方向通过|51更新候选：持续/粗方向通过|
|---|---:|---:|
|前进|8/4|8/5|
|后退|7/2|7/3|
|纯左转|6/6|6/6|
|纯右转|7/7|8/8|
|走左转|8/7|8/7|
|走右转|8/8|8/8|

分母均8。粗前后方向允许净方向角<60°、航向净变化<90°；纯转/走转要角速度方向持续正确。纯转门限不代表零平移；本次不单独检查运动中指令切换。

能力基本保留，右转有恢复；**不能称全部DR方向与速度合格**。新版前进稳态速度0.181–0.288m/s，持续后退组幅值0.126–0.357m/s；多个前后命令条件严重偏航。case1有高频小步/低抬脚和后退后续停滞。代表视频case7六方向合格，不能以这一个视频代替其他条件。

此前刚体第二轮的159条件/指令扩大验收也未全部通过，见 `evidence/xduck_r2_dr_audit_20261010/`。不把8DR结果当作连续参数空间的穷举，也未真机验证。

## 11. 完整性校验与恢复工程

```bash
# 从本目录运行，只做哈希校验，不加载模型、不训练
uv run --no-project python tools/verify_package.py --runtimes

# 在新空目录中恢复全部8个历史工程、参数、模型及数据，不自动训练
uv run --no-project python tools/restore_workspace.py --workspace /path/to/new_empty_workspace
```

运行环境用八阶段源码的基础快照+差异覆盖保存，包含依赖声明/锁文件及资源；恢复时先逐文件SHA校验，再在新工作区重定位服务器绝对路径。归档里的原始配置/脚本保留原路径，便于审计。`restore_workspace.py` 会补齐关键checkpoint别名和历史兄弟目录输入，并恢复被后续实验覆盖的R2 owner。

这些工具不安装CUDA/驱动，也不自动启动旧supervisor。GPU训练需相容的Linux/NVIDIA、相应MuJoCo/Warp及锁定依赖；从工作区的相应 `project/` 用uv重建。checkpoint恢复并未保存每一步当时的具体物理状态/所有RNG；本包可恢复已保存源码/配置/权重/数据，不保证重新运行逐比特复现训练轨迹。重新跑全链时需将每个下游parent改指向你新生成的上游权重，并把日志/输出放在新目录。

旧版源快照保留上游LICENSE。未选用/失败的原始结果作为过程证据保留；它们不是最终模型父权重。参考历史细节见 `docs/ENGINEERING_HISTORY.md`，正文中的原本地路径仅供溯源，归档内的有效入口以本README和manifest为准。
