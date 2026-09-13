# xrange 工程目录、SSH、同步与编译规范

状态：强制执行  
生效日期：2026-09-04  
适用范围：本地 `/Users/mac/xinchen/MicriDuck/DuckResource/Xduck` 与 RK3566 `/home/xduck1/xrange`

本文是工程文件归属、目标机身份、代码上传和板端编译的唯一操作规范。新增模块时先更新本文，禁止为了临时方便绕过分区。

## 1. 两端角色与唯一事实源

| 位置 | 角色 | 是否允许直接改源码 |
|---|---|---|
| 本地 `/Users/mac/xinchen/MicriDuck/DuckResource/Xduck` | 所有源码、文档和部署脚本的唯一事实源 | 是 |
| RK3566 `/home/xduck1/xrange` | 只读式源码镜像、原生构建机、板端产物和日志 | 否 |

所有代码修改必须先在本地完成并审查差异，再单向同步到 RK3566。不得在远端 `10-source/` 中修代码；确需板端临时诊断时放入 `90-temp/`，验证后回到本地实现正式修改。

## 2. 固定目录架构

两端使用相同一级分区：

```text
<工程根目录>/
├── 00-docs/       规范、架构、硬件资料
├── 10-source/     可编辑源码或远端源码镜像
├── 20-build/      编译缓存、中间文件、虚拟环境
├── 30-artifacts/  可交付二进制、固件、校验清单
├── 40-deploy/     SSH 配置、部署脚本和发布配置
├── 50-logs/       build/test/deploy 工程日志与实验数据
├── 60-tools/      统一操作入口
└── 90-temp/       可丢弃临时文件，不得保存唯一副本
```

禁止在工程根目录或用户主目录散放源码、`target/`、`build/`、日志、下载包和临时脚本。

### 2.1 本地文件归属

| 内容 | 固定位置 |
|---|---|
| Android BLE App 源码 | `10-source/android/duck-ble/` |
| Android SDK/JDK/Gradle 与构建缓存 | `20-build/android/` |
| Android APK 与校验清单 | `30-artifacts/android/` |
| Android 构建/测试日志 | `50-logs/{build,test}/android/` |
| 策略模型管理/转换源码 | 两文件策略由 `robotd/src/models.rs`、`custom_policy.rs` 与 `custom_policy_worker.py` 管理；仅供研发分发、不上传板端的自包含资料包位于 `00-docs/XDUCK_POLICY_SDK/`；兼容权重转换保留 `model_import.py`；IPC 由 `duck-ipc-proto` 定义，Web 面板由 `mediad` 提供 |
| 模型导入回归工具与日志 | `60-tools/test-model-import.py`、`test-web-models.cjs`；`50-logs/test/model-*` |
| 电机实验接口 | `robotd/src/experiment.rs` 为控制与数据所有者；`mediad/src/experiment_tasks_http.rs` 提供 HTTP 适配；文档和 PC 示例位于 `mediad/webclient/`，实时接口采集 CSV 保存在调用者电脑；`robotd/src/experiment_tasks.rs` 管理预上传任务、固定缓冲执行和 `/var/lib/robotd/experiments/` 持久数据，`mediad/src/experiment_tasks_http.rs` 只流式转发文件。任务结果由电脑下载校验后显式删除；同目录 `joint_sine_sweep.py` 为正弦扫描与失能通信诊断工具 |
| 本体日志下载接口 | `mediad/src/journal_http.rs` 按请求流式读取 Journal，不在设备保存导出副本 |
| RK3566 Microduck Rust workspace | `10-source/rk3566/microduck/` |
| STM32↔RK3566 DMUSB v4 协议事实源 | `10-source/rk3566/STM32_DMUSB_V4_PROTOCOL.md` |
| STM32 H7 固件 | `10-source/stm32/stm32-h7-dm/` |
| GF43X40-10 / I2RT 通信与辨识工具 | `10-source/tools/motor/gf43x40-10-i2rt/` |
| GF43X40-10 厂商参数与 I2RT 指令手册 | 仓库同级 `../Motor/电机参数.pdf`、`../Motor/I2RT_CAN通信协议与指令手册.docx` |
| 电机建模与辨识方案 | `00-docs/hardware/motors/` |
| STM32 CMake 输出 | `20-build/stm32/stm32-h7-dm/` |
| Python 虚拟环境与缓存 | `20-build/tools/motor/gf43x40-10-i2rt/` |
| 电机采集数据与拟合结果 | `50-logs/test/motor/gf43x40-10-i2rt/data/` |

源码中的 `data` 和 `.venv` 允许使用相对符号链接指向上述固定位置，以兼容脚本默认参数；真实文件不得复制成多份。

### 2.2 RK3566 文件归属

| 内容 | 固定位置 |
|---|---|
| Microduck 源码镜像 | `/home/xduck1/xrange/10-source/rk3566/microduck/` |
| Cargo release 缓存 | `/home/xduck1/xrange/20-build/rk3566/microduck/` |
| 已验证板端二进制 | `/home/xduck1/xrange/30-artifacts/rk3566/microduck/<构建批次>/` |
| 部署输入 | `/home/xduck1/xrange/40-deploy/rk3566/microduck/` |
| 构建日志 | `/home/xduck1/xrange/50-logs/build/rk3566/microduck/` |
| 测试日志 | `/home/xduck1/xrange/50-logs/test/rk3566/microduck/` |
| 部署日志 | `/home/xduck1/xrange/50-logs/deploy/rk3566/microduck/` |
| 板端构建工具 | `/home/xduck1/xrange/60-tools/` |

部署后的策略覆盖文件、两条 ONNX 历史和原子清单由 `robotd` 单独写入 `/var/lib/robotd/policies/`（`StateDirectory=robotd`）。不得覆盖发布包中的原始 ONNX。模型导入转换器和两文件策略运行环境固定在 `/var/lib/robotd/model-python/`，由已授权部署时运行 `scripts/setup-model-import.sh` 安装；它是部署运行依赖，不属于工程编译缓存。`ROBOT_MODEL_PYTHON` 可显式指定其他受管理的 Python。两文件策略必须通过 `bubblewrap` 与 `setpriv` 以 `robot-policy` 用户在无网络、无硬件设备、只读主机文件系统和资源限制下运行；目标机缺少任一工具时部署和策略加载均须失败。模型替换与回滚只能由控制循环在放松且最新 STM32 反馈确认全部配置电机失能时执行。每个兼容旧版导入在同一清单中绑定 Kp、Kd 与动作缩放，切换/回滚一并恢复；两文件版本的逐关节 MIT 参数由 policy.py 每帧提供并经过平台硬边界校验。`duck-control` 在每帧目标中承载这些参数，未绑定的旧版本继续使用原程序参数。

## 3. RK3566 固定身份与 SSH 配置

| 字段 | 固定值 |
|---|---|
| SSH 别名 | `xrange-rk3566-1` |
| 地址 | `192.168.50.121` |
| 端口 | `888` |
| 用户 | `xduck1` |
| 设备对外名称 | `Xduck`（由 configd 持久保存，WebRTC 控制台和蓝牙广播共用） |
| hostname | `rk3566-1` |
| machine-id | `0160c5af701d40ad82a694adb73caf30` |
| SoC/板型 | `rockchip,rk3566-kickpi-k11c` |
| 架构 | `aarch64` |
| OS | Debian 12 |
| 当前内核基线 | `6.1.141`（升级内核不改变设备身份） |
| SSH ED25519 指纹 | `SHA256:sSfs38HNILZZKr274j8w6qSW6yLFN3cswNrxDlNjUIY` |

连接必须使用项目内配置，不能在脚本中再次手写 IP、端口或用户：

```sh
./60-tools/ssh-rk3566.sh
```

配置文件位于 `40-deploy/rk3566/ssh/config`，固定公钥位于 `40-deploy/rk3566/ssh/known_hosts`。如果 SSH 报主机公钥变化，应停止并在现场核实设备；禁止使用 `StrictHostKeyChecking=no`、删除 known_hosts 或静默接受新公钥。

## 4. Microduck 架构边界

以 `10-source/rk3566/microduck/docs/design/architecture.md` 和 `robotd-design.md` 为实现依据。Microduck 采用“单一状态所有者、独立 daemon、Unix socket JSON-RPC”架构，不按传统 Web 三层随意拆仓库。

| 层级 | 职责 | 模块 |
|---|---|---|
| L0 底层 | 硬件控制、运动学、里程计、声音与感知算法 | `duck-control`、`kinematics`、`odometry`、`sounds`、`pet-detect`、`duck-detect` |
| L1 契约 | IPC 数据契约、启动参数模型 | `duck-ipc-proto`、`robotd-params` |
| L2 核心服务 | 单一资源所有者 | `robotd`、`configd`、`updater/updaterd`、`tof/tofd` |
| L3 适配层 | 不拥有机器人状态的输入和传输 | `btd`、`padd`、`mediad` |
| L4 用户入口 | 本机与开发机 CLI | `robotctl`、`duckctl` |
| 工程层 | 打包、测试、安装和部署 | `xtask`、`test-support`、`scripts`、`hooks`、`deploy` |

必须保持的边界：

1. `robotd` 是唯一可驱动电机并作安全裁决的服务。
2. `configd`、`updaterd`、`btd` 必须能在 `robotd` 故障时继续工作。
3. `btd`、`padd`、`mediad` 只转发意图或拥有自己的媒体管线，不直接控制电机。
4. 一个状态只能有一个写入者；跨服务访问使用现有 IPC 契约。
5. Cargo workspace 必须整体保存，禁止为了“目录分层”物理拆散 crate。

## 5. 固定上传流程

先只读预览：

```sh
./60-tools/rk3566-workflow.sh check
```

确认差异后同步：

```sh
./60-tools/rk3566-workflow.sh sync
```

同步规则：

- 只允许本地 → RK3566 单向同步。
- 同步前校验 hostname、machine-id、aarch64、Debian 12 和远端管理标记。
- `.git/`、`target/`、`.DS_Store` 不上传。
- 同步不传播本机的 owner/group；远端工程统一归属 `xduck1:xduck1`。
- `--delete-delay` 仅作用于已验证的远端 Microduck 源码镜像目录。
- 远端源码目录产生的手工修改会被覆盖，不应作为工作成果保存。

## 6. 固定编译与产物流程

上传并编译：

```sh
./60-tools/rk3566-workflow.sh all
```

只编译已同步版本：

```sh
./60-tools/rk3566-workflow.sh build
```

板端构建固定执行 `cargo build --locked --release --bins`，使用 workspace 的 `default-members`，不构建仅供开发机使用的 `duckctl`。`--locked` 保证板端不得自行改写依赖锁；锁文件不完整时构建必须失败并回到本地修正。`CARGO_TARGET_DIR` 固定指向 `20-build/`，构建日志固定写入 `50-logs/build/`。

成功后仅将发布工作流列出的九个板端程序归档：`btd`、`configd`、`mediad`、`padd`、`robotctl`、`robotd`、`sounds`、`tofd`、`updaterd`。归档必须同时包含 `SHA256SUMS` 与 `BUILD-INFO.txt`，并验证每个文件为 ARM aarch64 ELF。

编译验证不等于部署。未经单独确认，不得写入 `/opt/robot`、修改 systemd unit、启动 daemon 或访问电机总线。

## 7. 固定部署与启动流程

首次部署必须获得明确授权，然后执行：

```sh
./60-tools/rk3566-workflow.sh deploy
```

部署脚本仍会重复校验 hostname、machine-id、架构、系统版本和管理标记，并且必须先调用固定构建脚本；禁止直接打包 `20-build` 中上一次留下的二进制。发布包先用板级开发密钥签名并通过 updater 健康门禁，校验通过后才写入 `/opt/robot` 和 `/etc/robot`。开发密钥固定放在远端 `40-deploy/rk3566/secrets/`，目录权限 `0700`、私钥权限 `0600`；该目录在源码仓库之外，严禁同步回本地、提交 Git 或复制到客户设备。开发板的 `/etc/robot/updater.toml` 明确启用 `allow_dev_keys`，但在正式 release 仓库配置完成前禁用定时检查和自动安装。

这台 K11C 的摄像头固定为 BL-1080P-S10（USB UVC，`05a3:9230`）。本地事实源 `40-deploy/rk3566/remote/99-xrange-camera.rules` 只匹配 UVC 的 `index=0` 视频采集接口，并创建稳定节点 `/dev/video-xrange-camera`；`index=1` 元数据接口不得交给 `mediad`。配置固定为 `camera = true`、`camera_backend = "uvc-mjpeg"`、`camera_device = "/dev/video-xrange-camera"`、`quality = "1080p30"`、`rotation = 0`。摄像头在 USB 上输出 MJPEG，由独立、可重启的采集管线通过 `intervideosink` 送入常驻 WebRTC 管线；摄像头拔掉后 `intervideosrc` 输出黑帧，信令和控制 DataChannel 必须保持，页面显示“摄像头已断开”，重插后采集管线自动恢复且不得重启 `mediad`。正常视频仍由 `mppjpegdec` 硬件解码，再由 `mpph264enc` 以 constrained-baseline、1 秒 GOP 编码给 WebRTC；空闲时不创建不需要的原始帧分支。`gstreamer1.0-nice`（`nicesrc`/`nicesink`）和 `intervideosrc`/`intervideosink` 是部署硬依赖，检查缺失时必须失败。禁止改成 720p60 软件 JPEG 解码来换取名义分辨率。物理端口是 USB 3.0 不代表设备会以 SuperSpeed 枚举：该摄像头自身是 USB 2.0 High-Speed（480 Mbit/s），这是正常能力而非降级。2026-09-09 已现场确认板载 RK809 的 SPK 播放正常，固定为 `audio.enabled = true`、`audio.device = "xduck_speaker"`（ALSA softvol 转发到 RK809；网页通过 `robot.volume` 调节 `Xduck` 控件，避免厂商 DAC 控件写入归零），使用已有 `/var/lib/robot/sounds` 音效库；麦克风抚摸识别仍保持默认关闭。`robot-boot-check.timer` 只启用供下次开机使用，不得在已运行很久的系统上手工立即触发。

这台样机当前也没有连接 ToF 传感器。`tofd` 二进制和 unit 必须随发布包保留，但 `tofd.service` 保持 disabled/inactive，不进行无意义的硬件轮询；以后接入传感器并确认 I²C 设备、权限和采样稳定后，再显式执行 `sudo systemctl enable --now tofd.service`。

STM32 H7 通过 `[bus].port = "/dev/ttyACM0"` 同时提供电机状态和主 BMI088 IMU 数据；主 IMU 不再占用独立 USB。通信唯一事实源是上述 DMUSB v4 文档，禁止凭旧版帧布局修改实现。`[bus].imu_port` 只作为未来双 IMU 的预留接口保留，当前版本不得打开或依赖它。运行中 STM32 USB 连续读失败达到门限时，`robotd` 必须清空速度和使能、标记为安全降级状态、关闭失效串口并在同一进程内循环重新打开和校验；重插 USB 不得要求重启 `robotd`。重连成功后保持未使能，必须由操作者重新初始化/使能，禁止自动恢复此前运动。STM32 固件自身在 100 ms 收不到主机命令时关闭全部电机路由，Linux 侧安全状态不能替代这条板端保护。电机控制要求 DMUSB v4 能力位 `0x0007`（空槽、完整使能列表和安全标定），并先同步位置限位；状态位 `0x0008` 表示本次启动已装载限位，旧固件仍保持只读。标定与默认站姿由 robotd 持久保存到 `/var/lib/robotd/motor-calibration.json`（systemd `StateDirectory=robotd`），重连后重新同步 STM32 RAM 限位。STM32 的 `App/Src/dmusb_motor_bridge.c` 负责固定槽到实际电机命令的转换，RK3566 在使能前检查固件能力、配置映射和在线/故障反馈。STM32 构建继续保持 `H7DM_MOTOR_AUTO_ENABLE=0`，开放权限不会自动使能。

电机反馈超时恢复与重新使能分离：STM32 在全体配置电机持续 1 秒反馈新鲜、实际失能且温度正常，IMU 健康、管理操作空闲、无其他锁存故障时，自动清除反馈超时位；仍保持失能并丢弃旧控制命令，必须由操作者重新初始化/使能。其他故障不由该路径清除。DMUSB 可选能力 `0x0010` 提供最近保护故障的电机 ID/槽位，`duck-control` 将故障发生与清除分别写入 robotd 的 Journal，当前故障清除不得删除历史。详细字段和恢复条件见协议 §7.4。

`robotd.service` 的停止超时固定为 5 秒。USB CDC 驱动异常时阻塞读可能无法响应 SIGTERM；systemd 必须在 5 秒后结束旧进程，避免一次更新把 30 秒健康门控窗口全部耗在停止旧版本上。不得通过延长健康门控掩盖这个退出问题。

K11C 厂商内核启用了 uinput，但没有 `xpad`、`joydev` 或可加载的内核模块。飞智 Dune Fox 接收器以 Xbox 360 兼容设备 `045e:028e` 枚举，但 Debian `xboxdrv 0.8.8` 不能把它的输入包转换成 evdev 事件。因此硬件适配固定在底层 `10-source/rk3566/microduck/drivers/flydigi-xpad/`：`xpad-usbd.service` 独占 USB interface 0，把 20 字节 Xbox 360 报告转换为标准 `/dev/input/event*`，`padd` 只读取标准 evdev，禁止在 `padd` 中解析该接收器的私有 USB 数据。驱动支持拔插自动重连，断开时立即销毁虚拟输入设备，使 `padd` 上报手柄不可用并触发安全回退。运行日志读取：`journalctl -u xpad-usbd -b`。

验收至少检查：当前 release 链接、服务的 active/enabled 状态、`systemctl --failed`、`robotctl health`、各进程实际可执行路径、30 秒 CPU/load/内存/温度采样。缺少 `/dev/ttyACM0` 时 `robotd` 的 degraded 是真实硬件状态，不能写成“全部功能正常”。

## 8. 日志系统：三个唯一事实源

日志完全沿用 Microduck 原始边界，只分为下面三类。服务运行日志不得复制成长期维护的 `.log` 文件，升级日志不得移入工程目录，构建和部署输出不得写进源码目录。

| 类别 | 内容 | 唯一存储位置 | 保留方式 |
|---|---|---|---|
| 服务运行日志 | `robotd`、`mediad`、`configd`、`btd`、`padd`、`tofd`、`updaterd` 的 stdout/stderr | systemd Journal；由 `journalctl` 读取，底层通常位于 `/var/log/journal/` | 总上限 500 MB、单文件 20 MB、最多 100 个轮转文件、最长 3 个月、压缩 |
| 升级持久日志 | 安装、拒绝、失败、回滚和每次升级的完整过程 | `/var/lib/robot/updater/update-log.jsonl` 与 `/var/lib/robot/updater/runs/*.jsonl` | update log 保留最近 200 条；完整 transcript 保留最近 20 次；逐条同步到持久存储 |
| 工程日志 | 编译、硬件/功能测试、部署操作输出 | `/home/xduck1/xrange/50-logs/{build,test,deploy}/` | 按平台、模块和时间戳分层；不进入发布包和源码仓库 |

### 8.1 从哪里读取服务运行日志

服务日志只从 Journal 读取。不得在工程目录中寻找或创建 `robotd.log`、`mediad.log` 等影子副本。

在本地通过固定 SSH 入口读取：

```sh
# 本次开机的某个服务日志
./60-tools/ssh-rk3566.sh "sudo journalctl -u robotd.service -b --no-pager"

# 最近 100 行
./60-tools/ssh-rk3566.sh "sudo journalctl -u mediad.service -n 100 --no-pager"

# 实时跟随；按 Ctrl-C 退出
./60-tools/ssh-rk3566.sh "sudo journalctl -u mediad.service -f"

# 上一次开机的日志
./60-tools/ssh-rk3566.sh "sudo journalctl -u robotd.service -b -1 --no-pager"

# 指定时间之后、只看 warning 及以上
./60-tools/ssh-rk3566.sh "sudo journalctl -u robotd.service --since '2026-09-04 08:00:00' -p warning --no-pager"

# 查看 Journal 当前占用和可读取的启动周期
./60-tools/ssh-rk3566.sh "sudo journalctl --disk-usage; sudo journalctl --list-boots"
```

可替换的服务名固定为 `updaterd`、`robotd`、`configd`、`btd`、`padd`、`mediad`、`tofd`。Journal 是二进制数据库，禁止用 `cat /var/log/journal/...` 读取。需要为一次故障留证时，可将命令输出导出到对应的 `50-logs/test/rk3566/microduck/<事件>-<时间戳>/`；导出件只是快照，Journal 仍是运行日志的唯一事实源。

### 8.2 从哪里读取升级持久日志

优先使用 Microduck 自带接口：

```sh
./60-tools/ssh-rk3566.sh "sudo robotctl update log"
./60-tools/ssh-rk3566.sh "sudo robotctl update show 42"
```

第一条列出升级历史；第二条读取编号为 `42` 的完整执行记录。仅当 `robotctl` 损坏时才直接读取 JSON Lines 文件：

```sh
./60-tools/ssh-rk3566.sh "sudo cat /var/lib/robot/updater/update-log.jsonl"
./60-tools/ssh-rk3566.sh "sudo cat /var/lib/robot/updater/runs/000042.jsonl"
```

升级记录在 `/var/lib`，不依赖 Journal，发布目录切换或回滚不得删除它。

### 8.3 从哪里读取工程日志

```sh
# 构建日志
./60-tools/ssh-rk3566.sh "ls -lt /home/xduck1/xrange/50-logs/build/rk3566/microduck/"

# 测试和实验记录
./60-tools/ssh-rk3566.sh "find /home/xduck1/xrange/50-logs/test/rk3566/microduck/ -maxdepth 2 -type f"

# 部署日志
./60-tools/ssh-rk3566.sh "ls -lt /home/xduck1/xrange/50-logs/deploy/rk3566/microduck/"
```

文件名必须带时间戳。`build/` 只保存编译输出，`test/` 保存可复现的验收和采样证据，`deploy/` 保存安装发布过程。`50-logs/runtime/` 为废弃路径，不得重新创建。

## 9. 修改纪律

1. 开始前运行 `git status --short -- .`，不得覆盖不属于本次任务的改动。
2. 修改源码只使用本地事实源；远端只用于编译、硬件测试和日志采集。
3. 构建产物不得提交到源码仓库，也不得从 `20-build/` 直接交付。
4. 工程日志按 `build/test/deploy` 分类并保留时间戳；服务运行日志和升级日志遵循第 8 节，临时输出进入 `90-temp/`。
5. 删除、覆盖、发布、安装或启动服务前必须明确目标和影响范围。
6. 目录规范变化必须同时更新本文、本地检查脚本和远端副本。

## 10. 日常检查

```sh
./60-tools/check-layout.sh
git status --short -- .
./60-tools/rk3566-workflow.sh check
```

三条命令分别检查目录归属、源码改动和待同步差异。

`check-layout.sh` 还会依据 `40-deploy/rk3566/CONTROL_FILES.sha256` 校验 SSH 配置、主机公钥、管理标记和流程脚本。需要修改这些控制文件时必须先审查影响，再显式更新校验基线，不能通过跳过检查规避。
