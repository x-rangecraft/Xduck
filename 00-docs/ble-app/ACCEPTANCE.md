# BLE App 首版验收记录

日期：2026-09-07。此文区分已执行的软件验证和待执行的真实机器人联调。

## 已执行

- Android debug APK 构建成功，Android lint 无错误。
- Android 35 ARM64 模拟器成功安装并启动 APK；连接页、移动页实际截图保存在 `50-logs/test/android/`。
- 本地 btd 60 项测试通过，包含真实临时 Unix socket 的认证/独占/超时/重复指令/限频/零速度测试。
- Android/Rust 20字节移动 golden vector 一致；188字节状态解析、UTF-8跨片、缺片拒绝、未知值和14电机单位检查通过。
- 工程目录与控制文件校验通过。

构建、测试证据：`50-logs/{build,test}/android/`。APK 与 SHA256 位于 `30-artifacts/android/`。

## 0.1.1 配对弹窗修复（2026-09-07）

用户实机截图显示系统配对对话框仍打开，App 已显示“已暂停控制，返回后重新连接”。旧版 onPause 延迟断开只豁免“配对”阶段，进入 MTU/API/CCC 阶段就失去保护；失败又自动重连，可能重复发起配对。

- onPause 仍立即释放控制，连接关闭改由 onStop 处理。配对保护覆盖整个初始化流程，直到 App 回到前台才宣布 ready，后台不认证或接管。
- 首次连接/配对失败暂停重试；已建立连接丢失 bond 时不自动调用 createBond。显式点击设备允许重试。
- ConnectionLifecycleChecks 覆盖弹窗期间暂停、订阅先完成/恢复先发生两种顺序、延迟后台回调、失败/取消后停止重试、正常后台断开。
- 协议检查、生命周期回归检查、assembleDebug、lintDebug 和目录校验通过。APK versionCode=2，签名与 0.1.0 相同，可覆盖安装。
- 修复版尚待用户手机验证；不将无 Android 运行时的生命周期检查等同于实机蓝牙测试。本次没有更新机器人服务。

构建日志：`50-logs/build/android/pairing-fix-20260907.log`。

## 真机交付前仍需执行

2026-09-08 配对诊断：用户连续两次连接，机器人 D-Bus 分别记录
`RequestAuthorization`（1788831108.557066、1788831114.504146）。现有 bluer
0.17.4 的空回调默认拒绝该请求；只增加此回调又会将能力推导为 DisplayYesNo。
因此新增独立 NoInputNoOutput 代理，显式接受来自 bluetoothd 的当前适配器配对授权，
保留 PIN 与加密要求，拒绝未支持的口令交互及其他蓝牙服务授权。
Linux 内存消息回归测试 2 项通过。用户授权后已按签名更新流程部署
`0.10.0-dev.local.1788831881.g590b986`，更新健康门禁通过。
D-Bus 实测新代理 `/org/microduck/pairing_agent` 注册为 `NoInputNoOutput`，
运行 btd 的 SHA256 与归档产物一致（45ac699744cd000bbac1c7a4f65be02a70abc39ef405f484eb54bd4b3493852d）。
所有预期服务 active/enabled，tofd 依规范 disabled/stopped，失败单元为 0。
部署后 robot healthy、控制循环 50 Hz 无漏周期，摄像头 1080p30；30秒运行采样见
`50-logs/test/android/pairing-deployed-health-20260908.txt`。手机实际配对仍待重试确认。
证据：`50-logs/test/android/pairing-agent-confirmed-20260908.txt`。
发布编译日志：`50-logs/build/android/pairing-agent-native-20260908.log`。

| 项目 | 验收方式 | 当前状态 |
| --- | --- | --- |
| 安装机器人端新版本 | 按 ENGINEERING_WORKFLOW 的部署流程，确认重启范围 | 2026-09-08 用户授权部署，签名校验及健康门禁通过 |
| Android 8–11 / 12+ 扫描和加密配对 | 真机权限、蓝牙关闭/开启、配对取消、错误PIN | 待测 |
| 离线使用 | 手机关闭 Wi-Fi/移动网络，仍可连接、控制和看状态 | 待测 |
| 蓝牙扫描与自动重连 | 距离、干扰、锁屏、强杀App、机器人重启 | 待测 |
| 双摇杆移动 | 支撑条件下确认 vx/vy/yaw 方向、限速、双指操作、松手停车 | 待测 |
| 500 ms deadman | 停止发送指令并从 robotd 日志/电机反馈测量停车时间 | 待测 |
| 断线停车 | 主动断开、蓝牙关闭、超出范围，确认无运动自动恢复 | 待测 |
| 两台手机 | 第二台不能注入或接管第一台的已认证会话；第一台断线可恢复连接 | 待测 |
| 高风险动作 | 初始化/放松/策略启停/坐站/拾取/踢球/翻滚逐项低风险验收 | 待测 |
| 电机状态 | 14槽实际映射、故障和过期标志、温度/力矩对照 STM32 数据 | 待测 |
| 链路与 CPU | 2Hz状态、10Hz命令、延迟分布、MTU=23与较大MTU、CPU采样 | 待测 |
| 声音 | 音频硬件与资源实际可用后验证；当前样机规范明确关闭音频 | 待硬件 |

本次没有通过 App 发出实机运动、初始化、放松或动作命令。不能将模拟器截图和协议测试视为真实蓝牙/电机验收。

## 0.1.2 回复通道与重连修复（2026-09-08）

实机 btd 日志确认 RequestAuthorization accepted=true，PIN 认证成功；之后出现 central unsubscribed 及重复 write with no subscription。用户确认 App 报 system.authenticate 超时。尚不能仅凭这些日志确认通知最初丢失的底层原因。

App 改为每次 GATT 连接重置 CCC（关闭→开启），并用 hello 验证完整请求/回复往返，成功后才宣布连接就绪。只在认证和 ble.info 验证成功后清零重试计数，回复探测失败直接暂停重试。非必要 RSSI 查询超时不再断开连接。

0.1.2 APK versionCode=3，签名与旧版相同；assembleDebug、lintDebug、已有协议及生命周期检查通过。构建日志位于 50-logs/build/android/subscription-fix-20260908.log。修复版仍需手机复测，不能标记连接问题已实机解决。此次未更改机器人服务。

## 0.1.3 控制优先级修复（2026-09-08）

固定仲裁：手柄 > 蓝牙有效租约 > 网页拖动 > 键盘。覆盖手柄重新接入、蓝牙释放但保持连接、蓝牙断连不抢走手柄、低优先级停车包、租约超时、网页拖动与持有键盘的切换、硬件抢占后清除旧输入。

本地验证：robotd 意图测试 15 项、btd 60 项、mediad 30 项全部通过；真实网页脚本的 Node VM 输入事件回归及完整脚本语法检查通过；Android 协议/生命周期检查、assembleDebug、lintDebug 通过。APK 0.1.3/versionCode 4、名称 Xduck，签名与之前版本一致。日志：50-logs/test/android/control-priority-{robotd,transports}-20260908.log、50-logs/build/android/Xduck-0.1.3-20260908.log。

部署完成：用户明确授权后，固定流程同步、原生 release 构建（15m41s）和驱动解析自测通过；签名更新 0.10.0-dev.local.1788836195.g590b986 经健康门禁提交成功。各服务 active/enabled，实际程序路径属于新 release；robotd/btd/padd/mediad 哈希与本次归档一致。systemctl 无失败单元；tofd 按硬件配置 disabled/inactive。30 秒采样 CPU 空闲 85–87%、可用内存 2729 MB、温度 40.625°C，机器人 50 Hz、0 missed、总线正常、IMU ready、相机 30 fps。只读状态显示手柄与蓝牙均断开，控制源 drag。日志：50-logs/deploy/control-priority-20260908.log、50-logs/test/android/control-priority-deployed-health-20260908.txt。实机插拔手柄及蓝牙释放仍待操作者验证；未进行电机动作测试。
