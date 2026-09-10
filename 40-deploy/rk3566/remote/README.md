# xrange RK3566 工程区

本目录由本地 `/Users/mac/xinchen/MicriDuck/DuckResource/Xduck` 单向管理。

完整规范：`/home/xduck1/xrange/00-docs/ENGINEERING_WORKFLOW.md`

- 不得直接修改 `10-source/` 中的源码。
- 构建缓存只放 `20-build/`。
- 可交付文件只放 `30-artifacts/`。
- 工程日志只分为构建、测试、部署，分别放入 `50-logs/build`、`test`、`deploy`；禁止使用已废弃的 `50-logs/runtime`。
- 服务运行日志只从 systemd Journal 读取；升级持久日志只从 `/var/lib/robot/updater/` 或 `robotctl update log/show` 读取。完整命令见 `00-docs/ENGINEERING_WORKFLOW.md` 第 8 节。
- 临时诊断文件只放 `90-temp/`。
- 编译使用 `/home/xduck1/xrange/60-tools/build-rk3566-microduck.sh`。
- 编译验证不等于部署；未经确认不得修改 `/opt/robot` 或启动服务。
- 部署使用 `/home/xduck1/xrange/60-tools/deploy-rk3566-microduck.sh`；它会强制先执行固定 release 构建，不能打包旧缓存。开发私钥仅存放于 `40-deploy/rk3566/secrets/`，不得外传。
- 摄像头为 BL-1080P-S10（UVC `05a3:9230`）：udev 将采集接口固定为 `/dev/video-xrange-camera`，`mediad` 使用 `camera_backend = "uvc-mjpeg"`、`quality = "1080p30"`，通过 `mppjpegdec` 硬解和 `mpph264enc` 硬编输出 WebRTC。`gstreamer1.0-nice` 是浏览器 ICE 连接必需依赖。音频设备未连接，保持 `audio.enabled = false`。
- 设备对外名称固定为 `xduck1`，由 configd 持久保存，WebRTC 控制台与蓝牙广播都从该身份读取。
- 本机当前没有连接 ToF；`tofd` 随发布包保留，但保持 disabled/inactive，接入并验收传感器后再启用。
- K11C 厂商内核没有 `xpad` 和 `joydev`；底层 `xpad-usbd.service` 将飞智 Dune Fox USB 接收器 `045e:028e` 转成标准 `/dev/input/event*`，`padd` 只消费 evdev。该服务必须先于 `padd` 启动，且接收器拔插后自动重连。
