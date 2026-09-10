# Xduck / xrange 工程工作区

本目录是本地工程根目录。目录布局、文件归属、RK3566 SSH 配置、上传和编译流程统一由
[`00-docs/ENGINEERING_WORKFLOW.md`](00-docs/ENGINEERING_WORKFLOW.md) 约束。

常用入口：

```sh
./60-tools/rk3566-workflow.sh check   # 只读预览同步差异
./60-tools/rk3566-workflow.sh all     # 上传后在 RK3566 原生 release 编译
./60-tools/ssh-rk3566.sh              # 连接固定目标机
./60-tools/check-layout.sh            # 检查本地目录规范
```

不要在根目录新增临时源码、日志或构建目录。

## Git 工作目录

本项目位于 `x-rangecraft/DuckResource` 仓库的 `Xduck/` 子目录。
本机开发目录为 `/Users/mac/xinchen/MicriDuck/DuckResource/Xduck`。
Android、RK3566、STM32 和工具源码统一由仓库根目录的 Git 管理，RK3566 不再是嵌套仓库。

```sh
cd /Users/mac/xinchen/MicriDuck/DuckResource/Xduck
git status --short -- .
git add .
git commit -m "Describe the Xduck change"
git push origin main
```

构建缓存、生成的固件和 APK、日志、临时文件及密钥不提交；这些目录仅保留占位文件。
芯片厂商提供的预编译依赖库和运行所需模型保留在源码目录中。
旧目录 `/Users/mac/xinchen/MicriDuck` 中原有工程保留为迁移前备份，后续修改在本目录完成。
远端 RK3566 目录保持 `/home/xduck1/xrange`，更新后的工程规范将在下一次显式源码同步时传到板端。
