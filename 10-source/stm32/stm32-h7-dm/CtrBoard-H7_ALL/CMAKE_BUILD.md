# CMake build

This project now supports ARM GCC builds in addition to the original Keil project.

Prerequisites:

```sh
arm-none-eabi-gcc --version
cmake --version
```

Use a complete Arm GNU Toolchain that includes newlib headers and libraries.
Some Homebrew `arm-none-eabi-gcc` builds are configured `--without-headers` and
cannot build this firmware because standard headers such as `stdint.h` are
missing.

Configure and build:

```sh
XRANGE_ROOT="$(cd ../../../.. && pwd)"
STM32_BUILD_DIR="$XRANGE_ROOT/20-build/stm32/stm32-h7-dm/release"
cmake -S . -B "$STM32_BUILD_DIR" -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi-gcc.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build "$STM32_BUILD_DIR"
```

Outputs:

```text
$STM32_BUILD_DIR/CtrBoard-H7_ALL.elf
$STM32_BUILD_DIR/CtrBoard-H7_ALL.hex
$STM32_BUILD_DIR/CtrBoard-H7_ALL.bin
$STM32_BUILD_DIR/CtrBoard-H7_ALL.map
```
