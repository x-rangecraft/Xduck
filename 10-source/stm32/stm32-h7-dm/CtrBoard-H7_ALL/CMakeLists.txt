cmake_minimum_required(VERSION 3.20)

project(CtrBoard_H7_ALL C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Explicit maintenance image: one populated route, incompatible with the
# production robotd's required 14-slot stream. Never enable automatically.
option(H7DM_BENCH_ID1_ONLY "Build an isolated CAN1 ID1 commissioning image" OFF)
set(TARGET_NAME CtrBoard-H7_ALL)
set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/STM32H723VGTX_FLASH.ld)
set(DM_OPENOCD_INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/openocd/dm-cmsis-dap-port-1-2.cfg)

set(CPU_FLAGS
  -mcpu=cortex-m7
  -mthumb
  -mfpu=fpv5-d16
  -mfloat-abi=hard
)

add_executable(${TARGET_NAME}.elf
  App/Src/dmusb_motor_bridge.c
  Freestanding/Src/freestanding_runtime.c
  Core/Startup/startup_stm32h723xx_gcc.s
  Core/Src/main.c
  Core/Src/gpio.c
  Core/Src/freertos.c
  Core/Src/adc.c
  Core/Src/dma.c
  Core/Src/fdcan.c
  Core/Src/octospi.c
  Core/Src/spi.c
  Core/Src/tim.c
  Core/Src/usart.c
  Core/Src/stm32h7xx_it.c
  Core/Src/stm32h7xx_hal_msp.c
  Core/Src/stm32h7xx_hal_timebase_tim.c
  Core/Src/system_stm32h7xx.c
  Bsp/Src/BSP_CAN.cpp
  Hardware/Src/Motor.cpp
  Hardware/Src/bmi088.c
  App/Src/motor_task.c
  App/Src/motor_app.cpp
  App/Src/h7spi_protocol.c
  App/Src/dmusb_protocol.c
  App/Src/dmusb_task.c
  App/Src/mahony_ahrs.c
  App/Src/imu_task.c
  USB_DEVICE/App/usb_device.c
  USB_DEVICE/App/usbd_desc.c
  USB_DEVICE/App/usbd_cdc_if.c
  USB_DEVICE/Target/usbd_conf.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_pcd.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_pcd_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_ll_usb.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_rcc.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_rcc_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_flash.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_flash_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_gpio.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_hsem.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_dma.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_dma_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_mdma.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_pwr.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_pwr_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_cortex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_i2c.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_i2c_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_exti.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_adc.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_adc_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_fdcan.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_ospi.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_spi.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_spi_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_tim.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_tim_ex.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_uart.c
  Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_uart_ex.c
  Middlewares/Third_Party/FreeRTOS/Source/croutine.c
  Middlewares/Third_Party/FreeRTOS/Source/event_groups.c
  Middlewares/Third_Party/FreeRTOS/Source/list.c
  Middlewares/Third_Party/FreeRTOS/Source/queue.c
  Middlewares/Third_Party/FreeRTOS/Source/stream_buffer.c
  Middlewares/Third_Party/FreeRTOS/Source/tasks.c
  Middlewares/Third_Party/FreeRTOS/Source/timers.c
  Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS/cmsis_os.c
  Middlewares/Third_Party/FreeRTOS/Source/portable/MemMang/heap_4.c
  Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1/port.c
  Middlewares/ST/STM32_USB_Device_Library/Core/Src/usbd_core.c
  Middlewares/ST/STM32_USB_Device_Library/Core/Src/usbd_ctlreq.c
  Middlewares/ST/STM32_USB_Device_Library/Core/Src/usbd_ioreq.c
  Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Src/usbd_cdc.c
)

set_target_properties(${TARGET_NAME}.elf PROPERTIES LINKER_LANGUAGE C)

target_include_directories(${TARGET_NAME}.elf PRIVATE
  Freestanding/Include
  Core/Inc
  Bsp/Inc
  Hardware/Inc
  App/Inc
  USB_DEVICE/App
  USB_DEVICE/Target
  Drivers/STM32H7xx_HAL_Driver/Inc
  Drivers/STM32H7xx_HAL_Driver/Inc/Legacy
  Middlewares/Third_Party/FreeRTOS/Source/include
  Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS
  Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1
  Middlewares/ST/STM32_USB_Device_Library/Core/Inc
  Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc
  Drivers/CMSIS/Device/ST/STM32H7xx/Include
  Drivers/CMSIS/Core/Include
)

target_compile_definitions(${TARGET_NAME}.elf PRIVATE
  USE_HAL_DRIVER
  STM32H723xx
  DM_CAN_FD_MOTOR=1
  H7DM_MOTOR_AUTO_ENABLE=0
  H7DM_BENCH_ID1_ONLY=$<BOOL:${H7DM_BENCH_ID1_ONLY}>
  DMUSB_BINARY_PROTOCOL=1
)

target_compile_options(${TARGET_NAME}.elf PRIVATE
  ${CPU_FLAGS}
  -ffreestanding
  -ffunction-sections
  -fdata-sections
  -Wall
  -Wextra
  -Wno-unused-parameter
  $<$<COMPILE_LANGUAGE:C>:-Wno-missing-field-initializers>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
)

target_link_options(${TARGET_NAME}.elf PRIVATE
  ${CPU_FLAGS}
  -T${LINKER_SCRIPT}
  -Wl,-Map=${TARGET_NAME}.map
  -Wl,--gc-sections
  -Wl,--print-memory-usage
  -nostdlib
  -nodefaultlibs
  -nolibc
)

target_link_libraries(${TARGET_NAME}.elf PRIVATE
  gcc
)

add_custom_command(TARGET ${TARGET_NAME}.elf POST_BUILD
  COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${TARGET_NAME}.elf>
  COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:${TARGET_NAME}.elf> ${TARGET_NAME}.hex
  COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${TARGET_NAME}.elf> ${TARGET_NAME}.bin
  COMMENT "Generating ${TARGET_NAME}.hex and ${TARGET_NAME}.bin"
)

find_program(OPENOCD_EXECUTABLE openocd)

if(OPENOCD_EXECUTABLE)
  add_custom_target(flash
    COMMAND ${OPENOCD_EXECUTABLE}
      -f ${DM_OPENOCD_INTERFACE}
      -f target/stm32h7x.cfg
      -c "init; assert_stm32h7_dev_id 0x483 STM32H723; program $<TARGET_FILE:${TARGET_NAME}.elf> verify reset exit"
    DEPENDS ${TARGET_NAME}.elf
    USES_TERMINAL
    VERBATIM
  )

  add_custom_target(reset
    COMMAND ${OPENOCD_EXECUTABLE}
      -f ${DM_OPENOCD_INTERFACE}
      -f target/stm32h7x.cfg
      -c "init; assert_stm32h7_dev_id 0x483 STM32H723; reset run; shutdown"
    USES_TERMINAL
    VERBATIM
  )
endif()
