#include "motor_task.h"
#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"
#include "motor_app.h"
#if H7DM_CAN_CAPTURE
#include "can_capture.h"
#endif

#include <string.h>

osThreadId motorTaskHandle;

static const TickType_t kMotorTaskCadenceTicks[] = {
  1U,
  1U,
  1U,
  1U,
};

static void StartMotorTask(void const *argument)
{
  TickType_t last_wake_tick;
  unsigned int cadence_index;
  (void)argument;

  last_wake_tick = xTaskGetTickCount();
  cadence_index = 0U;
  /* Apply startup register settings when enabled by the selected CAN mode.
   * This does not enable torque. */
  osDelay(100U);
  (void)MotorApp_ConfigureStartupRegisters();

#if H7DM_MOTOR_AUTO_ENABLE
  osDelay(3500U);
  MotorApp_EnableDefault();
#endif

  for (;;) {
    TickType_t delay_ticks;
    TickType_t now_tick;

    MotorApp_ProcessPendingSpiMotorCommand();
    MotorApp_Tick();
#if H7DM_CAN_CAPTURE
    CanCapture_Poll();
#endif
    delay_ticks = kMotorTaskCadenceTicks[cadence_index];
    now_tick = xTaskGetTickCount();
    if ((now_tick - last_wake_tick) >= delay_ticks) {
      last_wake_tick = now_tick;
    }
    vTaskDelayUntil(&last_wake_tick, delay_ticks);
    cadence_index++;
    if (cadence_index >= (sizeof(kMotorTaskCadenceTicks) / sizeof(kMotorTaskCadenceTicks[0]))) {
      cadence_index = 0U;
    }
  }
}

void MotorTask_Init(void)
{
  osThreadDef(motorTask, StartMotorTask, osPriorityNormal, 0, 1024);
  motorTaskHandle = osThreadCreate(osThread(motorTask), NULL);
}

/**********************************END OF FILE***********************************/
