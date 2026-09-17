#include "battery_meter.h"
#include "cmsis_os.h"
#include "usart.h"
#include "FreeRTOS.h"
#include "task.h"
#include <string.h>

/* The confirmed 3.3 V TTL meter connects to P10 UART without level shifting. */

#define METER_ADDRESS 1U
#define METER_STALE_MS 3000U
static battery_meter_sample_t latest;
static uint8_t valid;

static uint16_t modbus_crc(const uint8_t *p, uint8_t len)
{
  uint16_t crc = 0xffffU;
  uint8_t i, bit;
  for (i = 0; i < len; ++i) {
    crc ^= p[i];
    for (bit = 0; bit < 8; ++bit)
      crc = (crc & 1U) ? (uint16_t)((crc >> 1) ^ 0xa001U) : (uint16_t)(crc >> 1);
  }
  return crc;
}

uint8_t BatteryMeter_GetSnapshot(battery_meter_sample_t *sample)
{
  uint8_t ok;
  if (sample == 0) return 0U;
  taskENTER_CRITICAL();
  *sample = latest;
  ok = valid;
  taskEXIT_CRITICAL();
  return (uint8_t)(ok && (uint32_t)(HAL_GetTick() - sample->sample_tick_ms) <= METER_STALE_MS);
}

static void poll_meter(void)
{
  uint8_t request[8] = {METER_ADDRESS, 0x04, 0, 0, 0, 3, 0, 0};
  uint8_t reply[11];
  uint16_t crc = modbus_crc(request, 6);
  battery_meter_sample_t sample;
  request[6] = (uint8_t)crc;
  request[7] = (uint8_t)(crc >> 8);
  /* Only this task uses USART1. Clear a partial prior response before sending. */
  __HAL_UART_FLUSH_DRREGISTER(&huart1);
  if (HAL_UART_Transmit(&huart1, request, sizeof(request), 30) != HAL_OK ||
      HAL_UART_Receive(&huart1, reply, sizeof(reply), 250) != HAL_OK) return;
  crc = modbus_crc(reply, 9);
  if (reply[0] != METER_ADDRESS || reply[1] != 0x04 || reply[2] != 6 ||
      reply[9] != (uint8_t)crc || reply[10] != (uint8_t)(crc >> 8)) return;
  sample.percent = (uint16_t)((reply[3] << 8) | reply[4]);
  sample.voltage_centi_v = (uint16_t)((reply[5] << 8) | reply[6]);
  sample.alarm = (uint16_t)((reply[7] << 8) | reply[8]);
  if (sample.percent > 100U || sample.voltage_centi_v == 0U || sample.alarm > 1U) return;
  sample.sample_tick_ms = HAL_GetTick();
  taskENTER_CRITICAL();
  latest = sample;
  valid = 1U;
  taskEXIT_CRITICAL();
}

static void StartBatteryMeterTask(void const *argument)
{
  (void)argument;
  for (;;) {
    poll_meter();
    osDelay(1000U);
  }
}

void BatteryMeter_TaskInit(void)
{
  osThreadDef(batteryMeter, StartBatteryMeterTask, osPriorityLow, 0, 384);
  (void)osThreadCreate(osThread(batteryMeter), 0);
}
