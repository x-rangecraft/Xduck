#ifndef BATTERY_METER_H
#define BATTERY_METER_H
#include <stdint.h>
typedef struct {
  uint16_t percent;
  uint16_t voltage_centi_v;
  uint16_t alarm;
  uint32_t sample_tick_ms;
} battery_meter_sample_t;
void BatteryMeter_TaskInit(void);
uint8_t BatteryMeter_GetSnapshot(battery_meter_sample_t *sample);
#endif
