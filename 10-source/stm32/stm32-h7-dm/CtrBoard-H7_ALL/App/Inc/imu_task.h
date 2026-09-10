#ifndef __IMU_TASK_H
#define __IMU_TASK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define IMU_FLAG_SENSOR_OK  (1U << 0)
#define IMU_FLAG_CALIBRATED (1U << 1)
#define IMU_FLAG_MAG_VALID  (1U << 2)

typedef struct {
  float base_ang_vel[3];
  float projected_gravity[3];
  float quaternion[4];
  uint32_t sample_tick_ms;
  uint32_t sample_sequence;
  uint8_t flags;
} imu_snapshot_t;

void IMU_TaskInit(void);
uint8_t IMU_GetSnapshot(imu_snapshot_t *snapshot);

/* Optional board-frame magnetometer input. The MC02 board has no onboard
 * magnetometer; a future external driver can feed calibrated values here. */
void IMU_SetMagnetometer(const float mag[3], uint8_t valid);

#ifdef __cplusplus
}
#endif
#endif
