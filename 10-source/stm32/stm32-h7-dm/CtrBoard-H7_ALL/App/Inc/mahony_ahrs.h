#ifndef __MAHONY_AHRS_H
#define __MAHONY_AHRS_H

#include <stdint.h>

typedef struct {
  float q[4]; /* body to world, scalar first: w, x, y, z */
  float integral[3];
  float two_kp;
  float two_ki;
} mahony_ahrs_t;

void MahonyAhrs_Init(mahony_ahrs_t *filter, const float accel[3]);
void MahonyAhrs_Update(mahony_ahrs_t *filter, const float gyro_rad_s[3],
                       const float accel_m_s2[3], const float mag[3],
                       uint8_t mag_valid, float dt_s);

#endif
