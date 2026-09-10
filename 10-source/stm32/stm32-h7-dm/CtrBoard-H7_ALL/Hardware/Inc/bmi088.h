#ifndef __BMI088_H
#define __BMI088_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct {
  float gyro_rad_s[3];
  float accel_m_s2[3];
  float temperature_c;
} bmi088_sample_t;

typedef enum {
  BMI088_OK = 0,
  BMI088_SPI_ERROR = 1,
  BMI088_ACCEL_NOT_FOUND = 2,
  BMI088_GYRO_NOT_FOUND = 3,
  BMI088_CONFIG_ERROR = 4
} bmi088_status_t;

bmi088_status_t BMI088_Init(void);
bmi088_status_t BMI088_Read(bmi088_sample_t *sample);

#ifdef __cplusplus
}
#endif
#endif
