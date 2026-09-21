#include "imu_task.h"

#include "FreeRTOS.h"
#include "bmi088.h"
#include "cmsis_os.h"
#include "mahony_ahrs.h"
#include "main.h"
#include "task.h"

#include <string.h>

#define IMU_PERIOD_MS          1U
#define IMU_MAX_INTEGRATION_GAP_MS 50U
#define GYRO_CALIBRATION_SAMPLES 1000U
#define IMU_READY_SAMPLES        500U /* 25 consecutive 50 Hz observation periods */
#define IMU_LPF_TIME_CONSTANT_S  0.005305165f /* 30 Hz one-pole cutoff */
#define STANDARD_GRAVITY_M_S2    9.80665f
#define STATIONARY_GYRO_RAD_S    0.05f
#define STATIONARY_ACCEL_MIN_G   0.85f
#define STATIONARY_ACCEL_MAX_G   1.15f

/* BMI088 sensor axes -> robot body forward/left/up.  The production mounting
 * was measured on the assembled robot with a six-face gravity test:
 * body X = +sensor Z, body Y = -sensor Y, body Z = +sensor X. */
#define BODY_X_SENSOR_AXIS 2U
#define BODY_Y_SENSOR_AXIS 1U
#define BODY_Z_SENSOR_AXIS 0U
#define BODY_X_SENSOR_SIGN 1.0f
#define BODY_Y_SENSOR_SIGN -1.0f
#define BODY_Z_SENSOR_SIGN 1.0f

static imu_snapshot_t g_snapshot = {
  {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, 0U, 0U, 0U
};
static float g_magnetometer[3];
static uint8_t g_magnetometer_valid;
static uint32_t g_magnetometer_tick_ms;
static osThreadId g_imu_task;

/* BMI088 sensor frame -> robot body frame. Keep gyro and acceleration on the
 * same proper rotation so Mahony integration and projected gravity agree. */
static void sensor_to_body(const float sensor[3], float body[3])
{
  body[0] = BODY_X_SENSOR_SIGN * sensor[BODY_X_SENSOR_AXIS];
  body[1] = BODY_Y_SENSOR_SIGN * sensor[BODY_Y_SENSOR_AXIS];
  body[2] = BODY_Z_SENSOR_SIGN * sensor[BODY_Z_SENSOR_AXIS];
}

static uint8_t calibration_sample_is_stationary(const float gyro[3],
                                                const float accel[3])
{
  const float gyro_limit_sq = STATIONARY_GYRO_RAD_S * STATIONARY_GYRO_RAD_S;
  const float gravity_sq = STANDARD_GRAVITY_M_S2 * STANDARD_GRAVITY_M_S2;
  const float accel_min_sq = STATIONARY_ACCEL_MIN_G * STATIONARY_ACCEL_MIN_G * gravity_sq;
  const float accel_max_sq = STATIONARY_ACCEL_MAX_G * STATIONARY_ACCEL_MAX_G * gravity_sq;
  float gyro_sq = gyro[0]*gyro[0] + gyro[1]*gyro[1] + gyro[2]*gyro[2];
  float accel_sq = accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2];
  return (uint8_t)(gyro_sq <= gyro_limit_sq &&
                   accel_sq >= accel_min_sq && accel_sq <= accel_max_sq);
}

static void low_pass3(float state[3], const float input[3], float dt_s)
{
  float alpha = dt_s / (IMU_LPF_TIME_CONSTANT_S + dt_s);
  uint8_t axis;
  for (axis = 0U; axis < 3U; axis++) {
    state[axis] += alpha * (input[axis] - state[axis]);
  }
}

static void invalidate_snapshot(void)
{
  taskENTER_CRITICAL();
  g_snapshot.flags = 0U;
  taskEXIT_CRITICAL();
}

static void publish(const float gyro[3], const mahony_ahrs_t *filter, uint8_t flags)
{
  imu_snapshot_t next;
  const float q0 = filter->q[0], q1 = filter->q[1], q2 = filter->q[2], q3 = filter->q[3];
  memcpy(next.base_ang_vel, gyro, sizeof(next.base_ang_vel));
  /* q is body->world. Rotate world gravity [0,0,-1] into the body frame. */
  next.projected_gravity[0] = -2.0f * (q1 * q3 - q0 * q2);
  next.projected_gravity[1] = -2.0f * (q0 * q1 + q2 * q3);
  next.projected_gravity[2] = -(q0*q0 - q1*q1 - q2*q2 + q3*q3);
  memcpy(next.quaternion, filter->q, sizeof(next.quaternion));
  next.sample_tick_ms = HAL_GetTick();
  next.flags = flags;
  taskENTER_CRITICAL();
  next.sample_sequence = g_snapshot.sample_sequence + 1U;
  g_snapshot = next;
  taskEXIT_CRITICAL();
}

void IMU_SetMagnetometer(const float mag[3], uint8_t valid)
{
  taskENTER_CRITICAL();
  if (mag != 0 && valid != 0U) memcpy(g_magnetometer, mag, sizeof(g_magnetometer));
  g_magnetometer_valid = (uint8_t)(mag != 0 && valid != 0U);
  g_magnetometer_tick_ms = HAL_GetTick();
  taskEXIT_CRITICAL();
}

uint8_t IMU_GetSnapshot(imu_snapshot_t *snapshot)
{
  if (snapshot == 0) return 0U;
  taskENTER_CRITICAL();
  *snapshot = g_snapshot;
  taskEXIT_CRITICAL();
  return (uint8_t)((snapshot->flags & IMU_FLAG_CALIBRATED) != 0U);
}

static void StartIMUTask(void const *argument)
{
  bmi088_sample_t sample;
  mahony_ahrs_t filter;
  float gyro_bias_sum[3] = {0.0f, 0.0f, 0.0f};
  float gyro_bias[3] = {0.0f, 0.0f, 0.0f};
  float filtered_gyro[3] = {0.0f, 0.0f, 0.0f};
  float filtered_accel[3] = {0.0f, 0.0f, 0.0f};
  uint32_t calibration_count = 0U;
  uint32_t ready_sample_count = 0U;
  TickType_t last_wake;
  TickType_t last_sample_tick;
  uint8_t filter_initialized = 0U;
  (void)argument;

  while (BMI088_Init() != BMI088_OK) osDelay(100U);
  last_wake = xTaskGetTickCount();
  last_sample_tick = last_wake;
  for (;;) {
    float gyro[3];
    float accel[3];
    float mag[3];
    uint8_t mag_valid;
    uint32_t mag_tick_ms;
    uint8_t axis;
    TickType_t now_tick = xTaskGetTickCount();
    TickType_t elapsed_ticks = now_tick - last_sample_tick;
    float dt_s = (float)elapsed_ticks * (float)portTICK_PERIOD_MS * 0.001f;
    last_sample_tick = now_tick;
    if (elapsed_ticks == 0U) {
      /* A catch-up iteration in the same RTOS tick has no time to integrate. */
      last_wake = now_tick;
      vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(IMU_PERIOD_MS));
      continue;
    }
    /* Integrate real elapsed time after a short scheduling delay. A much longer
     * gap cannot be reconstructed from one new gyro sample: fail safe instead
     * of silently treating the whole gap as a 1 ms update. */
    if (elapsed_ticks > pdMS_TO_TICKS(IMU_MAX_INTEGRATION_GAP_MS)) {
      ready_sample_count = 0U;
      invalidate_snapshot();
      last_wake = now_tick;
      vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(IMU_PERIOD_MS));
      continue;
    }
    if (BMI088_Read(&sample) == BMI088_OK) {
      sensor_to_body(sample.gyro_rad_s, gyro);
      sensor_to_body(sample.accel_m_s2, accel);
      if (calibration_count < GYRO_CALIBRATION_SAMPLES) {
        if (calibration_sample_is_stationary(gyro, accel) != 0U) {
          for (axis = 0U; axis < 3U; axis++) gyro_bias_sum[axis] += gyro[axis];
          calibration_count++;
          if (calibration_count == GYRO_CALIBRATION_SAMPLES) {
            for (axis = 0U; axis < 3U; axis++) {
              gyro_bias[axis] = gyro_bias_sum[axis] /
                                (float)GYRO_CALIBRATION_SAMPLES;
            }
          }
        } else {
          calibration_count = 0U;
          gyro_bias_sum[0] = 0.0f;
          gyro_bias_sum[1] = 0.0f;
          gyro_bias_sum[2] = 0.0f;
        }
      } else {
        for (axis = 0U; axis < 3U; axis++) gyro[axis] -= gyro_bias[axis];
        if (filter_initialized == 0U) {
          memcpy(filtered_gyro, gyro, sizeof(filtered_gyro));
          memcpy(filtered_accel, accel, sizeof(filtered_accel));
          MahonyAhrs_Init(&filter, filtered_accel);
          filter_initialized = 1U;
        } else {
          low_pass3(filtered_gyro, gyro, dt_s);
          low_pass3(filtered_accel, accel, dt_s);
        }
        taskENTER_CRITICAL();
        memcpy(mag, g_magnetometer, sizeof(mag));
        mag_valid = g_magnetometer_valid;
        mag_tick_ms = g_magnetometer_tick_ms;
        taskEXIT_CRITICAL();
        if ((uint32_t)(HAL_GetTick() - mag_tick_ms) > 20U) mag_valid = 0U;
        MahonyAhrs_Update(&filter, filtered_gyro, filtered_accel,
                          mag, mag_valid, dt_s);
        if (ready_sample_count < IMU_READY_SAMPLES) ready_sample_count++;
        publish(filtered_gyro, &filter,
                (uint8_t)(IMU_FLAG_SENSOR_OK |
                (ready_sample_count >= IMU_READY_SAMPLES ? IMU_FLAG_CALIBRATED : 0U) |
                (mag_valid != 0U ? IMU_FLAG_MAG_VALID : 0U)));
      }
    } else {
      ready_sample_count = 0U;
      invalidate_snapshot();
    }
    vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(IMU_PERIOD_MS));
  }
}

void IMU_TaskInit(void)
{
  osThreadDef(imuTask, StartIMUTask, osPriorityHigh, 0, 1024);
  g_imu_task = osThreadCreate(osThread(imuTask), 0);
}
