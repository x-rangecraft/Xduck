#include "mahony_ahrs.h"

#include <stdint.h>

#define STANDARD_GRAVITY_M_S2       9.80665f
#define ACCEL_FULL_WEIGHT_DELTA_G   0.10f
#define ACCEL_ZERO_WEIGHT_DELTA_G   0.30f

static float inverse_sqrt(float value)
{
  union { float f; uint32_t u; } bits;
  float half = 0.5f * value;
  bits.f = value;
  bits.u = 0x5F3759DFU - (bits.u >> 1);
  bits.f *= 1.5f - half * bits.f * bits.f;
  bits.f *= 1.5f - half * bits.f * bits.f;
  return bits.f;
}

static uint8_t normalize3(float v[3])
{
  float norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  float inverse;
  if (norm_sq < 1.0e-12f) return 0U;
  inverse = inverse_sqrt(norm_sq);
  v[0] *= inverse; v[1] *= inverse; v[2] *= inverse;
  return 1U;
}

static void normalize_quaternion(float q[4])
{
  float norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  float inverse;
  if (norm_sq < 1.0e-12f) {
    q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;
    return;
  }
  inverse = inverse_sqrt(norm_sq);
  q[0] *= inverse; q[1] *= inverse; q[2] *= inverse; q[3] *= inverse;
}

/* Linear acceleration is indistinguishable from gravity to a 6DoF AHRS. Keep
 * full correction near 1 g, then fade it out instead of forcing a false tilt
 * during a foot impact, rapid start, or fall. */
static float accel_correction_weight(float norm_m_s2)
{
  float delta_g = norm_m_s2 / STANDARD_GRAVITY_M_S2 - 1.0f;
  if (delta_g < 0.0f) delta_g = -delta_g;
  if (delta_g <= ACCEL_FULL_WEIGHT_DELTA_G) return 1.0f;
  if (delta_g >= ACCEL_ZERO_WEIGHT_DELTA_G) return 0.0f;
  return (ACCEL_ZERO_WEIGHT_DELTA_G - delta_g) /
         (ACCEL_ZERO_WEIGHT_DELTA_G - ACCEL_FULL_WEIGHT_DELTA_G);
}

void MahonyAhrs_Init(mahony_ahrs_t *filter, const float accel[3])
{
  float a[3] = {accel[0], accel[1], accel[2]};
  /* This implementation forms full-length reference vectors (rather than the
   * half-vectors in Mahony's original code), hence the halved gains here. */
  filter->two_kp = 0.5f;
  filter->two_ki = 0.01f;
  filter->integral[0] = 0.0f; filter->integral[1] = 0.0f; filter->integral[2] = 0.0f;
  if (normalize3(a) == 0U) {
    filter->q[0] = 1.0f; filter->q[1] = 0.0f; filter->q[2] = 0.0f; filter->q[3] = 0.0f;
    return;
  }
  /* Shortest rotation from measured +gravity in body coordinates to world +Z. */
  if (a[2] < -0.9999f) {
    filter->q[0] = 0.0f; filter->q[1] = 1.0f; filter->q[2] = 0.0f; filter->q[3] = 0.0f;
  } else {
    filter->q[0] = 1.0f + a[2];
    filter->q[1] = a[1];
    filter->q[2] = -a[0];
    filter->q[3] = 0.0f;
    normalize_quaternion(filter->q);
  }
}

void MahonyAhrs_Update(mahony_ahrs_t *f, const float gyro_rad_s[3],
                       const float accel_m_s2[3], const float mag_input[3],
                       uint8_t mag_valid, float dt_s)
{
  float a[3] = {accel_m_s2[0], accel_m_s2[1], accel_m_s2[2]};
  float m[3] = {0.0f, 0.0f, 0.0f};
  float gx = gyro_rad_s[0], gy = gyro_rad_s[1], gz = gyro_rad_s[2];
  float q0 = f->q[0], q1 = f->q[1], q2 = f->q[2], q3 = f->q[3];
  float ex = 0.0f, ey = 0.0f, ez = 0.0f;
  float vx, vy, vz;
  float accel_norm_sq = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  float accel_weight = 0.0f;

  if (accel_norm_sq >= 1.0e-12f) {
    accel_weight = accel_correction_weight(
        accel_norm_sq * inverse_sqrt(accel_norm_sq));
  }

  if (normalize3(a) != 0U) {
    vx = 2.0f * (q1 * q3 - q0 * q2);
    vy = 2.0f * (q0 * q1 + q2 * q3);
    vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
    ex = (a[1] * vz - a[2] * vy) * accel_weight;
    ey = (a[2] * vx - a[0] * vz) * accel_weight;
    ez = (a[0] * vy - a[1] * vx) * accel_weight;

    if (mag_valid != 0U && mag_input != 0) {
      float hx, hy, bx, bz, wx, wy, wz;
      m[0] = mag_input[0]; m[1] = mag_input[1]; m[2] = mag_input[2];
      if (normalize3(m) != 0U) {
        hx = 2.0f * (m[0] * (0.5f - q2*q2 - q3*q3) + m[1] * (q1*q2 - q0*q3) + m[2] * (q1*q3 + q0*q2));
        hy = 2.0f * (m[0] * (q1*q2 + q0*q3) + m[1] * (0.5f - q1*q1 - q3*q3) + m[2] * (q2*q3 - q0*q1));
        bx = (hx * hx + hy * hy) * inverse_sqrt(hx * hx + hy * hy);
        bz = 2.0f * (m[0] * (q1*q3 - q0*q2) + m[1] * (q2*q3 + q0*q1) + m[2] * (0.5f - q1*q1 - q2*q2));
        wx = 2.0f * (bx * (0.5f - q2*q2 - q3*q3) + bz * (q1*q3 - q0*q2));
        wy = 2.0f * (bx * (q1*q2 - q0*q3) + bz * (q0*q1 + q2*q3));
        wz = 2.0f * (bx * (q0*q2 + q1*q3) + bz * (0.5f - q1*q1 - q2*q2));
        ex += m[1] * wz - m[2] * wy;
        ey += m[2] * wx - m[0] * wz;
        ez += m[0] * wy - m[1] * wx;
      }
    }

    f->integral[0] += f->two_ki * ex * dt_s;
    f->integral[1] += f->two_ki * ey * dt_s;
    f->integral[2] += f->two_ki * ez * dt_s;
    gx += f->two_kp * ex + f->integral[0];
    gy += f->two_kp * ey + f->integral[1];
    gz += f->two_kp * ez + f->integral[2];
  }

  gx *= 0.5f * dt_s; gy *= 0.5f * dt_s; gz *= 0.5f * dt_s;
  f->q[0] += -q1 * gx - q2 * gy - q3 * gz;
  f->q[1] +=  q0 * gx + q2 * gz - q3 * gy;
  f->q[2] +=  q0 * gy - q1 * gz + q3 * gx;
  f->q[3] +=  q0 * gz + q1 * gy - q2 * gx;
  normalize_quaternion(f->q);
}
