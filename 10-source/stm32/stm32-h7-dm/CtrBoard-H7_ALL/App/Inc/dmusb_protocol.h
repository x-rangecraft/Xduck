#ifndef __DMUSB_PROTOCOL_H
#define __DMUSB_PROTOCOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Wire format reused from STM32_from_166. Mode switching is deliberately
 * absent: this gateway has one control mode, MIT. */
#define DMUSB_MAGIC              0x4D47U
#define DMUSB_VERSION            4U
#define DMUSB_HEADER_SIZE        18U
#define DMUSB_MAX_PAYLOAD        1536U
#define DMUSB_MAX_MOTORS         24U

#define DMUSB_MSG_COMMAND_FRAME  1U
#define DMUSB_MSG_STATE_FRAME    2U
#define DMUSB_MSG_ADMIN_OP       12U
#define DMUSB_MSG_ADMIN_RESULT   13U
#define DMUSB_MODE_MIT           1U
#define DMUSB_ADMIN_ENABLE       1U
#define DMUSB_ADMIN_DISABLE      2U
#define DMUSB_ADMIN_MARK_ZERO    3U
#define DMUSB_ADMIN_SET_LIMITS   4U
#define DMUSB_MSG_SET_LIMITS     14U
/* Backward-compatible capabilities in the state prefix's reserved uint16. */
#define DMUSB_CAP_SPARSE_COMMAND  0x0001U
#define DMUSB_CAP_ENABLE_TARGETS  0x0002U
#define DMUSB_CAP_CALIBRATION     0x0004U
/* Dynamic state bit; clears on MCU reset until the host installs limits. */
#define DMUSB_STATE_LIMITS_READY  0x0008U
/* imu.reserved[0:2] carries the last protection fault's motor ID and route.
 * It survives disable/recovery; ID 0 / route 0xff means no specific motor. */
#define DMUSB_CAP_FAULT_DIAGNOSTICS 0x0010U

/* Exact-length payload: request_seq:u16, count:u8, reserved:u8, then
 * count records {motor_id:u8, reserved[3], min_mrad:i32, max_mrad:i32}.
 * All configured motors are required; bounds are installed atomically in RAM. */
typedef struct {
  uint8_t motor_id;
  uint8_t reserved[3];
  int32_t min_mrad;
  int32_t max_mrad;
} dmusb_position_limit_t;

typedef struct {
  int32_t p_mrad;
  int32_t v_mrad_s;
  int32_t torque_mnm;
  uint16_t kp_centi;
  uint16_t kd_milli;
  uint16_t flags;
} dmusb_motor_command_t;

typedef struct {
  uint16_t command_seq;
  uint8_t mode;
  uint8_t motor_count;
  uint32_t host_tick_us;
  dmusb_motor_command_t motors[DMUSB_MAX_MOTORS];
} dmusb_command_frame_t;

typedef struct {
  int32_t p_mrad;
  int32_t v_mrad_s;
  int32_t torque_mnm;
  uint32_t rx_ts_ms;
  uint8_t flags;
  uint8_t rotor_temperature_c;
  uint8_t reserved[2];
} dmusb_motor_observation_t;

typedef struct {
  float base_ang_vel[3];       /* body X/Y/Z, rad/s */
  float projected_gravity[3];  /* world [0,0,-1] expressed in body frame */
  float quaternion[4];         /* body to world, scalar-first [w,x,y,z] */
  uint32_t sample_sequence;
  uint8_t flags;
  uint8_t reserved[3];
} dmusb_imu_observation_t;

typedef struct {
  uint16_t state_seq;
  uint16_t ack_command_seq;
  uint32_t stm32_tick_ms;
  uint32_t fault_flags;
  uint8_t mode;
  uint8_t motor_count;
  uint16_t reserved;
  dmusb_imu_observation_t imu;
  dmusb_motor_observation_t motors[DMUSB_MAX_MOTORS];
} dmusb_state_frame_t;

typedef struct {
  uint16_t request_seq;
  uint8_t op;
  uint8_t motor_count;
  uint8_t motor_ids[DMUSB_MAX_MOTORS];
} dmusb_admin_op_frame_t;

typedef struct {
  uint16_t response_seq;
  uint8_t op;
  uint8_t result;
  uint32_t stm32_tick_ms;
  uint32_t fault_flags;
  uint32_t requested_mask;
  uint32_t succeeded_mask;
  uint8_t busy;
  uint8_t requested_count;
  uint8_t success_count;
  uint8_t failed_motor_id;
  uint8_t failed_route_index;
  uint8_t reserved[3];
} dmusb_admin_result_frame_t;

typedef struct {
  uint8_t msg_type;
  uint16_t seq;
  uint16_t payload_len;
  uint32_t tick_us;
  uint32_t flags;
} dmusb_header_t;

uint16_t DMUSB_Crc16(const uint8_t *data, uint32_t len);
uint8_t DMUSB_Validate(const uint8_t *frame, uint16_t frame_len,
                       dmusb_header_t *header, const uint8_t **payload);
uint8_t DMUSB_Pack(uint8_t msg_type, uint16_t seq, uint32_t tick_us,
                   uint32_t flags, const void *payload, uint16_t payload_len,
                   uint8_t *out, uint16_t capacity, uint16_t *out_len);

#ifdef __cplusplus
}
#endif
#endif
