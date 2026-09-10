#ifndef __H7SPI_PROTOCOL_H
#define __H7SPI_PROTOCOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define H7SPI_TRANSFER_SIZE  512U
#define H7SPI_MAX_MOTORS     24U
#define H7SPI_MAGIC          0xA55AU
#define H7SPI_VERSION        2U

#define H7SPI_MSG_MOTOR_CMD       0x01U
#define H7SPI_MSG_MOTOR_STATE     0x02U
#define H7SPI_MSG_SET_MODE        0x03U
#define H7SPI_MSG_MODE_STATE      0x04U
#define H7SPI_MSG_ADMIN_OP        0x05U
#define H7SPI_MSG_ADMIN_RESULT    0x06U
#define H7SPI_MSG_HEARTBEAT       0x07U
#define H7SPI_MSG_ACK             0x08U
#define H7SPI_MSG_ERROR           0x09U
#define H7SPI_MSG_GET_MOTOR_REG   0x0AU
#define H7SPI_MSG_MOTOR_REG_VALUES 0x0BU
#define H7SPI_MSG_SET_MOTOR_REG   0x0CU
#define H7SPI_MSG_MOTOR_REG_WRITE_RESULT 0x0DU
#define H7SPI_MSG_MOTOR_CMD_COMPACT   0x0EU
#define H7SPI_MSG_MOTOR_STATE_COMPACT 0x0FU

#define H7SPI_ADMIN_OP_ENABLE_ALL   0x01U
#define H7SPI_ADMIN_OP_DISABLE_ALL  0x02U
#define H7SPI_ADMIN_OP_MARK_ZERO    0x03U
#define H7SPI_ADMIN_OP_CLEAR_ERROR  0x04U
#define H7SPI_ADMIN_OP_CONFIG_TIMEOUT 0x05U

#define H7SPI_MODE_DISABLED       0x00U
#define H7SPI_MODE_ADMIN          0x10U
#define H7SPI_MODE_CONTROL_MIT    0x21U
#define H7SPI_MODE_CONTROL_SPEED  0x22U
#define H7SPI_MODE_CONTROL_POS    0x23U

#define H7SPI_RESULT_OK              0x00U
#define H7SPI_RESULT_BAD_MAGIC       0x01U
#define H7SPI_RESULT_BAD_VERSION     0x02U
#define H7SPI_RESULT_BAD_CRC         0x03U
#define H7SPI_RESULT_BAD_LENGTH      0x04U
#define H7SPI_RESULT_BAD_TYPE        0x05U
#define H7SPI_RESULT_BAD_MODE        0x06U
#define H7SPI_RESULT_BUSY            0x07U
#define H7SPI_RESULT_FAULT           0x08U
#define H7SPI_RESULT_DUPLICATE_MOTOR_ID 0x09U
#define H7SPI_RESULT_UNKNOWN_MOTOR_ID   0x0AU
#define H7SPI_RESULT_BAD_MOTOR_COUNT    0x0BU
#define H7SPI_RESULT_LIMIT_EXCEEDED     0x0CU
#define H7SPI_RESULT_COMMAND_POSITION_LIMIT 0x0DU
#define H7SPI_RESULT_COMMAND_TORQUE_LIMIT   0x0EU

#define H7SPI_GATEWAY_LIFECYCLE_STOPPED         0U
#define H7SPI_GATEWAY_LIFECYCLE_STARTING        1U
#define H7SPI_GATEWAY_LIFECYCLE_RUNNING_ADMIN   2U
#define H7SPI_GATEWAY_LIFECYCLE_RUNNING_CONTROL 3U
#define H7SPI_GATEWAY_LIFECYCLE_STOPPING        4U
#define H7SPI_GATEWAY_LIFECYCLE_FAULT           5U

#define H7SPI_GATEWAY_PHASE_NONE           0U
#define H7SPI_GATEWAY_PHASE_CAN_STOP       1U
#define H7SPI_GATEWAY_PHASE_CAN_START      2U
#define H7SPI_GATEWAY_PHASE_RX_DRAIN       3U
#define H7SPI_GATEWAY_PHASE_TIMEOUT_READ   4U
#define H7SPI_GATEWAY_PHASE_TIMEOUT_WRITE  5U
#define H7SPI_GATEWAY_PHASE_TIMEOUT_SAVE   6U
#define H7SPI_GATEWAY_PHASE_TIMEOUT_VERIFY 7U
#define H7SPI_GATEWAY_PHASE_MOTOR_OFF      8U
#define H7SPI_GATEWAY_PHASE_MOTOR_ON       9U
#define H7SPI_GATEWAY_PHASE_MODE_ADMIN     10U
#define H7SPI_GATEWAY_PHASE_MODE_CONTROL   11U

#define H7SPI_MOTOR_FLAG_CONFIGURED  (1U << 0)
#define H7SPI_MOTOR_FLAG_ONLINE      (1U << 1)
#define H7SPI_MOTOR_FLAG_ENABLED     (1U << 2)
#define H7SPI_MOTOR_FLAG_FAULT       (1U << 3)
#define H7SPI_MOTOR_FLAG_CMD_VALID   (1U << 4)
#define H7SPI_MOTOR_FLAG_CMD_STALE   (1U << 5)
#define H7SPI_MOTOR_FLAG_UPDATED_BY_LAST_FRAME (1U << 6)

#pragma pack(push, 1)
typedef struct {
  uint16_t magic;
  uint8_t version;
  uint8_t msg_type;
  uint16_t seq;
  uint16_t payload_len;
  uint32_t tick_ms;
  uint16_t flags;
  uint16_t crc16;
} h7spi_header_t;

typedef struct {
  uint8_t motor_id;
  uint8_t flags;
  uint16_t reserved0;
  int32_t p_mrad;
  int32_t v_mrad_s;
  int32_t torque_mnm;
  uint16_t kp_centi;
  uint16_t kd_milli;
  uint16_t reserved1;
  uint16_t reserved2;
} h7spi_motor_cmd_t;

typedef struct {
  uint16_t command_seq;
  uint8_t mode;
  uint8_t motor_count;
  uint32_t host_tick_ms;
  h7spi_motor_cmd_t motors[H7SPI_MAX_MOTORS];
} h7spi_motor_cmd_frame_t;

typedef struct {
  uint8_t motor_id;
  uint8_t flags;
  int16_t p_10mrad;
  int16_t v_10mrad_s;
  int16_t torque_10mnm;
  uint16_t kp_centi;
  uint16_t kd_milli;
} h7spi_motor_cmd_compact_t;

typedef struct {
  uint16_t command_seq;
  uint8_t mode;
  uint8_t motor_count;
  uint32_t host_tick_ms;
  h7spi_motor_cmd_compact_t motors[H7SPI_MAX_MOTORS];
} h7spi_motor_cmd_compact_frame_t;

typedef struct {
  uint8_t motor_id;
  uint8_t state;
  uint8_t flags;
  uint8_t temperature;
  int32_t p_mrad;
  int32_t v_mrad_s;
  int32_t torque_mnm;
  uint32_t rx_ts_ms;
  uint16_t last_command_seq;
  uint16_t command_age_ms;
} h7spi_motor_state_t;

typedef struct {
  uint16_t state_seq;
  uint16_t ack_command_seq;
  uint32_t stm32_tick_ms;
  uint32_t fault_flags;
  uint8_t mode;
  uint8_t motor_count;
  uint16_t reserved;
  h7spi_motor_state_t motors[H7SPI_MAX_MOTORS];
} h7spi_motor_state_frame_t;

typedef struct {
  uint8_t motor_id;
  uint8_t state;
  uint8_t flags;
  uint8_t temperature;
  int16_t p_10mrad;
  int16_t v_10mrad_s;
  int16_t torque_10mnm;
  uint16_t rx_ts_ms_lsb;
  uint16_t last_command_seq;
  uint16_t command_age_ms;
} h7spi_motor_state_compact_t;

typedef struct {
  uint16_t state_seq;
  uint16_t ack_command_seq;
  uint32_t stm32_tick_ms;
  uint32_t fault_flags;
  uint8_t mode;
  uint8_t motor_count;
  uint16_t reserved;
  h7spi_motor_state_compact_t motors[H7SPI_MAX_MOTORS];
} h7spi_motor_state_compact_frame_t;

typedef struct {
  uint16_t request_seq;
  uint8_t op;
  uint8_t motor_count;
  uint8_t motor_codes[H7SPI_MAX_MOTORS];
} h7spi_admin_op_frame_t;

typedef struct {
  uint16_t response_seq;
  uint8_t op;
  uint8_t result;
  uint32_t stm32_tick_ms;
  uint32_t fault_flags;
  uint8_t mode;
  uint8_t enabled;
  uint8_t busy;
  uint8_t motor_count;
  uint8_t failed_motor_id;
  uint8_t failed_route_index;
  uint8_t lifecycle_state;
  uint8_t failed_phase;
  uint32_t enabled_motor_mask;
  uint32_t requested_mask;
  uint32_t succeeded_mask;
  uint8_t requested_count;
  uint8_t success_count;
  uint16_t reserved;
} h7spi_admin_result_frame_t;

typedef struct {
  uint16_t request_seq;
  uint8_t reg_addr;
  uint8_t motor_count;
  uint8_t motor_codes[H7SPI_MAX_MOTORS];
} h7spi_motor_reg_request_frame_t;

typedef struct {
  uint16_t request_seq;
  uint8_t reg_addr;
  uint8_t save_after_write;
  uint8_t motor_count;
  uint8_t reserved[3];
  uint8_t motor_codes[H7SPI_MAX_MOTORS];
  int32_t values[H7SPI_MAX_MOTORS];
} h7spi_motor_reg_write_request_frame_t;

typedef struct {
  uint16_t response_seq;
  uint8_t reg_addr;
  uint8_t result;
  uint32_t stm32_tick_ms;
  uint32_t fault_flags;
  uint8_t motor_count;
  uint8_t failed_motor_id;
  uint8_t failed_route_index;
  uint8_t reserved;
  uint8_t motor_codes[H7SPI_MAX_MOTORS];
  int32_t values[H7SPI_MAX_MOTORS];
} h7spi_motor_reg_values_frame_t;

typedef struct {
  uint16_t ack_seq;
  uint8_t result;
  uint8_t msg_type;
  uint32_t detail;
} h7spi_ack_frame_t;

typedef struct {
  uint16_t request_seq;
  uint8_t result;
  uint8_t msg_type;
  uint32_t detail;
} h7spi_error_frame_t;
#pragma pack(pop)

uint16_t H7SPI_Crc16Ccitt(const uint8_t *data, uint16_t len);
uint16_t H7SPI_FrameCrc(const uint8_t *frame);
uint8_t H7SPI_CheckFrame(const uint8_t *frame);
void H7SPI_BuildFrame(uint8_t *frame, uint8_t msg_type, uint16_t seq,
                      uint32_t tick_ms, const void *payload, uint16_t payload_len);

#ifdef __cplusplus
}
#endif

#endif
