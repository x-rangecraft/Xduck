#ifndef __MOTOR_APP_H
#define __MOTOR_APP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include "h7spi_protocol.h"
#include "dmusb_protocol.h"

#define MOTOR_APP_FAULT_NONE             (0UL)
#define MOTOR_APP_FAULT_ENABLE_FAILED    (1UL << 0)
#define MOTOR_APP_FAULT_FEEDBACK_STALE   (1UL << 1)
#define MOTOR_APP_FAULT_MOS_OVER_TEMP    (1UL << 2)
#define MOTOR_APP_FAULT_ROTOR_OVER_TEMP  (1UL << 3)
#define MOTOR_APP_FAULT_MOTOR_HW_FAULT   (1UL << 4)
#define MOTOR_APP_FAULT_CAN_TX_DROP      (1UL << 5)
#define MOTOR_APP_FAULT_TIMEOUT_CONFIG   (1UL << 6)
#define MOTOR_APP_FAULT_HOST_COMMAND_STALE (1UL << 7)
#define MOTOR_APP_FAULT_IMU_INVALID        (1UL << 8)

#define MOTOR_APP_MODE_ADMIN       (0x10U)
#define MOTOR_APP_MODE_CONTROL_MIT (0x21U)

typedef struct {
  unsigned long fault_flags;
  unsigned char mode;
  unsigned char enabled;
  unsigned char fault;
  unsigned char busy;
  unsigned char failed_motor_id;
  unsigned char failed_route_index;
  unsigned char motor_count;
  unsigned char lifecycle_state;
  unsigned char failed_phase;
  unsigned long enabled_motor_mask;
  unsigned long requested_mask;
  unsigned long succeeded_mask;
  unsigned char requested_count;
  unsigned char success_count;
} MotorApp_Status_t;

void MotorApp_ConfigDefault(void);
unsigned char MotorApp_PositionLimitsReady(void);
unsigned char MotorApp_SetPositionLimits(const dmusb_position_limit_t *limits, unsigned char count);
unsigned char MotorApp_ConfigureStartupRegisters(void);
unsigned char MotorApp_RequestEnableDefault(void);
unsigned char MotorApp_RequestDisableDefault(void);
unsigned char MotorApp_ApplySpiAdminOp(const h7spi_admin_op_frame_t *AdminOp);
unsigned char MotorApp_ReadSpiMotorRegs(const h7spi_motor_reg_request_frame_t *Request,
                                        h7spi_motor_reg_values_frame_t *Response);
unsigned char MotorApp_WriteSpiMotorRegs(const h7spi_motor_reg_write_request_frame_t *Request,
                                         h7spi_motor_reg_values_frame_t *Response);
void MotorApp_EnableDefault(void);
void MotorApp_Tick(void);
unsigned char MotorApp_IsEnabled(void);
unsigned char MotorApp_IsFault(void);
unsigned char MotorApp_GetFailedMotorID(void);
unsigned long MotorApp_GetFaultFlags(void);
unsigned char MotorApp_GetMode(void);
void MotorApp_GetStatus(MotorApp_Status_t *Status);
unsigned char MotorApp_ApplySpiMotorCommand(const h7spi_motor_cmd_frame_t *Command);
unsigned char MotorApp_ApplySpiMotorCommandCompact(const h7spi_motor_cmd_compact_frame_t *Command);
unsigned char MotorApp_SubmitSpiMotorCommandAsync(const h7spi_motor_cmd_frame_t *Command);
void MotorApp_ProcessPendingSpiMotorCommand(void);
void MotorApp_FillSpiState(h7spi_motor_state_frame_t *State, unsigned short AckCommandSeq);
void MotorApp_FillSpiStateCompact(h7spi_motor_state_compact_frame_t *State, unsigned short AckCommandSeq);
unsigned char MotorApp_GetMotorCount(void);
unsigned char MotorApp_GetMotorIdByIndex(unsigned char Index);
unsigned short MotorApp_GetLastCommandSeq(void);

#ifdef __cplusplus
}
#endif

#endif

/**********************************END OF FILE***********************************/
