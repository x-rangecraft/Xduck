#include "motor_app.h"
#include "BSP_CAN.h"
#include "Motor.h"
#include "imu_task.h"

#include <stddef.h>
#include <string.h>

#define MOTOR_APP_ADMIN_NONE       (0U)
#define MOTOR_APP_ADMIN_ENABLE_ALL (1U)
#define MOTOR_APP_ADMIN_CLEAR_ERROR (2U)

#define MOTOR_APP_PHASE_IDLE        (0U)
#define MOTOR_APP_PHASE_ROUTE_SEND  (1U)
#define MOTOR_APP_PHASE_ROUTE_WAIT  (2U)
#define MOTOR_APP_PHASE_SETTLE_WAIT (3U)
#define MOTOR_APP_PHASE_FAST_START  (4U)

#define MOTOR_APP_MOTOR_WAIT_MS     (200U)
#define MOTOR_APP_CLEAR_WAIT_MS     (80U)
#define MOTOR_APP_ROUTE_SETTLE_MS   (50U)
#define MOTOR_APP_MAX_RETRIES       (5U)
#define MOTOR_APP_DM_TIMEOUT_MS     (8000U)
#define MOTOR_APP_FEEDBACK_TIMEOUT_MS (500U)
#define MOTOR_APP_HOST_COMMAND_TIMEOUT_MS (100U)
#define MOTOR_APP_ADMIN_POLL_INTERVAL_MS (5U)
#define MOTOR_APP_MOS_OVER_TEMP_C     (75U)
#define MOTOR_APP_ROTOR_OVER_TEMP_C   (80U)
#define MOTOR_APP_REG_OT_VALUE        (0x02U)
#define MOTOR_APP_REG_OC_VALUE        (0x03U)
#define MOTOR_APP_REG_OV_VALUE        (0x1DU)
#define MOTOR_APP_REG_CAN_BR          (0x23U)
#define MOTOR_APP_CAN_BR_FD_BRS       (8U)
#define MOTOR_APP_STARTUP_ITEM_OT     (0U)
#define MOTOR_APP_STARTUP_ITEM_OC     (1U)
#define MOTOR_APP_STARTUP_ITEM_OV     (2U)
#define MOTOR_APP_STARTUP_ITEM_CAN_BR (3U)
#define MOTOR_APP_STARTUP_ITEM_TIMEOUT (4U)
#define MOTOR_APP_PROTECT_OT_C       (100.0f)
#define MOTOR_APP_PROTECT_OC_10422   (0.8f)
#define MOTOR_APP_PROTECT_OV_10422   (65.0f)
#define MOTOR_APP_PROTECT_OC_10010   (0.8f)
#define MOTOR_APP_PROTECT_OV_10010   (52.0f)
#define MOTOR_APP_POSITION_RAD       (3.0f)
#define MOTOR_APP_SPEED_10422_RAD_S  (12.566f)
#define MOTOR_APP_TORQUE_10422_NM    (400.0f)
#define MOTOR_APP_SPEED_10010_RAD_S  (15.708f)
#define MOTOR_APP_TORQUE_10010_NM    (150.0f)

typedef struct {
  unsigned char port;
  unsigned short can_id;
  float p_min;
  float p_max;
  float v_min;
  float v_max;
  float t_min;
  float t_max;
  float kp_min;
  float kp_max;
  float kd_min;
  float kd_max;
  float protect_ot;
  float protect_oc;
  float protect_ov;
} MotorApp_DefaultMotorConfig_t;

typedef struct {
  float position;
  float speed;
  float kp;
  float kd;
  float torque;
} MotorApp_ShapedCommand_t;

static const MotorApp_DefaultMotorConfig_t kMotorAppDefaultConfig[] = {
  /* Fixed USB route order: CAN2 x5, CAN3 x4, CAN1 x5. Port 0 is an empty slot. */
  {2U, 0x02U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -10.0f, 10.0f, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -10.0f, 10.0f, -28.0f, 28.0f, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -MOTOR_APP_TORQUE_10010_NM, MOTOR_APP_TORQUE_10010_NM, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -MOTOR_APP_TORQUE_10010_NM, MOTOR_APP_TORQUE_10010_NM, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -MOTOR_APP_TORQUE_10010_NM, MOTOR_APP_TORQUE_10010_NM, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {3U, 0x01U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -10.0f, 10.0f, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -MOTOR_APP_TORQUE_10010_NM, MOTOR_APP_TORQUE_10010_NM, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -MOTOR_APP_TORQUE_10010_NM, MOTOR_APP_TORQUE_10010_NM, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -MOTOR_APP_TORQUE_10010_NM, MOTOR_APP_TORQUE_10010_NM, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {1U, 0x0DU, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -10.0f, 10.0f, -28.0f, 28.0f, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10422, MOTOR_APP_PROTECT_OV_10422},
  {1U, 0x0EU, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -10.0f, 10.0f, -28.0f, 28.0f, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10422, MOTOR_APP_PROTECT_OV_10422},
  {1U, 0x0FU, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -10.0f, 10.0f, -28.0f, 28.0f, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10422, MOTOR_APP_PROTECT_OV_10422},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -MOTOR_APP_TORQUE_10010_NM, MOTOR_APP_TORQUE_10010_NM, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
  {0U, 0x00U, -MOTOR_APP_POSITION_RAD, MOTOR_APP_POSITION_RAD, -MOTOR_APP_SPEED_10010_RAD_S, MOTOR_APP_SPEED_10010_RAD_S, -MOTOR_APP_TORQUE_10010_NM, MOTOR_APP_TORQUE_10010_NM, 0.00f, 500.00f, 0.00f, 5.00f, MOTOR_APP_PROTECT_OT_C, MOTOR_APP_PROTECT_OC_10010, MOTOR_APP_PROTECT_OV_10010},
};

/* CAN wire scaling is independent of the joint's permitted command range.
 * Read back from PMAX/VMAX/TMAX (0x15/0x16/0x17) on all six motors, 2026-09-07.
 * Keep the existing +/-3 rad position guard in kMotorAppDefaultConfig. */
typedef struct { float p_max; float v_max; float t_max; } MotorApp_WireRange_t;
static const MotorApp_WireRange_t kMotorAppWireRange[] = {
  {12.5f, 30.0f, 10.0f}, /* ID 2 */
  {0.0f, 0.0f, 0.0f}, /* ID 3 disconnected: keep its slot empty. */
  {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
  {12.5f, 30.0f, 10.0f}, /* ID 1 */
  {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
  {12.5f, 10.0f, 28.0f}, /* ID 13 */
  {12.5f, 10.0f, 28.0f}, /* ID 14 */
  {12.5f, 10.0f, 28.0f}, /* ID 15 */
  {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
};
static_assert(sizeof(kMotorAppWireRange) / sizeof(kMotorAppWireRange[0]) ==
              sizeof(kMotorAppDefaultConfig) / sizeof(kMotorAppDefaultConfig[0]),
              "Wire ranges must preserve every USB route slot");

static const unsigned char kMotorAppCanPort1MitList[] = {
  0x0DU, 0x0EU, 0x0FU,
};

static const unsigned char kMotorAppCanPort2MitList[] = {
  0x02U,
};

static const unsigned char kMotorAppCanPort3MitList[] = {
  0x01U,
};

static float g_positionMin[H7SPI_MAX_MOTORS];
static float g_positionMax[H7SPI_MAX_MOTORS];
static unsigned char g_positionLimitsReady = 0U;

static unsigned char g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
static unsigned char g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
static unsigned char g_motorAppRouteCursor = 0U;
static unsigned char g_motorAppRouteIndex = 0xFFU;
static unsigned char g_motorAppRetry = 0U;
static unsigned char g_motorAppEnabled = 0U;
static unsigned char g_motorAppFault = 0U;
static unsigned char g_motorAppMode = MOTOR_APP_MODE_ADMIN;
static unsigned char g_motorAppFailedMotorID = 0U;
static unsigned char g_motorAppFailedRouteIndex = 0xFFU;
static unsigned char g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_NONE;
static unsigned long g_motorAppFaultFlags = MOTOR_APP_FAULT_NONE;
static unsigned long g_motorAppRequestedMask = 0UL;
static unsigned long g_motorAppSucceededMask = 0UL;
static unsigned char g_motorAppRequestedCount = 0U;
static unsigned char g_motorAppSuccessCount = 0U;
static unsigned short g_motorAppLastCommandSeq = 0U;
static unsigned int g_motorAppLastHostCommandTickMs = 0U;
static h7spi_motor_cmd_frame_t g_motorAppPendingSpiCommand;
static unsigned char g_motorAppPendingSpiCommandValid = 0U;
static unsigned short g_motorAppMotorLastCommandSeq[MOTOR_NUM];
static unsigned int g_motorAppMotorLastCommandTickMs[MOTOR_NUM];
static unsigned char g_motorAppMotorCommandValid[MOTOR_NUM];
static unsigned char g_motorAppUpdatedByLastFrame[MOTOR_NUM];
static TickType_t g_motorAppDeadlineTick = 0U;
static TickType_t g_motorAppRouteStartRxTick = 0U;
static TickType_t g_motorAppProtectionStartTick = 0U;
static TickType_t g_motorAppAdminPollLastTick = 0U;
static unsigned short g_motorAppStateSeq = 0U;
static unsigned char g_motorAppStartupRegistersConfigured = 0U;
static unsigned char g_motorAppAdminPollIndex = 0U;
static volatile unsigned char g_motorAppRegisterBusy = 0U;
static unsigned char g_motorAppAdminRouteIndices[H7SPI_MAX_MOTORS];
static unsigned char g_motorAppCanPort1Cursor = 0U;

/* Optional debugger-triggered, read-only inspection. Request is a configured
 * motor ID; zero does nothing. High byte optionally selects a whitelisted
 * single register. Results: ID, done, valid bits, CAN ID, master ID,
 * control mode, firmware version, bus voltage (IEEE754). No register writes,
 * enable frames or zeroing are issued. Requests while running are rejected. */
volatile uint32_t motor_probe_request;
volatile uint32_t motor_probe_results[8];
/* Last failed enable: ID, raw motor state, feedback age ms, handshake age ms. */
volatile uint32_t motor_enable_last_failure[4];


static void MotorApp_EnterFault(unsigned char RouteIndex, unsigned long FaultFlag);
static void MotorApp_RecordAdminSuccess(unsigned char RouteIndex);
static unsigned char MotorApp_GetDefaultCount(void);
static unsigned char MotorApp_IsHardwareFaultState(unsigned char State);

static unsigned int MotorApp_FloatToU32(float Value)
{
  unsigned int Raw = 0U;
  memcpy(&Raw, &Value, sizeof(Raw));
  return Raw;
}

static float MotorApp_AbsFloat(float Value)
{
  return (Value < 0.0f) ? -Value : Value;
}

static float MotorApp_LimitFloat(float Value, float MinValue, float MaxValue)
{
  if (Value < MinValue) {
    return MinValue;
  }
  if (Value > MaxValue) {
    return MaxValue;
  }
  return Value;
}

static float MotorApp_MaxFloat(float A, float B)
{
  return (A > B) ? A : B;
}

static unsigned char MotorApp_IsCommandWarning(unsigned char Result)
{
  return (Result == H7SPI_RESULT_LIMIT_EXCEEDED ||
          Result == H7SPI_RESULT_COMMAND_POSITION_LIMIT ||
          Result == H7SPI_RESULT_COMMAND_TORQUE_LIMIT) ? 1U : 0U;
}

static void MotorApp_ClearAdminStats(void)
{
  g_motorAppRequestedMask = 0UL;
  g_motorAppSucceededMask = 0UL;
  g_motorAppRequestedCount = 0U;
  g_motorAppSuccessCount = 0U;
  g_motorAppFailedMotorID = 0U;
  g_motorAppFailedRouteIndex = 0xFFU;
  g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_NONE;
}

static void MotorApp_RecordStartupConfigFailure(unsigned char RouteIndex,
                                                unsigned char ItemIndex)
{
  if (RouteIndex < MotorApp_GetDefaultCount()) {
    g_motorAppFailedMotorID = (unsigned char)kMotorAppDefaultConfig[RouteIndex].can_id;
  }
  g_motorAppFailedRouteIndex = ItemIndex;
  g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_TIMEOUT_WRITE;
}

static unsigned char MotorApp_IsTickExpired(TickType_t NowTick, TickType_t DeadlineTick)
{
  return (unsigned char)(((int32_t)(NowTick - DeadlineTick)) >= 0);
}

static unsigned char MotorApp_GetDefaultCount(void)
{
  return (unsigned char)(sizeof(kMotorAppDefaultConfig) / sizeof(kMotorAppDefaultConfig[0]));
}

static Motor_t *MotorApp_GetRouteMotor(unsigned char RouteIndex)
{
  if (RouteIndex >= MotorApp_GetDefaultCount()) {
    return MotorPoint(0U);
  }
  return MotorPoint((unsigned char)kMotorAppDefaultConfig[RouteIndex].can_id);
}

static unsigned char MotorApp_IsActiveMitCode(unsigned char Code)
{
  unsigned char index;

  for (index = 0U; index < (unsigned char)(sizeof(kMotorAppCanPort1MitList) / sizeof(kMotorAppCanPort1MitList[0])); index++) {
    if (kMotorAppCanPort1MitList[index] == Code) {
      return 1U;
    }
  }
  for (index = 0U; index < (unsigned char)(sizeof(kMotorAppCanPort2MitList) / sizeof(kMotorAppCanPort2MitList[0])); index++) {
    if (kMotorAppCanPort2MitList[index] == Code) {
      return 1U;
    }
  }
  for (index = 0U; index < (unsigned char)(sizeof(kMotorAppCanPort3MitList) / sizeof(kMotorAppCanPort3MitList[0])); index++) {
    if (kMotorAppCanPort3MitList[index] == Code) {
      return 1U;
    }
  }
  return 0U;
}

static unsigned char MotorApp_IsActiveMitRoute(unsigned char RouteIndex)
{
  if (RouteIndex >= MotorApp_GetDefaultCount()) {
    return 0U;
  }
  return MotorApp_IsActiveMitCode((unsigned char)kMotorAppDefaultConfig[RouteIndex].can_id);
}

static void MotorApp_ClearCommandTracking(void)
{
  memset(g_motorAppMotorCommandValid, 0, sizeof(g_motorAppMotorCommandValid));
  memset(g_motorAppUpdatedByLastFrame, 0, sizeof(g_motorAppUpdatedByLastFrame));
}

static void MotorApp_PrepareFallbackCommands(void)
{
  unsigned char index;

  MotorApp_ClearCommandTracking();
  g_motorAppCanPort1Cursor = 0U;
  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    Motor_t *motor = MotorApp_GetRouteMotor(index);
    motor->ClearCommand();
    motor->SetRunFlag(0U);
  }
}

static void MotorApp_SendActiveMitOnce(void)
{
  unsigned char sent;

  if (g_motorAppFault != 0U) {
    return;
  }

  Motor_ClearTxFailure();
  /* FDCAN1 is 500 kbit/s and carries three motors. Sending all three commands
   * plus all three replies every 1 ms exceeds the physical bus capacity and
   * starves one feedback stream. Round-robin one motor per task tick gives
   * every motor a 333 Hz command rate while keeping the worst-case classic-CAN
   * load below the link rate. FDCAN2/3 each have one active motor. */
  sent = Motor_SendPortBudgeted(1U, &g_motorAppCanPort1Cursor, 1U);
  sent = (unsigned char)(sent + Motor_SendListOnce(kMotorAppCanPort2MitList,
                                                   (unsigned char)(sizeof(kMotorAppCanPort2MitList) /
                                                                   sizeof(kMotorAppCanPort2MitList[0]))));
  sent = (unsigned char)(sent + Motor_SendListOnce(kMotorAppCanPort3MitList,
                                                   (unsigned char)(sizeof(kMotorAppCanPort3MitList) /
                                                                   sizeof(kMotorAppCanPort3MitList[0]))));
  (void)sent;
}

static void MotorApp_AdminPollFeedback(TickType_t NowTick)
{
  Motor_t *motor;

  if (g_motorAppMode != MOTOR_APP_MODE_ADMIN ||
      g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE ||
      g_motorAppRegisterBusy != 0U ||
      (NowTick - g_motorAppAdminPollLastTick) < pdMS_TO_TICKS(MOTOR_APP_ADMIN_POLL_INTERVAL_MS)) {
    return;
  }

  if (g_motorAppAdminPollIndex >= MotorApp_GetDefaultCount()) {
    g_motorAppAdminPollIndex = 0U;
  }

  motor = MotorApp_GetRouteMotor(g_motorAppAdminPollIndex);
  Motor_ClearTxFailure();
  motor->SetRunFlag(0U);
  motor->SendDisableFrame();
  if (Motor_HasTxFailure() != 0U) {
    g_motorAppFailedMotorID = (unsigned char)kMotorAppDefaultConfig[g_motorAppAdminPollIndex].can_id;
    g_motorAppFailedRouteIndex = g_motorAppAdminPollIndex;
    g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_MOTOR_OFF;
    g_motorAppFaultFlags |= MOTOR_APP_FAULT_CAN_TX_DROP;
  }
  g_motorAppAdminPollIndex++;
  g_motorAppAdminPollLastTick = NowTick;
}

static void MotorApp_ClearRuntimeState(void)
{
  g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
  g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
  g_motorAppRouteCursor = 0U;
  g_motorAppRouteIndex = 0xFFU;
  g_motorAppRetry = 0U;
  g_motorAppEnabled = 0U;
  g_motorAppFault = 0U;
  g_motorAppMode = MOTOR_APP_MODE_ADMIN;
  g_motorAppFailedMotorID = 0U;
  g_motorAppFailedRouteIndex = 0xFFU;
  g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_NONE;
  g_motorAppFaultFlags = MOTOR_APP_FAULT_NONE;
  g_motorAppLastCommandSeq = 0U;
  g_motorAppLastHostCommandTickMs = 0U;
  g_motorAppStateSeq = 0U;
  g_motorAppDeadlineTick = 0U;
  g_motorAppRouteStartRxTick = 0U;
  g_motorAppProtectionStartTick = 0U;
  g_motorAppAdminPollLastTick = 0U;
  g_motorAppAdminPollIndex = 0U;
  g_motorAppRegisterBusy = 0U;
  memset(g_motorAppAdminRouteIndices, 0, sizeof(g_motorAppAdminRouteIndices));
}

static void MotorApp_StartRouteClearError(unsigned char RouteIndex)
{
  Motor_t *motor;

  g_motorAppRouteIndex = RouteIndex;
  g_motorAppPhase = MOTOR_APP_PHASE_ROUTE_WAIT;
  g_motorAppDeadlineTick = xTaskGetTickCount() + pdMS_TO_TICKS(MOTOR_APP_CLEAR_WAIT_MS);

  motor = MotorApp_GetRouteMotor(RouteIndex);
  g_motorAppRouteStartRxTick = motor->GetLastRxTick();
  motor->SetRunFlag(0U);
  motor->ClearCommand();
  Motor_ClearTxFailure();
  motor->SendClearErrorFrame();
  if (Motor_HasTxFailure() != 0U) {
    g_motorAppRouteCursor++;
    g_motorAppRouteIndex = 0xFFU;
    g_motorAppRetry = 0U;
    g_motorAppPhase = MOTOR_APP_PHASE_ROUTE_SEND;
  }
}

static void MotorApp_FinishClearError(void)
{
  g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
  g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
  g_motorAppRouteCursor = 0U;
  g_motorAppRouteIndex = 0xFFU;
  g_motorAppRetry = 0U;
  g_motorAppEnabled = 0U;
  g_motorAppFault = 0U;
  g_motorAppMode = MOTOR_APP_MODE_ADMIN;
  g_motorAppFailedMotorID = 0U;
  g_motorAppFailedRouteIndex = 0xFFU;
  g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_NONE;
  g_motorAppFaultFlags = MOTOR_APP_FAULT_NONE;
  g_motorAppLastCommandSeq = 0U;
  g_motorAppDeadlineTick = 0U;
  g_motorAppRouteStartRxTick = 0U;
  g_motorAppProtectionStartTick = 0U;
  MotorApp_PrepareFallbackCommands();
}

static void MotorApp_FailClearError(unsigned char RouteIndex)
{
  g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
  g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
  g_motorAppRouteCursor = 0U;
  g_motorAppRouteIndex = 0xFFU;
  g_motorAppRetry = 0U;
  g_motorAppEnabled = 0U;
  g_motorAppFault = 0U;
  g_motorAppMode = MOTOR_APP_MODE_ADMIN;
  g_motorAppFailedMotorID = (unsigned char)kMotorAppDefaultConfig[RouteIndex].can_id;
  g_motorAppFailedRouteIndex = RouteIndex;
  g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_TIMEOUT_READ;
  g_motorAppFaultFlags = MOTOR_APP_FAULT_NONE;
  g_motorAppLastCommandSeq = 0U;
  g_motorAppDeadlineTick = 0U;
  g_motorAppRouteStartRxTick = 0U;
  g_motorAppProtectionStartTick = 0U;
  MotorApp_PrepareFallbackCommands();
}

static void MotorApp_StartRouteEnable(unsigned char RouteIndex)
{
  Motor_t *motor;

  g_motorAppRouteIndex = RouteIndex;
  g_motorAppPhase = MOTOR_APP_PHASE_ROUTE_WAIT;
  g_motorAppDeadlineTick = xTaskGetTickCount() + pdMS_TO_TICKS(MOTOR_APP_MOTOR_WAIT_MS);

  motor = MotorApp_GetRouteMotor(RouteIndex);
  g_motorAppRouteStartRxTick = motor->GetLastRxTick();
  motor->SetRunFlag(0U);
  Motor_ClearTxFailure();
  motor->SendClearErrorFrame();
  osDelay(5U);
  motor->SendEnableFrame();
  if (Motor_HasTxFailure() != 0U) {
    MotorApp_EnterFault(RouteIndex, MOTOR_APP_FAULT_CAN_TX_DROP);
  }
}

static unsigned char MotorApp_ConfigureFdCanBr(const unsigned char *RouteIndices,
                                               unsigned char RouteCount)
{
#if !defined(DM_CAN_FD_MOTOR) || (DM_CAN_FD_MOTOR == 0)
  (void)RouteIndices;
  (void)RouteCount;
  return H7SPI_RESULT_OK;
#else
  unsigned char index;

  if (RouteIndices == 0) {
    return H7SPI_RESULT_BAD_LENGTH;
  }

  for (index = 0U; index < RouteCount; index++) {
    unsigned char route_index = RouteIndices[index];
    const MotorApp_DefaultMotorConfig_t *config;
    Motor_t *motor;
    unsigned int can_br = 0xFFFFFFFFU;

    if (route_index >= MotorApp_GetDefaultCount()) {
      return H7SPI_RESULT_BAD_MOTOR_COUNT;
    }

    config = &kMotorAppDefaultConfig[route_index];
    motor = MotorApp_GetRouteMotor(route_index);

    if (config->port < 1U || config->port > 3U) {
      continue;
    }

    if (motor->ReadRegister(MOTOR_APP_REG_CAN_BR, &can_br) == 0U) {
      MotorApp_EnterFault(route_index, MOTOR_APP_FAULT_TIMEOUT_CONFIG);
      MotorApp_RecordStartupConfigFailure(route_index, MOTOR_APP_STARTUP_ITEM_CAN_BR);
      return H7SPI_RESULT_FAULT;
    }

    if (can_br == MOTOR_APP_CAN_BR_FD_BRS) {
      continue;
    }

    if (motor->WriteRegister(MOTOR_APP_REG_CAN_BR, MOTOR_APP_CAN_BR_FD_BRS, 1U) == 0U) {
      MotorApp_EnterFault(route_index, MOTOR_APP_FAULT_TIMEOUT_CONFIG);
      MotorApp_RecordStartupConfigFailure(route_index, MOTOR_APP_STARTUP_ITEM_CAN_BR);
      return H7SPI_RESULT_FAULT;
    }
  }

  return H7SPI_RESULT_OK;
#endif
}

static unsigned char MotorApp_WriteRegisterIfNeeded(Motor_t *Motor,
                                                    unsigned char RouteIndex,
                                                    unsigned char Reg,
                                                    unsigned int Value,
                                                    unsigned char SaveAfterWrite)
{
  unsigned int current_value = 0U;

  if (Motor->ReadRegister(Reg, &current_value) == 0U) {
    MotorApp_EnterFault(RouteIndex, MOTOR_APP_FAULT_TIMEOUT_CONFIG);
    g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_TIMEOUT_WRITE;
    return H7SPI_RESULT_FAULT;
  }
  if (current_value == Value) {
    return H7SPI_RESULT_OK;
  }
  if (Motor->WriteRegister(Reg, Value, SaveAfterWrite) == 0U) {
    MotorApp_EnterFault(RouteIndex, MOTOR_APP_FAULT_TIMEOUT_CONFIG);
    g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_TIMEOUT_WRITE;
    return H7SPI_RESULT_FAULT;
  }
  return H7SPI_RESULT_OK;
}

static unsigned char MotorApp_ConfigureProtectionRegisters(const unsigned char *RouteIndices,
                                                           unsigned char RouteCount)
{
  unsigned char index;

  if (RouteIndices == 0) {
    return H7SPI_RESULT_BAD_LENGTH;
  }

  for (index = 0U; index < RouteCount; index++) {
    unsigned char route_index = RouteIndices[index];
    const MotorApp_DefaultMotorConfig_t *config;
    Motor_t *motor;
    unsigned char result;

    if (route_index >= MotorApp_GetDefaultCount()) {
      return H7SPI_RESULT_BAD_MOTOR_COUNT;
    }

    config = &kMotorAppDefaultConfig[route_index];
    motor = MotorApp_GetRouteMotor(route_index);
    result = MotorApp_WriteRegisterIfNeeded(motor,
                                            route_index,
                                            MOTOR_APP_REG_OT_VALUE,
                                            MotorApp_FloatToU32(config->protect_ot),
                                            0U);
    if (result != H7SPI_RESULT_OK) {
      MotorApp_RecordStartupConfigFailure(route_index, MOTOR_APP_STARTUP_ITEM_OT);
      return result;
    }
    result = MotorApp_WriteRegisterIfNeeded(motor,
                                            route_index,
                                            MOTOR_APP_REG_OC_VALUE,
                                            MotorApp_FloatToU32(config->protect_oc),
                                            0U);
    if (result != H7SPI_RESULT_OK) {
      MotorApp_RecordStartupConfigFailure(route_index, MOTOR_APP_STARTUP_ITEM_OC);
      return result;
    }
    result = MotorApp_WriteRegisterIfNeeded(motor,
                                            route_index,
                                            MOTOR_APP_REG_OV_VALUE,
                                            MotorApp_FloatToU32(config->protect_ov),
                                            0U);
    if (result != H7SPI_RESULT_OK) {
      MotorApp_RecordStartupConfigFailure(route_index, MOTOR_APP_STARTUP_ITEM_OV);
      return result;
    }
  }

  return H7SPI_RESULT_OK;
}

static unsigned char MotorApp_ConfigureTimeouts(const unsigned char *RouteIndices,
                                                unsigned char RouteCount,
                                                unsigned char RecordSuccess)
{
  unsigned char index;

  if (RouteIndices == 0) {
    return H7SPI_RESULT_BAD_LENGTH;
  }

  for (index = 0U; index < RouteCount; index++) {
    unsigned char route_index = RouteIndices[index];
    Motor_t *motor;

    if (route_index >= MotorApp_GetDefaultCount()) {
      return H7SPI_RESULT_BAD_MOTOR_COUNT;
    }

    motor = MotorApp_GetRouteMotor(route_index);
    if (motor->SetupTimeout(MOTOR_APP_DM_TIMEOUT_MS) == 0U) {
      MotorApp_RecordStartupConfigFailure(route_index, MOTOR_APP_STARTUP_ITEM_TIMEOUT);
      return H7SPI_RESULT_FAULT;
    }
    if (RecordSuccess != 0U) {
      MotorApp_RecordAdminSuccess(route_index);
    }
  }

  return H7SPI_RESULT_OK;
}

static unsigned char MotorApp_StartAllMitFast(void)
{
  unsigned char index;
  unsigned char attempt;
  TickType_t start_rx[H7SPI_MAX_MOTORS];

  /* Clear and enable are separate CAN transactions. Let each motor finish
   * processing clear-error before enable, as the per-route path already does.
   * FAST_START keeps MotorApp_Tick from entering that other enable state machine
   * while these delays yield to the motor task. No host watchdog is armed yet. */
  g_motorAppPhase = MOTOR_APP_PHASE_FAST_START;
  for (index = 0U; index < 4U; index++) motor_enable_last_failure[index] = 0U;
  Motor_ClearTxFailure();
  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    if (MotorApp_IsActiveMitRoute(index) == 0U) continue;
    Motor_t *motor = MotorApp_GetRouteMotor(index);
    start_rx[index] = motor->GetLastRxTick();
    motor->SetRunFlag(0U);
    motor->SendClearErrorFrame();
    if (Motor_HasTxFailure() != 0U) {
      MotorApp_EnterFault(index, MOTOR_APP_FAULT_CAN_TX_DROP);
      return H7SPI_RESULT_FAULT;
    }
  }
  osDelay(5U);
  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    if (MotorApp_IsActiveMitRoute(index) == 0U) continue;
    MotorApp_GetRouteMotor(index)->SendEnableFrame();
    if (Motor_HasTxFailure() != 0U) {
      MotorApp_EnterFault(index, MOTOR_APP_FAULT_CAN_TX_DROP);
      return H7SPI_RESULT_FAULT;
    }
  }
  /* Do not put the first MIT command directly behind the enable frame either. */
  osDelay(5U);

  MotorApp_PrepareFallbackCommands();
  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    if (MotorApp_IsActiveMitRoute(index) != 0U) {
      MotorApp_GetRouteMotor(index)->SetRunFlag(1U);
    }
  }
  g_motorAppEnabled = 1U;
  g_motorAppMode = MOTOR_APP_MODE_CONTROL_MIT;
  g_motorAppLastHostCommandTickMs = HAL_GetTick();
  /* A queued CAN frame is not a confirmed motor enable. Retry only motors
   * missing a fresh enabled reply, within the existing 100 ms host deadline.
   * Fallback MIT has zero gains/velocity/torque throughout this handshake. */
  for (attempt = 0U; attempt < MOTOR_APP_MAX_RETRIES; attempt++) {
    unsigned char missing = 0xFFU;
    MotorApp_SendActiveMitOnce();
    if (Motor_HasTxFailure() != 0U) {
      MotorApp_EnterFault(0xFFU, MOTOR_APP_FAULT_CAN_TX_DROP);
      return H7SPI_RESULT_FAULT;
    }
    osDelay(10U);
    if ((HAL_GetTick() - g_motorAppLastHostCommandTickMs) > MOTOR_APP_HOST_COMMAND_TIMEOUT_MS) {
      MotorApp_EnterFault(0xFFU, MOTOR_APP_FAULT_HOST_COMMAND_STALE);
      return H7SPI_RESULT_FAULT;
    }
    for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
      if (MotorApp_IsActiveMitRoute(index) == 0U) continue;
      Motor_t *motor = MotorApp_GetRouteMotor(index);
      if (motor->GetState() >= Motor_State_OverVoltage) {
        MotorApp_EnterFault(index, MOTOR_APP_FAULT_MOTOR_HW_FAULT);
        return H7SPI_RESULT_FAULT;
      }
      if (motor->IsEnable() != 0U && motor->GetLastRxTick() != start_rx[index]) {
        if ((g_motorAppSucceededMask & (1UL << index)) == 0UL) {
          MotorApp_RecordAdminSuccess(index);
        }
      } else {
        missing = index;
        if ((attempt + 1U) < MOTOR_APP_MAX_RETRIES) {
          motor->SendEnableFrame();
          if (Motor_HasTxFailure() != 0U) {
            MotorApp_EnterFault(index, MOTOR_APP_FAULT_CAN_TX_DROP);
            return H7SPI_RESULT_FAULT;
          }
        }
      }
    }
    if (missing == 0xFFU) break;
    if ((attempt + 1U) == MOTOR_APP_MAX_RETRIES) {
      Motor_t *failed = MotorApp_GetRouteMotor(missing);
      motor_enable_last_failure[0] = failed->GetCANID();
      motor_enable_last_failure[1] = failed->GetState();
      motor_enable_last_failure[2] = xTaskGetTickCount() - failed->GetLastRxTick();
      motor_enable_last_failure[3] = HAL_GetTick() - g_motorAppLastHostCommandTickMs;
      MotorApp_EnterFault(missing, MOTOR_APP_FAULT_ENABLE_FAILED);
      return H7SPI_RESULT_FAULT;
    }
  }
  g_motorAppProtectionStartTick = xTaskGetTickCount();
  g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
  g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
  g_motorAppRouteCursor = MotorApp_GetDefaultCount();
  g_motorAppRouteIndex = 0xFFU;
  g_motorAppRetry = 0U;
  MotorApp_SendActiveMitOnce();
  return H7SPI_RESULT_OK;
}

static void MotorApp_DisableAllRoutes(void)
{
  unsigned char index;

  MotorApp_ClearCommandTracking();
  Motor_ClearTxFailure();
  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    if (MotorApp_IsActiveMitRoute(index) == 0U) continue;
    Motor_t *motor = MotorApp_GetRouteMotor(index);
    motor->SetRunFlag(0U);
    motor->SendDisableFrame();
  }
  if (Motor_HasTxFailure() != 0U) {
    MotorApp_EnterFault(0xFFU, MOTOR_APP_FAULT_CAN_TX_DROP);
  }
}

static unsigned char MotorApp_GetRouteIndexByMotorID(unsigned char MotorID, unsigned char *RouteIndex)
{
  unsigned char index;

  if (MotorID == 0U) return 0U;

  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    if ((unsigned char)kMotorAppDefaultConfig[index].can_id == MotorID) {
      if (RouteIndex != 0) {
        *RouteIndex = index;
      }
      return 1U;
    }
  }
  return 0U;
}

static unsigned char MotorApp_PrepareAdminTargets(const uint8_t *MotorIDs,
                                                  unsigned char MotorCount,
                                                  unsigned char RouteIndices[H7SPI_MAX_MOTORS])
{
  unsigned char index;
  unsigned char route_index;
  unsigned long seen_mask = 0UL;

  MotorApp_ClearAdminStats();
  if (RouteIndices == 0 || MotorCount > H7SPI_MAX_MOTORS) {
    return H7SPI_RESULT_BAD_MOTOR_COUNT;
  }

  if (MotorCount == 0U) {
    for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
      if (MotorApp_IsActiveMitRoute(index) == 0U) continue;
      RouteIndices[g_motorAppRequestedCount++] = index;
      g_motorAppRequestedMask |= (1UL << index);
    }
    return H7SPI_RESULT_OK;
  }

  for (index = 0U; index < MotorCount; index++) {
    unsigned char motor_id = MotorIDs[index];
    if (MotorApp_GetRouteIndexByMotorID(motor_id, &route_index) == 0U ||
        MotorPoint(motor_id)->IsConfigured() == 0U) {
      g_motorAppFailedMotorID = motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
    }
    if ((seen_mask & (1UL << route_index)) != 0UL) {
      g_motorAppFailedMotorID = motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_DUPLICATE_MOTOR_ID;
    }
    seen_mask |= (1UL << route_index);
    RouteIndices[index] = route_index;
    g_motorAppRequestedMask |= (1UL << route_index);
  }
  g_motorAppRequestedCount = MotorCount;
  return H7SPI_RESULT_OK;
}

static void MotorApp_RecordAdminSuccess(unsigned char RouteIndex)
{
  if (RouteIndex < 32U) {
    g_motorAppSucceededMask |= (1UL << RouteIndex);
  }
  g_motorAppSuccessCount++;
}

static void MotorApp_AdminDisableOne(Motor_t *Motor)
{
  if (Motor == 0) {
    return;
  }
  Motor->SetRunFlag(0U);
  Motor->ClearCommand();
  Motor->SendDisableFrame();
}

static void MotorApp_AdminMarkZeroOne(Motor_t *Motor)
{
  if (Motor == 0) {
    return;
  }
  Motor->SaveZero();
}

/* Both the gateway state and fresh motor feedback must say disabled. */
static unsigned char MotorApp_CalibrationAllowed(void)
{
  unsigned char index;
  if (g_motorAppEnabled != 0U || g_motorAppMode != MOTOR_APP_MODE_ADMIN ||
      g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE || g_motorAppRegisterBusy != 0U)
    return 0U;
  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    if (MotorApp_IsActiveMitRoute(index) == 0U) continue;
    Motor_t *motor = MotorApp_GetRouteMotor(index);
    if (motor->IsOnline(MOTOR_APP_FEEDBACK_TIMEOUT_MS) == 0U ||
        motor->GetState() != 0U) return 0U;
  }
  return 1U;
}

unsigned char MotorApp_PositionLimitsReady(void) { return g_positionLimitsReady; }

unsigned char MotorApp_SetPositionLimits(const dmusb_position_limit_t *limits, unsigned char count)
{
  unsigned char routes[H7SPI_MAX_MOTORS], index, configured = 0U;
  unsigned long seen = 0UL;
  if (limits == 0 || count == 0U || count > H7SPI_MAX_MOTORS)
    return H7SPI_RESULT_BAD_MOTOR_COUNT;
  if (MotorApp_CalibrationAllowed() == 0U) return H7SPI_RESULT_BUSY;
  for (index = 0U; index < MotorApp_GetDefaultCount(); index++)
    if (MotorApp_IsActiveMitRoute(index) != 0U) configured++;
  if (count != configured) return H7SPI_RESULT_BAD_MOTOR_COUNT;
  for (index = 0U; index < count; index++) {
    unsigned char route;
    if (MotorApp_GetRouteIndexByMotorID(limits[index].motor_id, &route) == 0U)
      return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
    if ((seen & (1UL << route)) != 0U) return H7SPI_RESULT_DUPLICATE_MOTOR_ID;
    seen |= 1UL << route;
    if (limits[index].min_mrad >= limits[index].max_mrad ||
        limits[index].min_mrad < -kMotorAppWireRange[route].p_max * 1000.0f ||
        limits[index].max_mrad > kMotorAppWireRange[route].p_max * 1000.0f)
      return H7SPI_RESULT_LIMIT_EXCEEDED;
    routes[index] = route;
  }
  taskENTER_CRITICAL();
  if (MotorApp_CalibrationAllowed() == 0U) {
    taskEXIT_CRITICAL();
    return H7SPI_RESULT_BUSY;
  }
  MotorApp_ClearAdminStats();
  g_motorAppRequestedCount = count;
  for (index = 0U; index < count; index++) {
    unsigned char route = routes[index];
    g_positionMin[route] = (float)limits[index].min_mrad / 1000.0f;
    g_positionMax[route] = (float)limits[index].max_mrad / 1000.0f;
    g_motorAppRequestedMask |= 1UL << route;
    MotorApp_RecordAdminSuccess(route);
  }
  g_positionLimitsReady = 1U;
  taskEXIT_CRITICAL();
  return H7SPI_RESULT_OK;
}

static unsigned char MotorApp_ImuHealthy(void)
{
  imu_snapshot_t imu;
  (void)IMU_GetSnapshot(&imu);
  if ((imu.flags & (IMU_FLAG_SENSOR_OK | IMU_FLAG_CALIBRATED)) !=
      (IMU_FLAG_SENSOR_OK | IMU_FLAG_CALIBRATED) ||
      (uint32_t)(HAL_GetTick() - imu.sample_tick_ms) > 100U) return 0U;
  /* Comparisons reject NaN and infinity without depending on libm. */
  float norm = 0.0f;
  for (unsigned char i = 0U; i < 4U; i++) norm += imu.quaternion[i] * imu.quaternion[i];
  if (!(norm > 0.5f && norm < 1.5f)) return 0U;
  for (unsigned char i = 0U; i < 3U; i++) {
    if (!(imu.base_ang_vel[i] >= -100.0f && imu.base_ang_vel[i] <= 100.0f) ||
        !(imu.projected_gravity[i] >= -1.1f && imu.projected_gravity[i] <= 1.1f)) return 0U;
  }
  return 1U;
}

static void MotorApp_EnterFault(unsigned char RouteIndex, unsigned long FaultFlag)
{
  unsigned char first_fault;
  unsigned char motor_id = 0U;

  if (RouteIndex < MotorApp_GetDefaultCount()) {
    motor_id = (unsigned char)kMotorAppDefaultConfig[RouteIndex].can_id;
    g_motorAppFailedMotorID = motor_id;
    g_motorAppFailedRouteIndex = RouteIndex;
  }
  if (g_motorAppAdminOp == MOTOR_APP_ADMIN_ENABLE_ALL) {
    g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_MOTOR_ON;
  } else {
    g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_MOTOR_OFF;
  }
  first_fault = (unsigned char)(g_motorAppFault == 0U);
  g_motorAppFaultFlags |= FaultFlag;
  g_motorAppFault = 1U;
  g_motorAppEnabled = 0U;
  g_motorAppMode = MOTOR_APP_MODE_ADMIN;
  g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
  g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
  g_motorAppRouteIndex = 0xFFU;
  if (first_fault != 0U) {
    MotorApp_DisableAllRoutes();
  }
}

static unsigned char MotorApp_IsHardwareFaultState(unsigned char State)
{
  return (unsigned char)((State >= 0x8U) ? 1U : 0U);
}

static void MotorApp_CheckRuntimeProtection(void)
{
  unsigned char index;
  TickType_t now_tick;

  if (g_motorAppEnabled == 0U || g_motorAppFault != 0U) {
    return;
  }
  if (MotorApp_ImuHealthy() == 0U) {
    MotorApp_EnterFault(0xFFU, MOTOR_APP_FAULT_IMU_INVALID);
    return;
  }
  now_tick = xTaskGetTickCount();
  if (MotorApp_IsTickExpired(now_tick, g_motorAppProtectionStartTick) == 0U) {
    return;
  }

  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    Motor_t *motor = MotorApp_GetRouteMotor(index);
    if (MotorApp_IsActiveMitRoute(index) == 0U) {
      continue;
    }
    if (motor->IsOnline(MOTOR_APP_FEEDBACK_TIMEOUT_MS) == 0U) {
      MotorApp_EnterFault(index, MOTOR_APP_FAULT_FEEDBACK_STALE);
      return;
    }
    if (motor->GetMosTemp() >= MOTOR_APP_MOS_OVER_TEMP_C) {
      MotorApp_EnterFault(index, MOTOR_APP_FAULT_MOS_OVER_TEMP);
      return;
    }
    if (motor->GetRotorTemp() >= MOTOR_APP_ROTOR_OVER_TEMP_C) {
      MotorApp_EnterFault(index, MOTOR_APP_FAULT_ROTOR_OVER_TEMP);
      return;
    }
    if (motor->GetState() != 1U) {
      MotorApp_EnterFault(index, MOTOR_APP_FAULT_MOTOR_HW_FAULT);
      return;
    }
  }
}

void MotorApp_ConfigDefault(void)
{
  unsigned char index;

  MotorApp_ClearRuntimeState();
  g_positionLimitsReady = 0U;

  for (index = 0U; index < (unsigned char)(sizeof(kMotorAppDefaultConfig) / sizeof(kMotorAppDefaultConfig[0])); index++) {
    const MotorApp_DefaultMotorConfig_t *config = &kMotorAppDefaultConfig[index];
    g_positionMin[index] = config->p_min;
    g_positionMax[index] = config->p_max;
    if (config->port == 0U || config->can_id == 0U) continue;
    MotorPoint((unsigned char)config->can_id)->SetConfig(config->port,
                                                         config->can_id,
                                                         (unsigned short)(config->can_id + MOTOR_FEEDBACK_ID_P16_OFFSET));
    const MotorApp_WireRange_t *wire = &kMotorAppWireRange[index];
    MotorPoint((unsigned char)config->can_id)->SetMITFullRange(-wire->p_max, wire->p_max,
                                                               -wire->v_max, wire->v_max,
                                                               -wire->t_max, wire->t_max,
                                                               config->kp_min, config->kp_max,
                                                               config->kd_min, config->kd_max);
  }
  MotorApp_PrepareFallbackCommands();
}

unsigned char MotorApp_ConfigureStartupRegisters(void)
{
  unsigned char index;
  unsigned char route_indices[H7SPI_MAX_MOTORS];
  unsigned char result = H7SPI_RESULT_OK;

#if !defined(DM_CAN_FD_MOTOR) || (DM_CAN_FD_MOTOR == 0)
  g_motorAppStartupRegistersConfigured = 1U;
  return H7SPI_RESULT_OK;
#endif

  if (g_motorAppStartupRegistersConfigured != 0U) {
    return H7SPI_RESULT_OK;
  }
  if (g_motorAppEnabled != 0U ||
      g_motorAppMode != MOTOR_APP_MODE_ADMIN ||
      g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE ||
      g_motorAppRegisterBusy != 0U) {
    return H7SPI_RESULT_BUSY;
  }

  g_motorAppRegisterBusy = 1U;
  MotorApp_ClearAdminStats();
  g_motorAppRequestedCount = MotorApp_GetDefaultCount();
  for (index = 0U; index < g_motorAppRequestedCount; index++) {
    route_indices[index] = index;
    g_motorAppRequestedMask |= (1UL << index);
  }

  result = MotorApp_ConfigureProtectionRegisters(route_indices, g_motorAppRequestedCount);
  if (result != H7SPI_RESULT_OK) {
    g_motorAppRegisterBusy = 0U;
    return result;
  }
  result = MotorApp_ConfigureFdCanBr(route_indices, g_motorAppRequestedCount);
  if (result != H7SPI_RESULT_OK) {
    g_motorAppRegisterBusy = 0U;
    return result;
  }
  result = MotorApp_ConfigureTimeouts(route_indices, g_motorAppRequestedCount, 1U);
  if (result != H7SPI_RESULT_OK) {
    g_motorAppRegisterBusy = 0U;
    return result;
  }

  g_motorAppStartupRegistersConfigured = 1U;
  g_motorAppRegisterBusy = 0U;
  return H7SPI_RESULT_OK;
}

unsigned char MotorApp_RequestEnableDefault(void)
{
  unsigned char index;

  if (g_positionLimitsReady == 0U || MotorApp_ImuHealthy() == 0U)
    return H7SPI_RESULT_FAULT;

  if (g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE || g_motorAppRegisterBusy != 0U) {
    return H7SPI_RESULT_BUSY;
  }
  if (g_motorAppFault != 0U) {
    return H7SPI_RESULT_FAULT;
  }
  g_motorAppFaultFlags &= ~MOTOR_APP_FAULT_HOST_COMMAND_STALE;

  MotorApp_ClearAdminStats();
  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    if (MotorApp_IsActiveMitRoute(index) == 0U) continue;
    g_motorAppRequestedCount++;
    g_motorAppRequestedMask |= (1UL << index);
  }
  if (g_motorAppRequestedCount == 0U) return H7SPI_RESULT_BAD_MOTOR_COUNT;
  MotorApp_PrepareFallbackCommands();
  g_motorAppAdminOp = MOTOR_APP_ADMIN_ENABLE_ALL;
  g_motorAppPhase = MOTOR_APP_PHASE_ROUTE_SEND;
  g_motorAppRouteCursor = 0U;
  g_motorAppRouteIndex = 0xFFU;
  g_motorAppRetry = 0U;
  g_motorAppEnabled = 0U;
  g_motorAppMode = MOTOR_APP_MODE_ADMIN;
  g_motorAppFailedMotorID = 0U;
  g_motorAppFailedRouteIndex = 0xFFU;
  g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_NONE;
  return MotorApp_StartAllMitFast();
}

unsigned char MotorApp_RequestDisableDefault(void)
{
  unsigned char index;

  if (g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE) {
    return H7SPI_RESULT_BUSY;
  }

  MotorApp_ClearAdminStats();
  g_motorAppRequestedCount = MotorApp_GetDefaultCount();
  for (index = 0U; index < g_motorAppRequestedCount; index++) {
    g_motorAppRequestedMask |= (1UL << index);
  }
  g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_NONE;
  MotorApp_DisableAllRoutes();
  if (Motor_HasTxFailure() != 0U) {
    g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_MOTOR_OFF;
    g_motorAppFaultFlags |= MOTOR_APP_FAULT_CAN_TX_DROP;
    return H7SPI_RESULT_FAULT;
  }
  MotorApp_PrepareFallbackCommands();
  g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
  g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
  g_motorAppRouteCursor = 0U;
  g_motorAppRouteIndex = 0xFFU;
  g_motorAppRetry = 0U;
  g_motorAppEnabled = 0U;
  g_motorAppMode = MOTOR_APP_MODE_ADMIN;
  g_motorAppSucceededMask = g_motorAppRequestedMask;
  g_motorAppSuccessCount = g_motorAppRequestedCount;
  return H7SPI_RESULT_OK;
}

unsigned char MotorApp_ApplySpiAdminOp(const h7spi_admin_op_frame_t *AdminOp)
{
  unsigned char result;
  unsigned char index;
  unsigned char route_indices[H7SPI_MAX_MOTORS];

  if (AdminOp == 0) {
    return H7SPI_RESULT_BAD_LENGTH;
  }

  MotorApp_ClearAdminStats();

  if (AdminOp->op == H7SPI_ADMIN_OP_ENABLE_ALL) {
    if (AdminOp->motor_count != 0U) {
      return H7SPI_RESULT_BAD_MOTOR_COUNT;
    }
    return MotorApp_RequestEnableDefault();
  }

  if (AdminOp->op == H7SPI_ADMIN_OP_DISABLE_ALL) {
    result = MotorApp_PrepareAdminTargets(AdminOp->motor_codes, AdminOp->motor_count, route_indices);
    if (result != H7SPI_RESULT_OK) {
      return result;
    }
    g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_NONE;
    Motor_ClearTxFailure();
    for (index = 0U; index < g_motorAppRequestedCount; index++) {
      MotorApp_AdminDisableOne(MotorApp_GetRouteMotor(route_indices[index]));
      if (Motor_HasTxFailure() != 0U) {
        g_motorAppFailedMotorID = (unsigned char)kMotorAppDefaultConfig[route_indices[index]].can_id;
        g_motorAppFailedRouteIndex = route_indices[index];
        continue;
      }
      MotorApp_RecordAdminSuccess(route_indices[index]);
    }
    if (Motor_HasTxFailure() != 0U) {
      g_motorAppFailedPhase = H7SPI_GATEWAY_PHASE_MOTOR_OFF;
      g_motorAppFaultFlags |= MOTOR_APP_FAULT_CAN_TX_DROP;
      return H7SPI_RESULT_FAULT;
    }
    MotorApp_PrepareFallbackCommands();
    g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
    g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
    g_motorAppEnabled = 0U;
    g_motorAppMode = MOTOR_APP_MODE_ADMIN;
    g_motorAppRouteIndex = 0xFFU;
    return H7SPI_RESULT_OK;
  }

  if (AdminOp->op == H7SPI_ADMIN_OP_MARK_ZERO) {
    if (MotorApp_CalibrationAllowed() == 0U) return H7SPI_RESULT_BUSY;
    result = MotorApp_PrepareAdminTargets(AdminOp->motor_codes, AdminOp->motor_count, route_indices);
    if (result != H7SPI_RESULT_OK) return result;
    Motor_ClearTxFailure();
    for (index = 0U; index < g_motorAppRequestedCount; index++) {
      MotorApp_AdminMarkZeroOne(MotorApp_GetRouteMotor(route_indices[index]));
      (void)Motor_SendAllOnce();
      if (Motor_HasTxFailure() != 0U) return H7SPI_RESULT_FAULT;
      MotorApp_RecordAdminSuccess(route_indices[index]);
    }
    MotorApp_ClearCommandTracking();
    MotorApp_PrepareFallbackCommands();
    return H7SPI_RESULT_OK;
  }

  if (AdminOp->op == H7SPI_ADMIN_OP_CLEAR_ERROR) {
    if (g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE) {
      return H7SPI_RESULT_BUSY;
    }
    result = MotorApp_PrepareAdminTargets(AdminOp->motor_codes, AdminOp->motor_count, route_indices);
    if (result != H7SPI_RESULT_OK) {
      return result;
    }
    for (index = 0U; index < g_motorAppRequestedCount; index++) {
      g_motorAppAdminRouteIndices[index] = route_indices[index];
    }
    g_motorAppAdminOp = MOTOR_APP_ADMIN_CLEAR_ERROR;
    g_motorAppPhase = MOTOR_APP_PHASE_ROUTE_SEND;
    g_motorAppRouteCursor = 0U;
    g_motorAppRouteIndex = 0xFFU;
    g_motorAppRetry = 0U;
    g_motorAppEnabled = 0U;
    g_motorAppFault = 0U;
    g_motorAppMode = MOTOR_APP_MODE_ADMIN;
    g_motorAppFailedMotorID = 0U;
    g_motorAppFailedRouteIndex = 0xFFU;
    g_motorAppFaultFlags = MOTOR_APP_FAULT_NONE;
    g_motorAppLastCommandSeq = 0U;
    g_motorAppDeadlineTick = 0U;
    g_motorAppRouteStartRxTick = 0U;
    g_motorAppProtectionStartTick = 0U;
    MotorApp_PrepareFallbackCommands();
    return H7SPI_RESULT_OK;
  }

  if (AdminOp->op == H7SPI_ADMIN_OP_CONFIG_TIMEOUT) {
    if (g_motorAppEnabled != 0U ||
        g_motorAppMode != MOTOR_APP_MODE_ADMIN ||
        g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE) {
      return H7SPI_RESULT_BUSY;
    }
    result = MotorApp_PrepareAdminTargets(AdminOp->motor_codes, AdminOp->motor_count, route_indices);
    if (result != H7SPI_RESULT_OK) {
      return result;
    }
    result = MotorApp_ConfigureProtectionRegisters(route_indices, g_motorAppRequestedCount);
    if (result != H7SPI_RESULT_OK) {
      return result;
    }
    result = MotorApp_ConfigureFdCanBr(route_indices, g_motorAppRequestedCount);
    if (result != H7SPI_RESULT_OK) {
      return result;
    }
    return MotorApp_ConfigureTimeouts(route_indices, g_motorAppRequestedCount, 1U);
  }

  return H7SPI_RESULT_BAD_TYPE;
}

void MotorApp_EnableDefault(void)
{
  (void)MotorApp_RequestEnableDefault();
}

static void MotorApp_ProcessReadOnlyProbe(void)
{
  static const unsigned char regs[] = {0x08U, 0x07U, 0x0AU, 0x0EU, 0x3CU};
  uint32_t requested = motor_probe_request;
  unsigned char selected = (unsigned char)(requested >> 8);
  unsigned char motor_id = (unsigned char)requested;
  unsigned char route;
  unsigned char index;
  if (requested == 0U) return;
  motor_probe_request = 0U;
  for (index = 0U; index < 8U; index++) motor_probe_results[index] = 0U;
  motor_probe_results[0] = requested;
  if (requested > 65535U || g_motorAppEnabled != 0U ||
      g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE || g_motorAppRegisterBusy != 0U ||
      MotorApp_GetRouteIndexByMotorID(motor_id, &route) == 0U) {
    motor_probe_results[1] = 2U;
    return;
  }
  if (selected != 0U && selected != 0x09U && selected != 0x15U &&
      selected != 0x16U && selected != 0x17U && selected != 0x23U && selected != 0x24U) {
    motor_probe_results[1] = 2U;
    return;
  }
  g_motorAppRegisterBusy = 1U;
  Motor_t *motor = MotorApp_GetRouteMotor(route);
  if (selected != 0U) {
    unsigned int value;
    motor_probe_results[3] = selected;
    if (motor->ReadRegister(selected, &value) != 0U) {
      motor_probe_results[4] = value;
      motor_probe_results[2] = 1U;
    }
    g_motorAppRegisterBusy = 0U;
    motor_probe_results[1] = 1U;
    return;
  }
  for (index = 0U; index < sizeof(regs); index++) {
    unsigned int value;
    if (motor->ReadRegister(regs[index], &value) != 0U) {
      motor_probe_results[3U + index] = value;
      motor_probe_results[2] |= (1UL << index);
    }
  }
  g_motorAppRegisterBusy = 0U;
  motor_probe_results[1] = 1U;
}

static void MotorApp_CheckHostWatchdog(void)
{
  if (g_motorAppEnabled != 0U &&
      (HAL_GetTick() - g_motorAppLastHostCommandTickMs) > MOTOR_APP_HOST_COMMAND_TIMEOUT_MS) {
    MotorApp_DisableAllRoutes();
    g_motorAppEnabled = 0U;
    g_motorAppMode = MOTOR_APP_MODE_ADMIN;
    g_motorAppFaultFlags |= MOTOR_APP_FAULT_HOST_COMMAND_STALE;
  }
}

void MotorApp_Tick(void)
{
  TickType_t now_tick;
  Motor_t *motor;
  unsigned char index;

  MotorApp_ProcessReadOnlyProbe();
  now_tick = xTaskGetTickCount();

  if (g_motorAppAdminOp == MOTOR_APP_ADMIN_ENABLE_ALL &&
      g_motorAppPhase == MOTOR_APP_PHASE_FAST_START) return;

  if (g_motorAppAdminOp == MOTOR_APP_ADMIN_ENABLE_ALL) {
    if (g_motorAppPhase == MOTOR_APP_PHASE_ROUTE_SEND) {
      if (g_motorAppRouteCursor >= MotorApp_GetDefaultCount()) {
        MotorApp_PrepareFallbackCommands();
        for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
          if (MotorApp_IsActiveMitRoute(index) != 0U) {
            MotorApp_GetRouteMotor(index)->SetRunFlag(1U);
          }
        }
        g_motorAppEnabled = 1U;
        g_motorAppMode = MOTOR_APP_MODE_CONTROL_MIT;
        g_motorAppLastHostCommandTickMs = HAL_GetTick();
        g_motorAppSucceededMask = g_motorAppRequestedMask;
        g_motorAppSuccessCount = g_motorAppRequestedCount;
        g_motorAppProtectionStartTick = now_tick;
        g_motorAppAdminOp = MOTOR_APP_ADMIN_NONE;
        g_motorAppPhase = MOTOR_APP_PHASE_IDLE;
        MotorApp_SendActiveMitOnce();
        return;
      }

      MotorApp_StartRouteEnable(g_motorAppRouteCursor);
      MotorApp_SendActiveMitOnce();
      return;
    }

    if (g_motorAppPhase == MOTOR_APP_PHASE_ROUTE_WAIT) {
      motor = MotorApp_GetRouteMotor(g_motorAppRouteIndex);
      if (motor->IsEnable() != 0U && motor->GetLastRxTick() != g_motorAppRouteStartRxTick) {
        motor->ClearCommand();
        motor->SetRunFlag(1U);
        g_motorAppRouteCursor++;
        g_motorAppRouteIndex = 0xFFU;
        g_motorAppRetry = 0U;
        g_motorAppPhase = MOTOR_APP_PHASE_SETTLE_WAIT;
        g_motorAppDeadlineTick = now_tick + pdMS_TO_TICKS(MOTOR_APP_ROUTE_SETTLE_MS);
        MotorApp_SendActiveMitOnce();
        return;
      }

      if (MotorApp_IsHardwareFaultState(motor->GetState()) != 0U) {
        MotorApp_EnterFault(g_motorAppRouteIndex, MOTOR_APP_FAULT_MOTOR_HW_FAULT);
        return;
      }

      if (MotorApp_IsTickExpired(now_tick, g_motorAppDeadlineTick) != 0U) {
        if (g_motorAppRetry < MOTOR_APP_MAX_RETRIES) {
          g_motorAppRetry++;
          MotorApp_StartRouteEnable(g_motorAppRouteIndex);
        } else {
          MotorApp_EnterFault(g_motorAppRouteIndex, MOTOR_APP_FAULT_ENABLE_FAILED);
        }
      }
      MotorApp_SendActiveMitOnce();
      return;
    }

    if (g_motorAppPhase == MOTOR_APP_PHASE_SETTLE_WAIT) {
      if (MotorApp_IsTickExpired(now_tick, g_motorAppDeadlineTick) != 0U) {
        g_motorAppPhase = MOTOR_APP_PHASE_ROUTE_SEND;
      }
      MotorApp_SendActiveMitOnce();
      return;
    }
  }

  if (g_motorAppAdminOp == MOTOR_APP_ADMIN_CLEAR_ERROR) {
    if (g_motorAppPhase == MOTOR_APP_PHASE_ROUTE_SEND) {
      if (g_motorAppRouteCursor >= g_motorAppRequestedCount) {
        MotorApp_FinishClearError();
        return;
      }

      MotorApp_StartRouteClearError(g_motorAppAdminRouteIndices[g_motorAppRouteCursor]);
      return;
    }

    if (g_motorAppPhase == MOTOR_APP_PHASE_ROUTE_WAIT) {
      motor = MotorApp_GetRouteMotor(g_motorAppRouteIndex);
      if (motor->GetLastRxTick() != g_motorAppRouteStartRxTick &&
          MotorApp_IsHardwareFaultState(motor->GetState()) == 0U) {
        MotorApp_RecordAdminSuccess(g_motorAppRouteIndex);
        g_motorAppRouteCursor++;
        g_motorAppRouteIndex = 0xFFU;
        g_motorAppRetry = 0U;
        g_motorAppPhase = MOTOR_APP_PHASE_ROUTE_SEND;
        return;
      }

      if (MotorApp_IsTickExpired(now_tick, g_motorAppDeadlineTick) != 0U) {
        if (g_motorAppRetry < MOTOR_APP_MAX_RETRIES) {
          g_motorAppRetry++;
          MotorApp_StartRouteClearError(g_motorAppRouteIndex);
        } else {
          MotorApp_FailClearError(g_motorAppRouteIndex);
        }
      }
      return;
    }
  }

  MotorApp_CheckRuntimeProtection();

  MotorApp_CheckHostWatchdog();

  if (g_motorAppEnabled != 0U && g_motorAppFault == 0U) {
    MotorApp_SendActiveMitOnce();
  }

  if (g_motorAppEnabled == 0U &&
      (g_motorAppFault != 0U || (g_motorAppFaultFlags & MOTOR_APP_FAULT_HOST_COMMAND_STALE) != 0U)) {
    static uint32_t last_disable_retry;
    if ((uint32_t)(HAL_GetTick() - last_disable_retry) >= 20U) {
      last_disable_retry = HAL_GetTick();
      for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
        if (MotorApp_IsActiveMitRoute(index) == 0U) continue;
        motor = MotorApp_GetRouteMotor(index);
        if (motor->IsOnline(MOTOR_APP_FEEDBACK_TIMEOUT_MS) == 0U || motor->IsEnable() != 0U) {
          motor->SetRunFlag(0U);
          motor->SendDisableFrame();
        }
      }
    }
  }
  MotorApp_AdminPollFeedback(now_tick);
}

unsigned char MotorApp_IsEnabled(void)
{
  return g_motorAppEnabled;
}

unsigned char MotorApp_IsFault(void)
{
  return g_motorAppFault;
}

unsigned char MotorApp_GetFailedMotorID(void)
{
  return g_motorAppFailedMotorID;
}

unsigned long MotorApp_GetFaultFlags(void)
{
  return g_motorAppFaultFlags;
}

unsigned char MotorApp_GetMode(void)
{
  return g_motorAppMode;
}

unsigned char MotorApp_GetMotorCount(void)
{
  return MotorApp_GetDefaultCount();
}

unsigned char MotorApp_GetMotorIdByIndex(unsigned char Index)
{
  if (Index >= MotorApp_GetDefaultCount()) {
    return 0U;
  }
  return (unsigned char)kMotorAppDefaultConfig[Index].can_id;
}

unsigned short MotorApp_GetLastCommandSeq(void)
{
  return g_motorAppLastCommandSeq;
}

void MotorApp_GetStatus(MotorApp_Status_t *Status)
{
  unsigned char index;
  unsigned long enabled_mask = 0UL;

  if (Status == 0) {
    return;
  }

  for (index = 0U; index < MotorApp_GetDefaultCount(); index++) {
    Motor_t *motor = MotorApp_GetRouteMotor(index);
    if (g_motorAppEnabled != 0U && motor->IsEnable() != 0U) {
      enabled_mask |= (1UL << ((unsigned char)kMotorAppDefaultConfig[index].can_id));
    }
  }

  Status->fault_flags = g_motorAppFaultFlags;
  Status->mode = g_motorAppMode;
  Status->enabled = g_motorAppEnabled;
  Status->fault = g_motorAppFault;
  Status->busy = (unsigned char)((g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE ||
                                  g_motorAppRegisterBusy != 0U) ? 1U : 0U);
  Status->failed_motor_id = g_motorAppFailedMotorID;
  Status->failed_route_index = g_motorAppFailedRouteIndex;
  Status->motor_count = MotorApp_GetDefaultCount();
  if (g_motorAppFault != 0U) {
    Status->lifecycle_state = H7SPI_GATEWAY_LIFECYCLE_FAULT;
  } else if (g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE) {
    Status->lifecycle_state = H7SPI_GATEWAY_LIFECYCLE_STARTING;
  } else if (g_motorAppEnabled != 0U && g_motorAppMode == MOTOR_APP_MODE_CONTROL_MIT) {
    Status->lifecycle_state = H7SPI_GATEWAY_LIFECYCLE_RUNNING_CONTROL;
  } else {
    Status->lifecycle_state = H7SPI_GATEWAY_LIFECYCLE_RUNNING_ADMIN;
  }
  Status->failed_phase = g_motorAppFailedPhase;
  Status->enabled_motor_mask = enabled_mask;
  Status->requested_mask = g_motorAppRequestedMask;
  Status->succeeded_mask = g_motorAppSucceededMask;
  Status->requested_count = g_motorAppRequestedCount;
  Status->success_count = g_motorAppSuccessCount;
}

static unsigned char MotorApp_ShapeMitCommand(const h7spi_motor_cmd_t *Cmd,
                                              unsigned char RouteIndex,
                                              MotorApp_ShapedCommand_t *Out)
{
  const MotorApp_DefaultMotorConfig_t *config;
  Motor_t *motor;
  float position;
  float speed;
  float kp;
  float kd;
  float torque;
  float limit;
  float tau_feedback;
  float tau_total;
  float scale;
  unsigned char result = H7SPI_RESULT_OK;

  if (Cmd == 0 || Out == 0 || RouteIndex >= MotorApp_GetDefaultCount()) {
    return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
  }

  config = &kMotorAppDefaultConfig[RouteIndex];
  motor = MotorApp_GetRouteMotor(RouteIndex);
  position = ((float)Cmd->p_mrad) / 1000.0f;
  speed = ((float)Cmd->v_mrad_s) / 1000.0f;
  kp = ((float)Cmd->kp_centi) / 100.0f;
  kd = ((float)Cmd->kd_milli) / 1000.0f;
  torque = ((float)Cmd->torque_mnm) / 1000.0f;

  if (position < g_positionMin[RouteIndex] || position > g_positionMax[RouteIndex]) {
    position = MotorApp_LimitFloat(position, g_positionMin[RouteIndex], g_positionMax[RouteIndex]);
    result = H7SPI_RESULT_COMMAND_POSITION_LIMIT;
  }
  if (speed < config->v_min || speed > config->v_max) {
    speed = MotorApp_LimitFloat(speed, config->v_min, config->v_max);
    if (result == H7SPI_RESULT_OK) {
      result = H7SPI_RESULT_LIMIT_EXCEEDED;
    }
  }

  kp = MotorApp_LimitFloat(kp, config->kp_min, config->kp_max);
  kd = MotorApp_LimitFloat(kd, config->kd_min, config->kd_max);
  limit = MotorApp_MaxFloat(MotorApp_AbsFloat(config->t_min), MotorApp_AbsFloat(config->t_max));
  if (limit <= 0.0f) {
    limit = MotorApp_AbsFloat(torque);
  }
  if (limit > 0.0f && MotorApp_AbsFloat(torque) > limit) {
    torque = MotorApp_LimitFloat(torque, -limit, limit);
    if (result == H7SPI_RESULT_OK) {
      result = H7SPI_RESULT_COMMAND_TORQUE_LIMIT;
    }
  }

  tau_feedback = (kp * (position - motor->GetPosition())) +
                 (kd * (speed - motor->GetSpeed()));
  tau_total = tau_feedback + torque;
  if (limit > 0.0f && MotorApp_AbsFloat(tau_total) > limit &&
      MotorApp_AbsFloat(tau_feedback) > 0.0001f) {
    scale = (limit - MotorApp_AbsFloat(torque)) / MotorApp_AbsFloat(tau_feedback);
    scale = MotorApp_LimitFloat(scale, 0.0f, 1.0f);
    kp *= scale;
    kd *= scale;
  }

  Out->position = position;
  Out->speed = speed;
  Out->kp = kp;
  Out->kd = kd;
  Out->torque = torque;
  return result;
}

unsigned char MotorApp_ReadSpiMotorRegs(const h7spi_motor_reg_request_frame_t *Request,
                                        h7spi_motor_reg_values_frame_t *Response)
{
  unsigned char index;
  unsigned char route_indices[H7SPI_MAX_MOTORS];
  unsigned char result;

  if (Request == 0 || Response == 0) {
    return H7SPI_RESULT_BAD_LENGTH;
  }

  memset(Response, 0, sizeof(*Response));
  Response->response_seq = Request->request_seq;
  Response->reg_addr = Request->reg_addr;
  Response->stm32_tick_ms = HAL_GetTick();
  Response->fault_flags = g_motorAppFaultFlags;

  if (g_motorAppMode != MOTOR_APP_MODE_ADMIN ||
      g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE) {
    Response->result = H7SPI_RESULT_BUSY;
    return H7SPI_RESULT_BUSY;
  }

  g_motorAppRegisterBusy = 1U;
  result = MotorApp_PrepareAdminTargets(Request->motor_codes,
                                        Request->motor_count,
                                        route_indices);
  Response->motor_count = g_motorAppRequestedCount;
  if (result != H7SPI_RESULT_OK) {
    Response->result = result;
    Response->failed_motor_id = g_motorAppFailedMotorID;
    Response->failed_route_index = g_motorAppFailedRouteIndex;
    g_motorAppRegisterBusy = 0U;
    return result;
  }

  for (index = 0U; index < g_motorAppRequestedCount; index++) {
    unsigned int value = 0U;
    unsigned char route_index = route_indices[index];
    Motor_t *motor = MotorApp_GetRouteMotor(route_index);

    Response->motor_codes[index] = (unsigned char)kMotorAppDefaultConfig[route_index].can_id;
    if (motor->ReadRegister(Request->reg_addr, &value) == 0U) {
      g_motorAppFailedMotorID = (unsigned char)kMotorAppDefaultConfig[route_index].can_id;
      g_motorAppFailedRouteIndex = route_index;
      Response->failed_motor_id = g_motorAppFailedMotorID;
      Response->failed_route_index = g_motorAppFailedRouteIndex;
      Response->result = H7SPI_RESULT_FAULT;
      g_motorAppRegisterBusy = 0U;
      return H7SPI_RESULT_FAULT;
    }
    Response->values[index] = (int32_t)value;
    MotorApp_RecordAdminSuccess(route_index);
  }

  Response->result = H7SPI_RESULT_OK;
  g_motorAppRegisterBusy = 0U;
  return H7SPI_RESULT_OK;
}

unsigned char MotorApp_WriteSpiMotorRegs(const h7spi_motor_reg_write_request_frame_t *Request,
                                         h7spi_motor_reg_values_frame_t *Response)
{
  unsigned char index;
  unsigned char route_indices[H7SPI_MAX_MOTORS];
  unsigned char result;

  if (Request == 0 || Response == 0) {
    return H7SPI_RESULT_BAD_LENGTH;
  }

  memset(Response, 0, sizeof(*Response));
  Response->response_seq = Request->request_seq;
  Response->reg_addr = Request->reg_addr;
  Response->stm32_tick_ms = HAL_GetTick();
  Response->fault_flags = g_motorAppFaultFlags;

  if (g_motorAppMode != MOTOR_APP_MODE_ADMIN ||
      g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE) {
    Response->result = H7SPI_RESULT_BUSY;
    return H7SPI_RESULT_BUSY;
  }

  g_motorAppRegisterBusy = 1U;
  result = MotorApp_PrepareAdminTargets(Request->motor_codes,
                                        Request->motor_count,
                                        route_indices);
  Response->motor_count = g_motorAppRequestedCount;
  if (result != H7SPI_RESULT_OK) {
    Response->result = result;
    Response->failed_motor_id = g_motorAppFailedMotorID;
    Response->failed_route_index = g_motorAppFailedRouteIndex;
    g_motorAppRegisterBusy = 0U;
    return result;
  }

  for (index = 0U; index < g_motorAppRequestedCount; index++) {
    unsigned int value = (unsigned int)Request->values[index];
    unsigned char route_index = route_indices[index];
    Motor_t *motor = MotorApp_GetRouteMotor(route_index);

    Response->motor_codes[index] = (unsigned char)kMotorAppDefaultConfig[route_index].can_id;
    Response->values[index] = Request->values[index];
    if (motor->WriteRegister(Request->reg_addr, value, Request->save_after_write) == 0U) {
      g_motorAppFailedMotorID = (unsigned char)kMotorAppDefaultConfig[route_index].can_id;
      g_motorAppFailedRouteIndex = route_index;
      Response->failed_motor_id = g_motorAppFailedMotorID;
      Response->failed_route_index = g_motorAppFailedRouteIndex;
      Response->result = H7SPI_RESULT_FAULT;
      g_motorAppRegisterBusy = 0U;
      return H7SPI_RESULT_FAULT;
    }
    MotorApp_RecordAdminSuccess(route_index);
  }

  Response->result = H7SPI_RESULT_OK;
  g_motorAppRegisterBusy = 0U;
  return H7SPI_RESULT_OK;
}

unsigned char MotorApp_ApplySpiMotorCommand(const h7spi_motor_cmd_frame_t *Command)
{
  unsigned char index;
  unsigned char count;
  unsigned int seen_mask = 0U;
  unsigned int now_ms = HAL_GetTick();
  MotorApp_ShapedCommand_t shaped[H7SPI_MAX_MOTORS];
  unsigned char route_indices[H7SPI_MAX_MOTORS];
  unsigned char final_result = H7SPI_RESULT_OK;

  if (Command == 0) {
    return H7SPI_RESULT_BAD_LENGTH;
  }
  if (Command->mode == H7SPI_MODE_DISABLED) {
    MotorApp_DisableAllRoutes();
    g_motorAppEnabled = 0U;
    g_motorAppMode = H7SPI_MODE_DISABLED;
    g_motorAppLastCommandSeq = Command->command_seq;
    return H7SPI_RESULT_OK;
  }
  if (Command->mode == H7SPI_MODE_ADMIN) {
    MotorApp_ClearCommandTracking();
    g_motorAppMode = H7SPI_MODE_ADMIN;
    g_motorAppLastCommandSeq = Command->command_seq;
    return H7SPI_RESULT_OK;
  }
  if (Command->mode != H7SPI_MODE_CONTROL_MIT) {
    return H7SPI_RESULT_BAD_MODE;
  }
  if (g_motorAppFault != 0U) {
    return H7SPI_RESULT_FAULT;
  }
  if (g_motorAppEnabled == 0U ||
      g_motorAppMode != MOTOR_APP_MODE_CONTROL_MIT ||
      g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE) {
    return H7SPI_RESULT_BUSY;
  }

  count = Command->motor_count;
  if (count > H7SPI_MAX_MOTORS || count > MotorApp_GetDefaultCount()) {
    return H7SPI_RESULT_BAD_MOTOR_COUNT;
  }

  memset(g_motorAppUpdatedByLastFrame, 0, sizeof(g_motorAppUpdatedByLastFrame));

  for (index = 0U; index < count; index++) {
    const h7spi_motor_cmd_t *cmd = &Command->motors[index];
    unsigned int bit;
    unsigned char route_index;

    if (cmd->motor_id >= MOTOR_NUM) {
      g_motorAppFailedMotorID = cmd->motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
    }

    bit = (1UL << cmd->motor_id);
    if ((seen_mask & bit) != 0U) {
      g_motorAppFailedMotorID = cmd->motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_DUPLICATE_MOTOR_ID;
    }
    seen_mask |= bit;

    if (MotorPoint(cmd->motor_id)->IsConfigured() == 0U) {
      g_motorAppFailedMotorID = cmd->motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
    }
    if (MotorApp_GetRouteIndexByMotorID(cmd->motor_id, &route_index) == 0U) {
      g_motorAppFailedMotorID = cmd->motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
    }
    route_indices[index] = route_index;
    {
      unsigned char shape_result = MotorApp_ShapeMitCommand(cmd, route_index, &shaped[index]);
      if (shape_result != H7SPI_RESULT_OK && MotorApp_IsCommandWarning(shape_result) == 0U) {
        g_motorAppFailedMotorID = cmd->motor_id;
        g_motorAppFailedRouteIndex = index;
        return shape_result;
      }
      if (shape_result != H7SPI_RESULT_OK && final_result == H7SPI_RESULT_OK) {
        final_result = shape_result;
      }
    }
  }

  for (index = 0U; index < count; index++) {
    const h7spi_motor_cmd_t *cmd = &Command->motors[index];
    Motor_t *motor;

    (void)route_indices[index];
    motor = MotorPoint(cmd->motor_id);
    motor->SetMITCommand(shaped[index].position,
                         shaped[index].speed,
                         shaped[index].kp,
                         shaped[index].kd,
                         shaped[index].torque);
    g_motorAppMotorLastCommandSeq[cmd->motor_id] = Command->command_seq;
    g_motorAppMotorLastCommandTickMs[cmd->motor_id] = now_ms;
    g_motorAppMotorCommandValid[cmd->motor_id] = 1U;
    g_motorAppUpdatedByLastFrame[cmd->motor_id] = 1U;
  }

  g_motorAppMode = H7SPI_MODE_CONTROL_MIT;
  g_motorAppLastCommandSeq = Command->command_seq;
  g_motorAppLastHostCommandTickMs = now_ms;
  return final_result;
}

unsigned char MotorApp_SubmitSpiMotorCommandAsync(const h7spi_motor_cmd_frame_t *Command)
{
  unsigned short payload_len;

  if (Command == 0 || Command->motor_count > H7SPI_MAX_MOTORS ||
      Command->motor_count > MotorApp_GetDefaultCount()) {
    return H7SPI_RESULT_BAD_MOTOR_COUNT;
  }

  payload_len = (unsigned short)(offsetof(h7spi_motor_cmd_frame_t, motors) +
                                ((unsigned short)Command->motor_count * sizeof(h7spi_motor_cmd_t)));

  taskENTER_CRITICAL();
  memset(&g_motorAppPendingSpiCommand, 0, sizeof(g_motorAppPendingSpiCommand));
  memcpy(&g_motorAppPendingSpiCommand, Command, payload_len);
  g_motorAppPendingSpiCommandValid = 1U;
  taskEXIT_CRITICAL();
  return H7SPI_RESULT_OK;
}

void MotorApp_ProcessPendingSpiMotorCommand(void)
{
  h7spi_motor_cmd_frame_t command;
  unsigned char has_command;

  taskENTER_CRITICAL();
  has_command = g_motorAppPendingSpiCommandValid;
  if (has_command != 0U) {
    command = g_motorAppPendingSpiCommand;
    g_motorAppPendingSpiCommandValid = 0U;
  }
  taskEXIT_CRITICAL();

  if (has_command != 0U) {
    (void)MotorApp_ApplySpiMotorCommand(&command);
  }
}

unsigned char MotorApp_ApplySpiMotorCommandCompact(const h7spi_motor_cmd_compact_frame_t *Command)
{
  unsigned char index;
  unsigned char count;
  unsigned int seen_mask = 0U;
  unsigned int now_ms = HAL_GetTick();
  MotorApp_ShapedCommand_t shaped[H7SPI_MAX_MOTORS];
  unsigned char route_indices[H7SPI_MAX_MOTORS];
  unsigned char final_result = H7SPI_RESULT_OK;

  if (Command == 0) {
    return H7SPI_RESULT_BAD_LENGTH;
  }
  if (Command->mode == H7SPI_MODE_DISABLED) {
    MotorApp_DisableAllRoutes();
    g_motorAppEnabled = 0U;
    g_motorAppMode = H7SPI_MODE_DISABLED;
    g_motorAppLastCommandSeq = Command->command_seq;
    return H7SPI_RESULT_OK;
  }
  if (Command->mode == H7SPI_MODE_ADMIN) {
    MotorApp_ClearCommandTracking();
    g_motorAppMode = H7SPI_MODE_ADMIN;
    g_motorAppLastCommandSeq = Command->command_seq;
    return H7SPI_RESULT_OK;
  }
  if (Command->mode != H7SPI_MODE_CONTROL_MIT) {
    return H7SPI_RESULT_BAD_MODE;
  }
  if (g_motorAppFault != 0U) {
    return H7SPI_RESULT_FAULT;
  }
  if (g_motorAppEnabled == 0U ||
      g_motorAppMode != MOTOR_APP_MODE_CONTROL_MIT ||
      g_motorAppAdminOp != MOTOR_APP_ADMIN_NONE) {
    return H7SPI_RESULT_BUSY;
  }

  count = Command->motor_count;
  if (count > H7SPI_MAX_MOTORS || count > MotorApp_GetDefaultCount()) {
    return H7SPI_RESULT_BAD_MOTOR_COUNT;
  }

  memset(g_motorAppUpdatedByLastFrame, 0, sizeof(g_motorAppUpdatedByLastFrame));

  for (index = 0U; index < count; index++) {
    const h7spi_motor_cmd_compact_t *cmd = &Command->motors[index];
    h7spi_motor_cmd_t full_cmd;
    unsigned int bit;
    unsigned char route_index;

    if (cmd->motor_id >= MOTOR_NUM) {
      g_motorAppFailedMotorID = cmd->motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
    }

    bit = (1UL << cmd->motor_id);
    if ((seen_mask & bit) != 0U) {
      g_motorAppFailedMotorID = cmd->motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_DUPLICATE_MOTOR_ID;
    }
    seen_mask |= bit;

    if (MotorPoint(cmd->motor_id)->IsConfigured() == 0U ||
        MotorApp_GetRouteIndexByMotorID(cmd->motor_id, &route_index) == 0U) {
      g_motorAppFailedMotorID = cmd->motor_id;
      g_motorAppFailedRouteIndex = index;
      return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
    }
    route_indices[index] = route_index;

    memset(&full_cmd, 0, sizeof(full_cmd));
    full_cmd.motor_id = cmd->motor_id;
    full_cmd.flags = cmd->flags;
    full_cmd.p_mrad = (int32_t)cmd->p_10mrad * 10;
    full_cmd.v_mrad_s = (int32_t)cmd->v_10mrad_s * 10;
    full_cmd.torque_mnm = (int32_t)cmd->torque_10mnm * 10;
    full_cmd.kp_centi = cmd->kp_centi;
    full_cmd.kd_milli = cmd->kd_milli;
    {
      unsigned char shape_result = MotorApp_ShapeMitCommand(&full_cmd, route_index, &shaped[index]);
      if (shape_result != H7SPI_RESULT_OK && MotorApp_IsCommandWarning(shape_result) == 0U) {
        g_motorAppFailedMotorID = cmd->motor_id;
        g_motorAppFailedRouteIndex = index;
        return shape_result;
      }
      if (shape_result != H7SPI_RESULT_OK && final_result == H7SPI_RESULT_OK) {
        final_result = shape_result;
      }
    }
  }

  for (index = 0U; index < count; index++) {
    const h7spi_motor_cmd_compact_t *cmd = &Command->motors[index];
    Motor_t *motor;

    (void)route_indices[index];
    motor = MotorPoint(cmd->motor_id);
    motor->SetMITCommand(shaped[index].position,
                         shaped[index].speed,
                         shaped[index].kp,
                         shaped[index].kd,
                         shaped[index].torque);
    g_motorAppMotorLastCommandSeq[cmd->motor_id] = Command->command_seq;
    g_motorAppMotorLastCommandTickMs[cmd->motor_id] = now_ms;
    g_motorAppMotorCommandValid[cmd->motor_id] = 1U;
    g_motorAppUpdatedByLastFrame[cmd->motor_id] = 1U;
  }

  g_motorAppMode = H7SPI_MODE_CONTROL_MIT;
  g_motorAppLastCommandSeq = Command->command_seq;
  g_motorAppLastHostCommandTickMs = now_ms;
  return final_result;
}

void MotorApp_FillSpiState(h7spi_motor_state_frame_t *State, unsigned short AckCommandSeq)
{
  unsigned char index;
  unsigned char count;

  if (State == 0) {
    return;
  }

  memset(State, 0, sizeof(*State));
  count = MotorApp_GetDefaultCount();
  State->state_seq = ++g_motorAppStateSeq;
  State->ack_command_seq = AckCommandSeq;
  State->stm32_tick_ms = HAL_GetTick();
  State->fault_flags = g_motorAppFaultFlags;
  State->mode = g_motorAppMode;
  State->motor_count = count;
  State->reserved = (uint16_t)(((uint16_t)g_motorAppFailedMotorID << 8) |
                               (uint16_t)g_motorAppFailedRouteIndex);

  for (index = 0U; index < count; index++) {
    Motor_t *motor = MotorApp_GetRouteMotor(index);
    h7spi_motor_state_t *out = &State->motors[index];
    unsigned char flags = 0U;
    unsigned char motor_id = (unsigned char)kMotorAppDefaultConfig[index].can_id;
    unsigned int age_ms = 0U;

    if (kMotorAppDefaultConfig[index].port == 0U) {
      continue;
    }

    out->motor_id = motor_id;
    out->state = motor->GetState();
    out->temperature = motor->GetRotorTemp();
    out->p_mrad = (int32_t)(motor->GetPosition() * 1000.0f);
    out->v_mrad_s = (int32_t)(motor->GetSpeed() * 1000.0f);
    out->torque_mnm = (int32_t)(motor->GetTorque() * 1000.0f);
    out->rx_ts_ms = (uint32_t)motor->GetLastRxTick();
    out->last_command_seq = g_motorAppMotorLastCommandSeq[motor_id];
    if (g_motorAppMotorCommandValid[motor_id] != 0U) {
      age_ms = HAL_GetTick() - g_motorAppMotorLastCommandTickMs[motor_id];
      out->command_age_ms = (age_ms > 0xFFFFU) ? 0xFFFFU : (uint16_t)age_ms;
    }
    if (motor->IsConfigured() != 0U) {
      flags |= H7SPI_MOTOR_FLAG_CONFIGURED;
    }
    if (motor->IsOnline(MOTOR_APP_FEEDBACK_TIMEOUT_MS) != 0U) {
      flags |= H7SPI_MOTOR_FLAG_ONLINE;
    }
    if (g_motorAppEnabled != 0U && motor->IsEnable() != 0U) {
      flags |= H7SPI_MOTOR_FLAG_ENABLED;
    }
    if (MotorApp_IsHardwareFaultState(motor->GetState()) != 0U) {
      flags |= H7SPI_MOTOR_FLAG_FAULT;
    }
    if (g_motorAppMotorCommandValid[motor_id] != 0U) {
      flags |= H7SPI_MOTOR_FLAG_CMD_VALID;
    }
    if (g_motorAppUpdatedByLastFrame[motor_id] != 0U) {
      flags |= H7SPI_MOTOR_FLAG_UPDATED_BY_LAST_FRAME;
    }
    out->flags = flags;
  }
}

static int16_t MotorApp_SaturateI16(int32_t Value)
{
  if (Value > 32767) {
    return 32767;
  }
  if (Value < -32768) {
    return -32768;
  }
  return (int16_t)Value;
}

void MotorApp_FillSpiStateCompact(h7spi_motor_state_compact_frame_t *State, unsigned short AckCommandSeq)
{
  unsigned char index;
  unsigned char count;

  if (State == 0) {
    return;
  }

  memset(State, 0, sizeof(*State));
  count = MotorApp_GetDefaultCount();
  State->state_seq = ++g_motorAppStateSeq;
  State->ack_command_seq = AckCommandSeq;
  State->stm32_tick_ms = HAL_GetTick();
  State->fault_flags = g_motorAppFaultFlags;
  State->mode = g_motorAppMode;
  State->motor_count = count;
  State->reserved = (uint16_t)(((uint16_t)g_motorAppFailedMotorID << 8) |
                               (uint16_t)g_motorAppFailedRouteIndex);

  for (index = 0U; index < count; index++) {
    Motor_t *motor = MotorApp_GetRouteMotor(index);
    h7spi_motor_state_compact_t *out = &State->motors[index];
    unsigned char flags = 0U;
    unsigned char motor_id = (unsigned char)kMotorAppDefaultConfig[index].can_id;
    unsigned int age_ms = 0U;

    if (kMotorAppDefaultConfig[index].port == 0U) {
      continue;
    }

    out->motor_id = motor_id;
    out->state = motor->GetState();
    out->temperature = motor->GetRotorTemp();
    out->p_10mrad = MotorApp_SaturateI16((int32_t)(motor->GetPosition() * 100.0f));
    out->v_10mrad_s = MotorApp_SaturateI16((int32_t)(motor->GetSpeed() * 100.0f));
    out->torque_10mnm = MotorApp_SaturateI16((int32_t)(motor->GetTorque() * 100.0f));
    out->rx_ts_ms_lsb = (uint16_t)motor->GetLastRxTick();
    out->last_command_seq = g_motorAppMotorLastCommandSeq[motor_id];
    if (g_motorAppMotorCommandValid[motor_id] != 0U) {
      age_ms = HAL_GetTick() - g_motorAppMotorLastCommandTickMs[motor_id];
      out->command_age_ms = (age_ms > 0xFFFFU) ? 0xFFFFU : (uint16_t)age_ms;
    }
    if (motor->IsConfigured() != 0U) {
      flags |= H7SPI_MOTOR_FLAG_CONFIGURED;
    }
    if (motor->IsOnline(MOTOR_APP_FEEDBACK_TIMEOUT_MS) != 0U) {
      flags |= H7SPI_MOTOR_FLAG_ONLINE;
    }
    if (g_motorAppEnabled != 0U && motor->IsEnable() != 0U) {
      flags |= H7SPI_MOTOR_FLAG_ENABLED;
    }
    if (MotorApp_IsHardwareFaultState(motor->GetState()) != 0U) {
      flags |= H7SPI_MOTOR_FLAG_FAULT;
    }
    if (g_motorAppMotorCommandValid[motor_id] != 0U) {
      flags |= H7SPI_MOTOR_FLAG_CMD_VALID;
    }
    if (g_motorAppUpdatedByLastFrame[motor_id] != 0U) {
      flags |= H7SPI_MOTOR_FLAG_UPDATED_BY_LAST_FRAME;
    }
    out->flags = flags;
  }
}

/**********************************END OF FILE***********************************/
