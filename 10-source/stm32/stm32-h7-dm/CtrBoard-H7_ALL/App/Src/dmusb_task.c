#include "dmusb_task.h"

#include "FreeRTOS.h"
#include "cmsis_os.h"
#include "dmusb_protocol.h"
#include "dmusb_motor_bridge.h"
#include "battery_meter.h"
#include "h7spi_protocol.h"
#include "imu_task.h"
#include "motor_app.h"
#if H7DM_CAN_CAPTURE
#include "can_capture.h"
#endif
#include "task.h"
#include "usbd_cdc_if.h"

#include <stddef.h>
#include <string.h>

#define DMUSB_RX_SIZE         1600U
#define DMUSB_TX_SIZE         1600U
#define DMUSB_STATE_PERIOD_MS 20U
#define DMUSB_TX_STUCK_PERIODS 10U

static uint8_t g_rx[DMUSB_RX_SIZE];
static volatile uint16_t g_rx_len;
static uint8_t g_work[DMUSB_RX_SIZE];
static uint8_t g_tx[DMUSB_TX_SIZE];
static uint16_t g_tx_seq;
static osThreadId g_task;

/* DAP-readable USB CDC diagnostics. These deliberately remain non-static and
 * volatile so a debugger can inspect the live send path without stopping USB. */
volatile uint32_t publish_count;
volatile uint32_t tx_ready_false_count;
volatile uint32_t tx_ok_count;
volatile uint32_t tx_busy_count;
volatile uint32_t tx_fail_count;
volatile uint32_t dmusb_task_loop_count;
volatile uint32_t usb_tx_state;
volatile uint32_t usb_tx_recovery_count;
volatile uint8_t usb_dev_state;
volatile uint8_t first_tx_result = 0xFFU;
volatile uint8_t last_tx_result = 0xFFU;

static float g_gyro_history[2][3];
static float g_gravity_history[2][3];
static uint8_t g_imu_median_initialized;

static float median3(float a, float b, float c)
{
  if (a > b) { float swap = a; a = b; b = swap; }
  if (b > c) { float swap = b; b = c; c = swap; }
  if (a > b) b = a;
  return b;
}

static void median_filter_imu(const float gyro[3], const float gravity[3],
                              float gyro_out[3], float gravity_out[3])
{
  uint8_t axis;
  if (g_imu_median_initialized == 0U) {
    for (axis = 0U; axis < 3U; axis++) {
      g_gyro_history[0][axis] = gyro[axis];
      g_gyro_history[1][axis] = gyro[axis];
      g_gravity_history[0][axis] = gravity[axis];
      g_gravity_history[1][axis] = gravity[axis];
      gyro_out[axis] = gyro[axis];
      gravity_out[axis] = gravity[axis];
    }
    g_imu_median_initialized = 1U;
    return;
  }
  for (axis = 0U; axis < 3U; axis++) {
    gyro_out[axis] = median3(g_gyro_history[0][axis],
                             g_gyro_history[1][axis], gyro[axis]);
    gravity_out[axis] = median3(g_gravity_history[0][axis],
                                g_gravity_history[1][axis], gravity[axis]);
    g_gyro_history[0][axis] = g_gyro_history[1][axis];
    g_gyro_history[1][axis] = gyro[axis];
    g_gravity_history[0][axis] = g_gravity_history[1][axis];
    g_gravity_history[1][axis] = gravity[axis];
  }
}

static uint16_t get_u16(const uint8_t *p)
{
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

void DMUSB_OnReceive(const uint8_t *data, uint16_t len)
{
  UBaseType_t mask;
  BaseType_t wake = pdFALSE;
  if (data == 0 || len == 0U) return;
  mask = taskENTER_CRITICAL_FROM_ISR();
  if ((uint32_t)g_rx_len + len <= sizeof(g_rx)) {
    memcpy(g_rx + g_rx_len, data, len);
    g_rx_len = (uint16_t)(g_rx_len + len);
  } else {
    g_rx_len = 0U;
  }
  taskEXIT_CRITICAL_FROM_ISR(mask);
  if (g_task != 0) {
    vTaskNotifyGiveFromISR((TaskHandle_t)g_task, &wake);
    portYIELD_FROM_ISR(wake);
  }
}

static uint16_t take_rx(void)
{
  uint16_t len;
  taskENTER_CRITICAL();
  len = g_rx_len;
  if (len != 0U) {
    memcpy(g_work, g_rx, len);
    g_rx_len = 0U;
  }
  taskEXIT_CRITICAL();
  return len;
}

static void restore_suffix(const uint8_t *data, uint16_t len)
{
  taskENTER_CRITICAL();
  if ((uint32_t)len + g_rx_len <= sizeof(g_rx)) {
    memmove(g_rx + len, g_rx, g_rx_len);
    memcpy(g_rx, data, len);
    g_rx_len = (uint16_t)(g_rx_len + len);
  } else {
    g_rx_len = 0U;
  }
  taskEXIT_CRITICAL();
}

static void send_admin_result(uint16_t request_seq, uint8_t op, uint8_t result)
{
  dmusb_admin_result_frame_t out;
  MotorApp_Status_t status;
  uint16_t frame_len;
  uint8_t retry;
  memset(&out, 0, sizeof(out));
  MotorApp_GetStatus(&status);
  out.response_seq = request_seq;
  out.op = op;
  out.result = result;
  out.stm32_tick_ms = HAL_GetTick();
  out.fault_flags = (uint32_t)status.fault_flags;
  out.requested_mask = (uint32_t)status.requested_mask;
  out.succeeded_mask = (uint32_t)status.succeeded_mask;
  out.busy = status.busy;
  out.requested_count = status.requested_count;
  out.success_count = status.success_count;
  out.failed_motor_id = status.failed_motor_id;
  out.failed_route_index = status.failed_route_index;
  for (retry = 0U; retry < 5U && USB_CDC_TxReady() == 0U; retry++) {
    osDelay(1U);
  }
  if (USB_CDC_TxReady() != 0U &&
      DMUSB_Pack(DMUSB_MSG_ADMIN_RESULT, ++g_tx_seq, HAL_GetTick() * 1000U,
                 out.fault_flags, &out, sizeof(out), g_tx, sizeof(g_tx),
                 &frame_len) != 0U) {
    (void)CDC_Transmit_HS(g_tx, frame_len);
  }
}

static void handle_frame(const dmusb_header_t *header, const uint8_t *payload)
{
#if H7DM_CAN_CAPTURE
  uint32_t received_cycle = DWT->CYCCNT;
#endif
  uint8_t route_ids[DMUSB_MAX_MOTORS];
  uint8_t route_count = MotorApp_GetMotorCount();
  uint8_t slot;
  if (route_count > DMUSB_MAX_MOTORS) return;
  for (slot = 0U; slot < route_count; slot++)
    route_ids[slot] = MotorApp_GetMotorIdByIndex(slot);
  if (header->msg_type == DMUSB_MSG_COMMAND_FRAME) {
    dmusb_command_frame_t in;
    h7spi_motor_cmd_frame_t command;
    uint16_t base = (uint16_t)offsetof(dmusb_command_frame_t, motors);
    uint16_t expected;
    if (header->payload_len < base) return;
    memset(&in, 0, sizeof(in));
    memcpy(&in, payload, base);
    if (in.mode != DMUSB_MODE_MIT || in.motor_count != MotorApp_GetMotorCount()) return;
    expected = (uint16_t)(base + in.motor_count * sizeof(in.motors[0]));
    if (header->payload_len != expected) return;
    memcpy(in.motors, payload + base, in.motor_count * sizeof(in.motors[0]));
    if (DMUSB_BuildMotorCommand(&in, route_ids, route_count, &command) != H7SPI_RESULT_OK)
      return;
    if (MotorApp_SubmitSpiMotorCommandAsync(&command) != H7SPI_RESULT_OK) return;
#if H7DM_CAN_CAPTURE
    for (slot = 0U; slot < command.motor_count; slot++) {
      const h7spi_motor_cmd_t *cmd = &command.motors[slot];
      CanCapture_CommandReceived(cmd->motor_id, command.command_seq,
                                 cmd->p_mrad, cmd->v_mrad_s,
                                 cmd->kp_centi, cmd->kd_milli,
                                 cmd->torque_mnm, received_cycle);
    }
#endif
  } else if (header->msg_type == DMUSB_MSG_SET_LIMITS) {
    dmusb_position_limit_t limits[DMUSB_MAX_MOTORS];
    uint8_t count;
    uint8_t result;
    if (header->payload_len < 4U) return;
    count = payload[2];
    if (get_u16(payload) != header->seq || count > DMUSB_MAX_MOTORS ||
        header->payload_len != 4U + count * sizeof(limits[0])) {
      send_admin_result(header->seq, DMUSB_ADMIN_SET_LIMITS, H7SPI_RESULT_BAD_LENGTH);
      return;
    }
    memcpy(limits, payload + 4U, count * sizeof(limits[0]));
    result = MotorApp_SetPositionLimits(limits, count);
    send_admin_result(header->seq, DMUSB_ADMIN_SET_LIMITS, result);
  } else if (header->msg_type == DMUSB_MSG_ADMIN_OP &&
             header->payload_len >= sizeof(dmusb_admin_op_frame_t)) {
    dmusb_admin_op_frame_t in;
    h7spi_admin_op_frame_t op;
    uint8_t result;
    memcpy(&in, payload, sizeof(in));
    memset(&op, 0, sizeof(op));
    op.motor_count = in.motor_count;
    memcpy(op.motor_codes, in.motor_ids, sizeof(op.motor_codes));
    if (in.op == DMUSB_ADMIN_ENABLE) {
      result = DMUSB_ValidateEnableTargets(&in, route_ids, route_count);
      if (result != H7SPI_RESULT_OK) {
        send_admin_result(header->seq, in.op, result);
        return;
      }
      op.op = H7SPI_ADMIN_OP_ENABLE_ALL;
      op.motor_count = 0U; /* Complete configured set was validated above. */
    }
    else if (in.op == DMUSB_ADMIN_DISABLE) op.op = H7SPI_ADMIN_OP_DISABLE_ALL;
    else if (in.op == DMUSB_ADMIN_MARK_ZERO) op.op = H7SPI_ADMIN_OP_MARK_ZERO;
    else { send_admin_result(header->seq, in.op, H7SPI_RESULT_BAD_TYPE); return; }
    result = MotorApp_ApplySpiAdminOp(&op);
    send_admin_result(header->seq, in.op, result);
  }
}

static void process_rx(uint16_t len)
{
  uint16_t offset = 0U;
  while ((uint16_t)(len - offset) >= DMUSB_HEADER_SIZE) {
    dmusb_header_t header;
    const uint8_t *payload;
    uint16_t total;
    if (g_work[offset] != (uint8_t)DMUSB_MAGIC ||
        g_work[offset + 1U] != (uint8_t)(DMUSB_MAGIC >> 8) ||
        g_work[offset + 2U] != DMUSB_VERSION) {
      offset++;
      continue;
    }
    total = (uint16_t)(DMUSB_HEADER_SIZE + get_u16(g_work + offset + 6U));
    if (total > DMUSB_RX_SIZE) { offset++; continue; }
    if (total > (uint16_t)(len - offset)) {
      restore_suffix(g_work + offset, (uint16_t)(len - offset));
      return;
    }
    if (DMUSB_Validate(g_work + offset, total, &header, &payload) != 0U) {
      handle_frame(&header, payload);
    }
    offset = (uint16_t)(offset + total);
  }
  if (offset < len) restore_suffix(g_work + offset, (uint16_t)(len - offset));
}

static void publish_state(void)
{
  static uint8_t tx_not_ready_periods;
  h7spi_motor_state_frame_t source;
  dmusb_state_frame_t out;
  imu_snapshot_t imu;
  uint16_t payload_len;
  uint16_t frame_len;
  uint8_t index;
  uint8_t tx_result;
  battery_meter_sample_t meter;
  dmusb_battery_meter_t meter_wire;
  float filtered_gyro[3];
  float filtered_gravity[3];
  publish_count++;
  USB_CDC_GetTxState((uint8_t *)&usb_dev_state, (uint32_t *)&usb_tx_state);
  if (USB_CDC_TxReady() == 0U) {
    tx_ready_false_count++;
    if (tx_not_ready_periods < 0xFFU) {
      tx_not_ready_periods++;
    }
    if (tx_not_ready_periods >= DMUSB_TX_STUCK_PERIODS) {
      if (USB_CDC_RecoverTx() == USBD_OK) {
        usb_tx_recovery_count++;
      }
      tx_not_ready_periods = 0U;
    }
    return;
  }
  tx_not_ready_periods = 0U;
  memset(&source, 0, sizeof(source));
  memset(&out, 0, sizeof(out));
  MotorApp_FillSpiState(&source, MotorApp_GetLastCommandSeq());
  out.state_seq = source.state_seq;
  out.ack_command_seq = source.ack_command_seq;
  out.stm32_tick_ms = source.stm32_tick_ms;
  out.fault_flags = source.fault_flags;
  out.mode = DMUSB_MODE_MIT;
  out.reserved = DMUSB_CAP_SPARSE_COMMAND | DMUSB_CAP_ENABLE_TARGETS |
                 DMUSB_CAP_CALIBRATION | DMUSB_CAP_FAULT_DIAGNOSTICS |
                 DMUSB_CAP_BATTERY_METER;
  if (MotorApp_PositionLimitsReady() != 0U) out.reserved |= DMUSB_STATE_LIMITS_READY;
  out.motor_count = source.motor_count;
  (void)IMU_GetSnapshot(&imu);
  median_filter_imu(imu.base_ang_vel, imu.projected_gravity,
                    filtered_gyro, filtered_gravity);
  memcpy(out.imu.base_ang_vel, filtered_gyro, sizeof(out.imu.base_ang_vel));
  memcpy(out.imu.projected_gravity, filtered_gravity,
         sizeof(out.imu.projected_gravity));
  memcpy(out.imu.quaternion, imu.quaternion, sizeof(out.imu.quaternion));
  out.imu.sample_sequence = imu.sample_sequence;
  out.imu.flags = imu.flags;
  out.imu.reserved[0] = MotorApp_GetLastFaultMotorID();
  out.imu.reserved[1] = MotorApp_GetLastFaultRouteIndex();
  for (index = 0U; index < source.motor_count; index++) {
    out.motors[index].p_mrad = source.motors[index].p_mrad;
    out.motors[index].v_mrad_s = source.motors[index].v_mrad_s;
    out.motors[index].torque_mnm = source.motors[index].torque_mnm;
    out.motors[index].rx_ts_ms = source.motors[index].rx_ts_ms;
    out.motors[index].flags = source.motors[index].flags;
    out.motors[index].rotor_temperature_c = source.motors[index].temperature;
  }
  payload_len = (uint16_t)(offsetof(dmusb_state_frame_t, motors) +
                           out.motor_count * sizeof(out.motors[0]));
  memset(&meter_wire, 0, sizeof(meter_wire));
  if (BatteryMeter_GetSnapshot(&meter) != 0U) {
    meter_wire.voltage_centi_v = meter.voltage_centi_v;
    meter_wire.percent = (uint8_t)meter.percent;
    meter_wire.alarm = (uint8_t)meter.alarm;
    meter_wire.valid = 1U;
  }
  /* Pack into the existing TX buffer, then append the suffix and refresh CRC.
   * The state struct contains storage for 24 routes, so append after the
   * actual motor count rather than after sizeof(out). */
  {
    uint8_t payload[sizeof(out) + sizeof(meter_wire)];
    memcpy(payload, &out, payload_len);
    memcpy(payload + payload_len, &meter_wire, sizeof(meter_wire));
    payload_len = (uint16_t)(payload_len + sizeof(meter_wire));
    if (DMUSB_Pack(DMUSB_MSG_STATE_FRAME, ++g_tx_seq, HAL_GetTick() * 1000U,
                   out.fault_flags, payload, payload_len, g_tx, sizeof(g_tx),
                   &frame_len) == 0U) return;
  }
  {
    tx_result = CDC_Transmit_HS(g_tx, frame_len);
    if (first_tx_result == 0xFFU) {
      first_tx_result = tx_result;
    }
    last_tx_result = tx_result;
    if (tx_result == USBD_OK) {
      tx_ok_count++;
    } else if (tx_result == USBD_BUSY) {
      tx_busy_count++;
    } else {
      tx_fail_count++;
    }
    USB_CDC_GetTxState((uint8_t *)&usb_dev_state, (uint32_t *)&usb_tx_state);
  }
}

static void StartDMUSBTask(void const *argument)
{
  TickType_t last_publish = xTaskGetTickCount();
  (void)argument;
  g_task = osThreadGetId();
  for (;;) {
    dmusb_task_loop_count++;
    uint16_t len = take_rx();
    if (len != 0U) process_rx(len);
    if ((xTaskGetTickCount() - last_publish) >= pdMS_TO_TICKS(DMUSB_STATE_PERIOD_MS)) {
      last_publish += pdMS_TO_TICKS(DMUSB_STATE_PERIOD_MS);
      publish_state();
    }
    (void)ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1U));
  }
}

void DMUSB_TaskInit(void)
{
  osThreadDef(dmusbTask, StartDMUSBTask, osPriorityAboveNormal, 0, 1024);
  g_task = osThreadCreate(osThread(dmusbTask), 0);
}
