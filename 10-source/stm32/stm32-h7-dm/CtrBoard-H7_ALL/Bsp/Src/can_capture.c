#include "can_capture.h"

#include "main.h"
#include <stddef.h>

#define CAN_CAPTURE_MAGIC        0x314e4143UL /* CAN1, little endian */
#define CAN_CAPTURE_VERSION      3UL
#define CAN_CAPTURE_EVENTS       18432UL
#define CAN_CAPTURE_COMMANDS     768UL
#define CAN_CAPTURE_DEFAULT_MS   3000UL
#define CAN_CAPTURE_DEFAULT_MASK 0x0842UL /* 2, 7, 12 */

typedef struct {
  uint32_t elapsed_cycles;
  uint16_t command_seq;
  uint8_t motor_id;
  uint8_t kind; /* 1: MIT TX FIFO accepted; 2: valid motor feedback RX */
  uint8_t data[8];
} CanCaptureEvent;

typedef struct {
  uint32_t elapsed_cycles;
  uint16_t command_seq;
  uint8_t motor_id;
  uint8_t reserved;
  int32_t p_mrad;
  int32_t v_mrad_s;
  int32_t torque_mnm;
  uint16_t kp_centi;
  uint16_t kd_milli;
} CanCaptureCommand;

typedef struct {
  uint32_t magic;
  uint32_t version;
  volatile uint32_t state; /* debugger writes 1 to arm; firmware sets 2/3 */
  volatile uint32_t event_count;
  volatile uint32_t event_dropped;
  volatile uint32_t command_count;
  volatile uint32_t command_dropped;
  uint32_t start_tick_ms;
  uint32_t start_cycle;
  uint32_t core_hz;
  volatile uint32_t window_ms;
  volatile uint32_t motor_mask; /* bit N-1 selects motor N */
  CanCaptureEvent events[CAN_CAPTURE_EVENTS];
  CanCaptureCommand commands[CAN_CAPTURE_COMMANDS];
} CanCaptureBuffer;

_Static_assert(sizeof(CanCaptureEvent) == 16U, "capture event must be 16 bytes");
_Static_assert(sizeof(CanCaptureCommand) == 24U, "capture command must be 24 bytes");
_Static_assert(offsetof(CanCaptureBuffer, events) == 48U, "capture header must be 48 bytes");
_Static_assert(offsetof(CanCaptureBuffer, commands) == 48U + CAN_CAPTURE_EVENTS * 16U,
               "capture command offset mismatch");
_Static_assert(sizeof(CanCaptureBuffer) <= 320U * 1024U, "capture exceeds AXI SRAM");

__attribute__((used, section(".RAM_D1")))
CanCaptureBuffer g_can_capture;
static uint16_t g_last_tx_seq[15];

static uint8_t CanCapture_Selected(uint8_t motor_id)
{
  return (motor_id >= 1U && motor_id <= 14U &&
          (g_can_capture.motor_mask & (1UL << (motor_id - 1U))) != 0U) ? 1U : 0U;
}

static uint8_t CanCapture_Active(uint32_t cycle)
{
  uint8_t index;
  if (g_can_capture.state == 1U) {
    g_can_capture.event_count = 0U;
    g_can_capture.event_dropped = 0U;
    g_can_capture.command_count = 0U;
    g_can_capture.command_dropped = 0U;
    for (index = 0U; index < 15U; index++) g_last_tx_seq[index] = 0U;
    g_can_capture.start_tick_ms = HAL_GetTick();
    g_can_capture.start_cycle = cycle;
    g_can_capture.core_hz = SystemCoreClock;
    g_can_capture.state = 2U;
  }
  if (g_can_capture.state != 2U) return 0U;
  if ((uint32_t)(HAL_GetTick() - g_can_capture.start_tick_ms) >= g_can_capture.window_ms) {
    __DMB();
    g_can_capture.state = 3U;
    return 0U;
  }
  return 1U;
}

void CanCapture_Init(void)
{
  g_can_capture.magic = CAN_CAPTURE_MAGIC;
  g_can_capture.version = CAN_CAPTURE_VERSION;
  g_can_capture.state = 0U;
  g_can_capture.event_count = 0U;
  g_can_capture.event_dropped = 0U;
  g_can_capture.command_count = 0U;
  g_can_capture.command_dropped = 0U;
  g_can_capture.window_ms = CAN_CAPTURE_DEFAULT_MS;
  g_can_capture.motor_mask = CAN_CAPTURE_DEFAULT_MASK;
  g_can_capture.core_hz = SystemCoreClock;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void CanCapture_Poll(void)
{
  uint32_t primask;
  if (g_can_capture.state != 2U) return;
  primask = __get_PRIMASK();
  __disable_irq();
  if (g_can_capture.state == 2U &&
      (uint32_t)(HAL_GetTick() - g_can_capture.start_tick_ms) >= g_can_capture.window_ms) {
    __DMB();
    g_can_capture.state = 3U;
  }
  __set_PRIMASK(primask);
}

void CanCapture_CommandReceived(uint8_t motor_id, uint16_t command_seq,
                                int32_t p_mrad, int32_t v_mrad_s,
                                uint16_t kp_centi, uint16_t kd_milli,
                                int32_t torque_mnm, uint32_t received_cycle)
{
  uint32_t primask;
  CanCaptureCommand *record;
  if (CanCapture_Selected(motor_id) == 0U ||
      (g_can_capture.state != 1U && g_can_capture.state != 2U)) return;

  primask = __get_PRIMASK();
  __disable_irq();
  if (CanCapture_Active(received_cycle) != 0U) {
    if (g_can_capture.command_count < CAN_CAPTURE_COMMANDS) {
      record = &g_can_capture.commands[g_can_capture.command_count];
      record->elapsed_cycles = received_cycle - g_can_capture.start_cycle;
      record->command_seq = command_seq;
      record->motor_id = motor_id;
      record->reserved = 0U;
      record->p_mrad = p_mrad;
      record->v_mrad_s = v_mrad_s;
      record->torque_mnm = torque_mnm;
      record->kp_centi = kp_centi;
      record->kd_milli = kd_milli;
      __DMB();
      g_can_capture.command_count++;
    } else {
      g_can_capture.command_dropped++;
    }
  }
  __set_PRIMASK(primask);
}

static void CanCapture_Event(uint8_t motor_id, uint16_t command_seq,
                             uint8_t kind, uint32_t cycle, const uint8_t data[8])
{
  uint32_t primask;
  CanCaptureEvent *record;
  uint8_t i;
  if (CanCapture_Selected(motor_id) == 0U ||
      (g_can_capture.state != 1U && g_can_capture.state != 2U)) return;

  primask = __get_PRIMASK();
  __disable_irq();
  if (CanCapture_Active(cycle) != 0U) {
    if (kind == 1U) g_last_tx_seq[motor_id] = command_seq;
    if (g_can_capture.event_count < CAN_CAPTURE_EVENTS) {
      record = &g_can_capture.events[g_can_capture.event_count];
      record->elapsed_cycles = cycle - g_can_capture.start_cycle;
      record->command_seq = command_seq;
      record->motor_id = motor_id;
      record->kind = kind;
      for (i = 0U; i < 8U; i++) record->data[i] = data[i];
      __DMB();
      g_can_capture.event_count++;
    } else {
      g_can_capture.event_dropped++;
    }
  }
  __set_PRIMASK(primask);
}

void CanCapture_Tx(uint8_t motor_id, uint16_t command_seq, const uint8_t data[8])
{
  /* The caller invokes this only after a normal MIT frame enters the TX FIFO. */
  if (motor_id < 1U || motor_id > 14U) return;
  CanCapture_Event(motor_id, command_seq, 1U, DWT->CYCCNT, data);
}

void CanCapture_Rx(uint8_t motor_id, uint32_t received_cycle, const uint8_t data[8])
{
  if (motor_id < 1U || motor_id > 14U) return;
  CanCapture_Event(motor_id, g_last_tx_seq[motor_id], 2U, received_cycle, data);
}
