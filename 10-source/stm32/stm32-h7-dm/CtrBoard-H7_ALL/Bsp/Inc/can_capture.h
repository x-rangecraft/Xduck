#ifndef CAN_CAPTURE_H
#define CAN_CAPTURE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Optional RAM trace. All timestamps use the same STM32 DWT cycle counter.
 * TX means accepted into the FDCAN FIFO, not completed on the wire. */
void CanCapture_Init(void);
void CanCapture_Poll(void);
void CanCapture_CommandReceived(uint8_t motor_id, uint16_t command_seq,
                                int32_t p_mrad, int32_t v_mrad_s,
                                uint16_t kp_centi, uint16_t kd_milli,
                                int32_t torque_mnm, uint32_t received_cycle);
void CanCapture_Tx(uint8_t motor_id, uint16_t command_seq, const uint8_t data[8]);
void CanCapture_Rx(uint8_t motor_id, uint32_t received_cycle, const uint8_t data[8]);

#ifdef __cplusplus
}
#endif

#endif
