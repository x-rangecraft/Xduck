#ifndef DMUSB_MOTOR_BRIDGE_H
#define DMUSB_MOTOR_BRIDGE_H

#include "dmusb_protocol.h"
#include "h7spi_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/* RouteIds retains the USB slot order; zero means an unpopulated slot. */
uint8_t DMUSB_BuildMotorCommand(const dmusb_command_frame_t *input,
                                const uint8_t *route_ids, uint8_t route_count,
                                h7spi_motor_cmd_frame_t *output);

/* ENABLE_ALL can only enable the complete configured set. Do not silently
 * broaden a request naming just a subset of motors. Count zero names all. */
uint8_t DMUSB_ValidateEnableTargets(const dmusb_admin_op_frame_t *input,
                                   const uint8_t *route_ids, uint8_t route_count);

#ifdef __cplusplus
}
#endif
#endif
