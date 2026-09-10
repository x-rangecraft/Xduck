#include "dmusb_motor_bridge.h"
#include <string.h>

uint8_t DMUSB_BuildMotorCommand(const dmusb_command_frame_t *input,
                                const uint8_t *route_ids, uint8_t route_count,
                                h7spi_motor_cmd_frame_t *output)
{
  uint8_t slot;
  if (input == 0 || route_ids == 0 || output == 0) return H7SPI_RESULT_BAD_LENGTH;
  memset(output, 0, sizeof(*output));
  if (route_count > DMUSB_MAX_MOTORS || route_count > H7SPI_MAX_MOTORS ||
      input->motor_count != route_count) return H7SPI_RESULT_BAD_MOTOR_COUNT;
  if (input->mode != DMUSB_MODE_MIT) return H7SPI_RESULT_BAD_MODE;
  output->command_seq = input->command_seq;
  output->mode = H7SPI_MODE_CONTROL_MIT;
  output->host_tick_ms = input->host_tick_us / 1000U;
  for (slot = 0U; slot < route_count; slot++) {
    h7spi_motor_cmd_t *target;
    const dmusb_motor_command_t *source = &input->motors[slot];
    if (route_ids[slot] == 0U) continue;
    /* A populated slot must carry a valid command; reject the whole frame
     * rather than refreshing the watchdog with only part of the robot. */
    if ((source->flags & 1U) == 0U) return H7SPI_RESULT_BAD_TYPE;
    target = &output->motors[output->motor_count++];
    target->motor_id = route_ids[slot];
    target->flags = (uint8_t)source->flags;
    target->p_mrad = source->p_mrad;
    target->v_mrad_s = source->v_mrad_s;
    target->torque_mnm = source->torque_mnm;
    target->kp_centi = source->kp_centi;
    target->kd_milli = source->kd_milli;
  }
  return output->motor_count != 0U ? H7SPI_RESULT_OK : H7SPI_RESULT_BAD_MOTOR_COUNT;
}

uint8_t DMUSB_ValidateEnableTargets(const dmusb_admin_op_frame_t *input,
                                   const uint8_t *route_ids, uint8_t route_count)
{
  uint8_t index, slot, configured = 0U;
  uint32_t seen = 0U;
  if (input == 0 || route_ids == 0) return H7SPI_RESULT_BAD_LENGTH;
  if (route_count > DMUSB_MAX_MOTORS || input->motor_count > DMUSB_MAX_MOTORS)
    return H7SPI_RESULT_BAD_MOTOR_COUNT;
  for (slot = 0U; slot < route_count; slot++) if (route_ids[slot] != 0U) configured++;
  if (configured == 0U) return H7SPI_RESULT_BAD_MOTOR_COUNT;
  if (input->motor_count == 0U) return H7SPI_RESULT_OK;
  for (index = 0U; index < input->motor_count; index++) {
    uint8_t id = input->motor_ids[index];
    for (slot = 0U; slot < route_count; slot++) {
      if (id != 0U && route_ids[slot] == id) break;
    }
    if (slot == route_count) return H7SPI_RESULT_UNKNOWN_MOTOR_ID;
    if ((seen & (1UL << slot)) != 0U) return H7SPI_RESULT_DUPLICATE_MOTOR_ID;
    seen |= 1UL << slot;
  }
  return input->motor_count == configured ? H7SPI_RESULT_OK : H7SPI_RESULT_BAD_MOTOR_COUNT;
}
