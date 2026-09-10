#include "dmusb_motor_bridge.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
  const uint8_t routes[14] = {2, 0, 0, 0, 0, 1, 0, 0, 0, 13, 14, 15, 0, 0};
  const uint8_t slots[5] = {0, 5, 9, 10, 11};
  dmusb_command_frame_t input = {0};
  h7spi_motor_cmd_frame_t output;
  dmusb_admin_op_frame_t admin = {0};
  uint8_t i;
  input.mode = DMUSB_MODE_MIT;
  input.motor_count = 14;
  input.command_seq = 17;
  input.host_tick_us = 987654;
  for (i = 0; i < 14; i++) {
    input.motors[i].p_mrad = 100 + i;
    input.motors[i].v_mrad_s = -200 - i;
    input.motors[i].torque_mnm = 300 + i;
    input.motors[i].kp_centi = 1234;
    input.motors[i].kd_milli = 4321;
    input.motors[i].flags = 1;
  }
  assert(DMUSB_BuildMotorCommand(&input, routes, 14, &output) == H7SPI_RESULT_OK);
  assert(output.motor_count == 5 && output.command_seq == 17 && output.host_tick_ms == 987);
  for (i = 0; i < 5; i++) {
    assert(output.motors[i].motor_id == routes[slots[i]]);
    assert(output.motors[i].p_mrad == input.motors[slots[i]].p_mrad);
    assert(output.motors[i].v_mrad_s == input.motors[slots[i]].v_mrad_s);
    assert(output.motors[i].torque_mnm == input.motors[slots[i]].torque_mnm);
    assert(output.motors[i].kp_centi == 1234 && output.motors[i].kd_milli == 4321);
  }
  /* Empty slots can have arbitrary contents and never become commands. */
  input.motors[1].flags = 0;
  assert(DMUSB_BuildMotorCommand(&input, routes, 14, &output) == H7SPI_RESULT_OK);
  input.motors[5].flags = 0;
  assert(DMUSB_BuildMotorCommand(&input, routes, 14, &output) != H7SPI_RESULT_OK);
  input.motors[5].flags = 1;
  input.motor_count = 6;
  assert(DMUSB_BuildMotorCommand(&input, routes, 14, &output) == H7SPI_RESULT_BAD_MOTOR_COUNT);
  input.motor_count = 14;
  input.mode = 99;
  assert(DMUSB_BuildMotorCommand(&input, routes, 14, &output) == H7SPI_RESULT_BAD_MODE);
  input.mode = DMUSB_MODE_MIT;
  const uint8_t empty[14] = {0};
  assert(DMUSB_BuildMotorCommand(&input, empty, 14, &output) == H7SPI_RESULT_BAD_MOTOR_COUNT);

  admin.motor_count = 5;
  for (i = 0; i < 5; i++) admin.motor_ids[i] = routes[slots[4-i]];
  assert(DMUSB_ValidateEnableTargets(&admin, routes, 14) == H7SPI_RESULT_OK);
  admin.motor_count = 4;
  assert(DMUSB_ValidateEnableTargets(&admin, routes, 14) == H7SPI_RESULT_BAD_MOTOR_COUNT);
  admin.motor_count = 5;
  admin.motor_ids[4] = admin.motor_ids[0];
  assert(DMUSB_ValidateEnableTargets(&admin, routes, 14) == H7SPI_RESULT_DUPLICATE_MOTOR_ID);
  admin.motor_ids[4] = 0;
  assert(DMUSB_ValidateEnableTargets(&admin, routes, 14) == H7SPI_RESULT_UNKNOWN_MOTOR_ID);
  admin.motor_ids[4] = 99;
  assert(DMUSB_ValidateEnableTargets(&admin, routes, 14) == H7SPI_RESULT_UNKNOWN_MOTOR_ID);
  admin.motor_count = 0;
  assert(DMUSB_ValidateEnableTargets(&admin, routes, 14) == H7SPI_RESULT_OK);
  assert(DMUSB_ValidateEnableTargets(&admin, empty, 14) == H7SPI_RESULT_BAD_MOTOR_COUNT);
  puts("dmusb_motor_bridge: sparse slots, exact mapping, invalid frames and enable scopes passed");
  return 0;
}
