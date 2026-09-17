#include "dmusb_protocol.h"
#include <stddef.h>
#include <string.h>

_Static_assert(sizeof(dmusb_motor_command_t) == 20U, "166 command ABI changed");
_Static_assert(sizeof(dmusb_motor_observation_t) == 20U, "166 state ABI changed");
_Static_assert(sizeof(dmusb_battery_meter_t) == 8U, "battery suffix ABI changed");
_Static_assert(offsetof(dmusb_motor_observation_t, rotor_temperature_c) == 17U,
               "motor temperature wire offset changed");
_Static_assert(sizeof(dmusb_admin_op_frame_t) == 28U, "166 admin ABI changed");
_Static_assert(offsetof(dmusb_command_frame_t, motors) == 8U, "166 command prefix changed");
_Static_assert(sizeof(dmusb_imu_observation_t) == 48U, "IMU wire ABI changed");
_Static_assert(offsetof(dmusb_imu_observation_t, quaternion) == 24U,
               "IMU quaternion wire offset changed");
_Static_assert(offsetof(dmusb_imu_observation_t, sample_sequence) == 40U,
               "IMU sequence wire offset changed");
_Static_assert(offsetof(dmusb_state_frame_t, motors) == 64U, "state prefix changed");

static uint16_t get_u16(const uint8_t *p) { return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }
static uint32_t get_u32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void put_u16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put_u32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

static uint16_t crc_update(uint16_t crc, const uint8_t *data, uint32_t len)
{
  uint32_t i;
  uint8_t bit;
  for (i = 0U; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (bit = 0U; bit < 8U; bit++) {
      crc = (crc & 0x8000U) ? (uint16_t)((crc << 1) ^ 0x1021U) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

uint16_t DMUSB_Crc16(const uint8_t *data, uint32_t len) { return crc_update(0xFFFFU, data, len); }

static uint16_t frame_crc(const uint8_t *frame, uint16_t len)
{
  const uint8_t zero[2] = {0U, 0U};
  uint16_t crc = crc_update(0xFFFFU, frame, 16U);
  crc = crc_update(crc, zero, 2U);
  if (len > DMUSB_HEADER_SIZE) crc = crc_update(crc, frame + DMUSB_HEADER_SIZE, len - DMUSB_HEADER_SIZE);
  return crc;
}

uint8_t DMUSB_Validate(const uint8_t *frame, uint16_t frame_len, dmusb_header_t *header, const uint8_t **payload)
{
  uint16_t payload_len;
  if (frame == 0 || header == 0 || payload == 0 || frame_len < DMUSB_HEADER_SIZE) return 0U;
  payload_len = get_u16(frame + 6U);
  if (get_u16(frame) != DMUSB_MAGIC || frame[2] != DMUSB_VERSION || payload_len > DMUSB_MAX_PAYLOAD ||
      frame_len != (uint16_t)(DMUSB_HEADER_SIZE + payload_len) || get_u16(frame + 16U) != frame_crc(frame, frame_len)) return 0U;
  header->msg_type = frame[3]; header->seq = get_u16(frame + 4U); header->payload_len = payload_len;
  header->tick_us = get_u32(frame + 8U); header->flags = get_u32(frame + 12U); *payload = frame + DMUSB_HEADER_SIZE;
  return 1U;
}

uint8_t DMUSB_Pack(uint8_t msg_type, uint16_t seq, uint32_t tick_us, uint32_t flags,
                   const void *payload, uint16_t payload_len, uint8_t *out, uint16_t capacity, uint16_t *out_len)
{
  uint16_t total = (uint16_t)(DMUSB_HEADER_SIZE + payload_len);
  if (out == 0 || out_len == 0 || payload_len > DMUSB_MAX_PAYLOAD || total > capacity || (payload_len != 0U && payload == 0)) return 0U;
  put_u16(out, DMUSB_MAGIC); out[2] = DMUSB_VERSION; out[3] = msg_type; put_u16(out + 4U, seq); put_u16(out + 6U, payload_len);
  put_u32(out + 8U, tick_us); put_u32(out + 12U, flags); put_u16(out + 16U, 0U);
  if (payload_len != 0U) memcpy(out + DMUSB_HEADER_SIZE, payload, payload_len);
  put_u16(out + 16U, DMUSB_Crc16(out, total)); *out_len = total; return 1U;
}
