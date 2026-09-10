#include "h7spi_protocol.h"

#include <stddef.h>
#include <string.h>

static uint16_t H7SPI_Crc16CcittUpdate(uint16_t crc, const uint8_t *data, uint16_t len)
{
  uint16_t i;
  uint8_t bit;

  for (i = 0U; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (bit = 0U; bit < 8U; bit++) {
      if ((crc & 0x8000U) != 0U) {
        crc = (uint16_t)((crc << 1) ^ 0x1021U);
      } else {
        crc = (uint16_t)(crc << 1);
      }
    }
  }
  return crc;
}

uint16_t H7SPI_Crc16Ccitt(const uint8_t *data, uint16_t len)
{
  return H7SPI_Crc16CcittUpdate(0xFFFFU, data, len);
}

uint16_t H7SPI_FrameCrc(const uint8_t *frame)
{
  const h7spi_header_t *header = (const h7spi_header_t *)frame;
  uint16_t payload_len = header->payload_len;
  uint16_t crc;
  uint8_t zero[2] = {0U, 0U};

  crc = H7SPI_Crc16Ccitt(frame, (uint16_t)offsetof(h7spi_header_t, crc16));
  crc = H7SPI_Crc16CcittUpdate(crc, zero, sizeof(zero));
  if (payload_len > 0U) {
    uint16_t i;
    const uint8_t *payload = frame + sizeof(h7spi_header_t);
    for (i = 0U; i < payload_len; i++) {
      uint8_t byte = payload[i];
      uint8_t bit;
      crc ^= (uint16_t)byte << 8;
      for (bit = 0U; bit < 8U; bit++) {
        if ((crc & 0x8000U) != 0U) {
          crc = (uint16_t)((crc << 1) ^ 0x1021U);
        } else {
          crc = (uint16_t)(crc << 1);
        }
      }
    }
  }
  return crc;
}

uint8_t H7SPI_CheckFrame(const uint8_t *frame)
{
  const h7spi_header_t *header = (const h7spi_header_t *)frame;

  if (header->magic != H7SPI_MAGIC) {
    return H7SPI_RESULT_BAD_MAGIC;
  }
  if (header->version != H7SPI_VERSION) {
    return H7SPI_RESULT_BAD_VERSION;
  }
  if ((uint32_t)sizeof(h7spi_header_t) + header->payload_len > H7SPI_TRANSFER_SIZE) {
    return H7SPI_RESULT_BAD_LENGTH;
  }
  if (H7SPI_FrameCrc(frame) != header->crc16) {
    return H7SPI_RESULT_BAD_CRC;
  }
  return H7SPI_RESULT_OK;
}

void H7SPI_BuildFrame(uint8_t *frame, uint8_t msg_type, uint16_t seq,
                      uint32_t tick_ms, const void *payload, uint16_t payload_len)
{
  h7spi_header_t *header = (h7spi_header_t *)frame;

  memset(frame, 0, H7SPI_TRANSFER_SIZE);
  header->magic = H7SPI_MAGIC;
  header->version = H7SPI_VERSION;
  header->msg_type = msg_type;
  header->seq = seq;
  header->payload_len = payload_len;
  header->tick_ms = tick_ms;
  header->flags = 0U;
  header->crc16 = 0U;

  if (payload != 0 && payload_len > 0U) {
    memcpy(frame + sizeof(h7spi_header_t), payload, payload_len);
  }
  header->crc16 = H7SPI_FrameCrc(frame);
}
