#include "bmi088.h"

#include "main.h"
#include "spi.h"

#define ACC_CHIP_ID       0x00U
#define ACC_CHIP_ID_VALUE 0x1EU
#define ACC_DATA           0x12U
#define ACC_TEMP           0x22U
#define ACC_CONF           0x40U
#define ACC_RANGE          0x41U
#define ACC_PWR_CONF       0x7CU
#define ACC_PWR_CTRL       0x7DU
#define ACC_SOFTRESET      0x7EU

#define GYRO_CHIP_ID       0x00U
#define GYRO_CHIP_ID_VALUE 0x0FU
#define GYRO_DATA          0x02U
#define GYRO_RANGE         0x0FU
#define GYRO_BANDWIDTH     0x10U
#define GYRO_LPM1          0x11U
#define GYRO_SOFTRESET     0x14U
#define GYRO_CTRL          0x15U

#define ACC_M_S2_PER_LSB   0.0008974358974f /* +/-3 g */
#define GYRO_RAD_S_PER_LSB 0.0002663161090f /* +/-500 deg/s */

static uint8_t spi_byte(uint8_t tx, uint8_t *rx)
{
  return HAL_SPI_TransmitReceive(&hspi2, &tx, rx, 1U, 10U) == HAL_OK;
}

static uint8_t write_reg(GPIO_TypeDef *port, uint16_t pin, uint8_t reg, uint8_t value)
{
  uint8_t ignored;
  uint8_t ok;
  HAL_GPIO_WritePin(port, pin, GPIO_PIN_RESET);
  ok = spi_byte(reg, &ignored) && spi_byte(value, &ignored);
  HAL_GPIO_WritePin(port, pin, GPIO_PIN_SET);
  return ok;
}

static uint8_t read_regs(GPIO_TypeDef *port, uint16_t pin, uint8_t reg,
                         uint8_t accel_dummy, uint8_t *data, uint8_t len)
{
  uint8_t ignored;
  uint8_t index;
  uint8_t ok;
  HAL_GPIO_WritePin(port, pin, GPIO_PIN_RESET);
  ok = spi_byte((uint8_t)(reg | 0x80U), &ignored);
  if (ok != 0U && accel_dummy != 0U) ok = spi_byte(0U, &ignored);
  for (index = 0U; index < len && ok != 0U; index++) ok = spi_byte(0U, &data[index]);
  HAL_GPIO_WritePin(port, pin, GPIO_PIN_SET);
  return ok;
}

static uint8_t accel_read(uint8_t reg, uint8_t *data, uint8_t len)
{
  return read_regs(ACC_CS_GPIO_Port, ACC_CS_Pin, reg, 1U, data, len);
}

static uint8_t gyro_read(uint8_t reg, uint8_t *data, uint8_t len)
{
  return read_regs(GYRO_CS_GPIO_Port, GYRO_CS_Pin, reg, 0U, data, len);
}

static uint8_t verify_accel(uint8_t reg, uint8_t value)
{
  uint8_t actual;
  if (write_reg(ACC_CS_GPIO_Port, ACC_CS_Pin, reg, value) == 0U) return 0U;
  HAL_Delay(1U);
  return accel_read(reg, &actual, 1U) != 0U && actual == value;
}

static uint8_t verify_gyro(uint8_t reg, uint8_t value)
{
  uint8_t actual;
  if (write_reg(GYRO_CS_GPIO_Port, GYRO_CS_Pin, reg, value) == 0U) return 0U;
  HAL_Delay(1U);
  return gyro_read(reg, &actual, 1U) != 0U && actual == value;
}

bmi088_status_t BMI088_Init(void)
{
  uint8_t id;
  HAL_GPIO_WritePin(ACC_CS_GPIO_Port, ACC_CS_Pin, GPIO_PIN_SET);
  HAL_GPIO_WritePin(GYRO_CS_GPIO_Port, GYRO_CS_Pin, GPIO_PIN_SET);
  HAL_Delay(5U);

  /* The first accelerometer read switches its interface from I2C to SPI. */
  if (accel_read(ACC_CHIP_ID, &id, 1U) == 0U ||
      accel_read(ACC_CHIP_ID, &id, 1U) == 0U) return BMI088_SPI_ERROR;
  if (id != ACC_CHIP_ID_VALUE) return BMI088_ACCEL_NOT_FOUND;
  if (write_reg(ACC_CS_GPIO_Port, ACC_CS_Pin, ACC_SOFTRESET, 0xB6U) == 0U) return BMI088_SPI_ERROR;
  HAL_Delay(80U);
  (void)accel_read(ACC_CHIP_ID, &id, 1U);
  if (accel_read(ACC_CHIP_ID, &id, 1U) == 0U) return BMI088_SPI_ERROR;
  if (id != ACC_CHIP_ID_VALUE) return BMI088_ACCEL_NOT_FOUND;
  if (verify_accel(ACC_PWR_CTRL, 0x04U) == 0U ||
      verify_accel(ACC_PWR_CONF, 0x00U) == 0U ||
      verify_accel(ACC_CONF, 0xABU) == 0U || /* normal bandwidth, 800 Hz */
      verify_accel(ACC_RANGE, 0x00U) == 0U) return BMI088_CONFIG_ERROR;

  if (gyro_read(GYRO_CHIP_ID, &id, 1U) == 0U) return BMI088_SPI_ERROR;
  if (id != GYRO_CHIP_ID_VALUE) return BMI088_GYRO_NOT_FOUND;
  if (write_reg(GYRO_CS_GPIO_Port, GYRO_CS_Pin, GYRO_SOFTRESET, 0xB6U) == 0U) return BMI088_SPI_ERROR;
  HAL_Delay(80U);
  if (gyro_read(GYRO_CHIP_ID, &id, 1U) == 0U) return BMI088_SPI_ERROR;
  if (id != GYRO_CHIP_ID_VALUE) return BMI088_GYRO_NOT_FOUND;
  if (verify_gyro(GYRO_RANGE, 0x02U) == 0U || /* +/-500 deg/s */
      verify_gyro(GYRO_BANDWIDTH, 0x82U) == 0U || /* 1 kHz ODR, 116 Hz BW */
      verify_gyro(GYRO_LPM1, 0x00U) == 0U ||
      verify_gyro(GYRO_CTRL, 0x80U) == 0U) return BMI088_CONFIG_ERROR;
  return BMI088_OK;
}

bmi088_status_t BMI088_Read(bmi088_sample_t *sample)
{
  uint8_t accel[6];
  uint8_t gyro[6];
  uint8_t temperature[2];
  int16_t raw;
  uint8_t axis;
  if (sample == 0) return BMI088_CONFIG_ERROR;
  if (accel_read(ACC_DATA, accel, sizeof(accel)) == 0U ||
      gyro_read(GYRO_DATA, gyro, sizeof(gyro)) == 0U ||
      accel_read(ACC_TEMP, temperature, sizeof(temperature)) == 0U) return BMI088_SPI_ERROR;
  for (axis = 0U; axis < 3U; axis++) {
    raw = (int16_t)((uint16_t)accel[2U * axis] | ((uint16_t)accel[2U * axis + 1U] << 8));
    sample->accel_m_s2[axis] = (float)raw * ACC_M_S2_PER_LSB;
    raw = (int16_t)((uint16_t)gyro[2U * axis] | ((uint16_t)gyro[2U * axis + 1U] << 8));
    sample->gyro_rad_s[axis] = (float)raw * GYRO_RAD_S_PER_LSB;
  }
  raw = (int16_t)(((uint16_t)temperature[0] << 3) | ((uint16_t)temperature[1] >> 5));
  if (raw > 1023) raw = (int16_t)(raw - 2048);
  sample->temperature_c = (float)raw * 0.125f + 23.0f;
  return BMI088_OK;
}
