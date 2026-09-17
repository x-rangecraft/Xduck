/* USB device configuration for STM32H723 OTG HS with its embedded FS PHY.
 * Adapted from STMicroelectronics STM32CubeH7 CDC device support.
 */
#ifndef XDUCK_USBD_CONF_H
#define XDUCK_USBD_CONF_H

#include "stm32h7xx_hal.h"
#include <string.h>

#define USBD_MAX_NUM_INTERFACES        1U
#define DEVICE_HS                      1U
#define USBD_MAX_NUM_CONFIGURATION     1U
#define USBD_MAX_STR_DESC_SIZ          512U
#define USBD_SELF_POWERED              1U
#define USBD_DEBUG_LEVEL               0U

#define USBD_malloc                    USBD_static_malloc
#define USBD_free                      USBD_static_free
#define USBD_memset                    memset
#define USBD_memcpy                    memcpy
#define USBD_Delay                     HAL_Delay
#define USBD_UsrLog(...)
#define USBD_ErrLog(...)
#define USBD_DbgLog(...)

void *USBD_static_malloc(uint32_t size);
void USBD_static_free(void *ptr);

#endif
