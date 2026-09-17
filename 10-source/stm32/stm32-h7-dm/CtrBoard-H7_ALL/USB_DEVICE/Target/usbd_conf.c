/* STM32H723 USB CDC low-level bridge: OTG HS core with embedded FS PHY.
 * Adapted from STMicroelectronics STM32CubeH7 CDC device support.
 */
#include "usbd_conf.h"
#include "usbd_core.h"
#include "usbd_cdc.h"
#include "main.h"

/* Referenced by OTG_HS_IRQHandler in Core/Src/stm32h7xx_it.c. */
PCD_HandleTypeDef hpcd_USB_OTG_HS;

static USBD_StatusTypeDef USB_Status(HAL_StatusTypeDef status)
{
  return (status == HAL_OK) ? USBD_OK : USBD_FAIL;
}

void HAL_PCD_MspInit(PCD_HandleTypeDef *hpcd)
{
  GPIO_InitTypeDef gpio = {0};
  RCC_PeriphCLKInitTypeDef clock = {0};
  if (hpcd->Instance != USB_OTG_HS) return;

  clock.PeriphClockSelection = RCC_PERIPHCLK_USB;
  clock.UsbClockSelection = RCC_USBCLKSOURCE_HSI48;
  if (HAL_RCCEx_PeriphCLKConfig(&clock) != HAL_OK) Error_Handler();
  HAL_PWREx_EnableUSBVoltageDetector();

  __HAL_RCC_GPIOA_CLK_ENABLE();
  gpio.Pin = GPIO_PIN_11 | GPIO_PIN_12;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  gpio.Alternate = GPIO_AF10_OTG1_HS;
  HAL_GPIO_Init(GPIOA, &gpio);

  __HAL_RCC_USB1_OTG_HS_CLK_ENABLE();
  HAL_NVIC_SetPriority(OTG_HS_IRQn, 5U, 0U);
  HAL_NVIC_EnableIRQ(OTG_HS_IRQn);
}

void HAL_PCD_MspDeInit(PCD_HandleTypeDef *hpcd)
{
  if (hpcd->Instance != USB_OTG_HS) return;
  HAL_NVIC_DisableIRQ(OTG_HS_IRQn);
  __HAL_RCC_USB1_OTG_HS_CLK_DISABLE();
  HAL_GPIO_DeInit(GPIOA, GPIO_PIN_11 | GPIO_PIN_12);
}

void HAL_PCD_SetupStageCallback(PCD_HandleTypeDef *hpcd)
{
  (void)USBD_LL_SetupStage(hpcd->pData, (uint8_t *)hpcd->Setup);
}

void HAL_PCD_DataOutStageCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum)
{
  (void)USBD_LL_DataOutStage(hpcd->pData, epnum, hpcd->OUT_ep[epnum].xfer_buff);
}

void HAL_PCD_DataInStageCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum)
{
  (void)USBD_LL_DataInStage(hpcd->pData, epnum, hpcd->IN_ep[epnum].xfer_buff);
}

void HAL_PCD_SOFCallback(PCD_HandleTypeDef *hpcd)
{
  (void)USBD_LL_SOF(hpcd->pData);
}

void HAL_PCD_ResetCallback(PCD_HandleTypeDef *hpcd)
{
  (void)USBD_LL_Reset(hpcd->pData);
  (void)USBD_LL_SetSpeed(hpcd->pData, USBD_SPEED_FULL);
}

void HAL_PCD_SuspendCallback(PCD_HandleTypeDef *hpcd)
{
  (void)USBD_LL_Suspend(hpcd->pData);
}

void HAL_PCD_ResumeCallback(PCD_HandleTypeDef *hpcd)
{
  (void)USBD_LL_Resume(hpcd->pData);
}

void HAL_PCD_ISOOUTIncompleteCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum)
{
  (void)USBD_LL_IsoOUTIncomplete(hpcd->pData, epnum);
}

void HAL_PCD_ISOINIncompleteCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum)
{
  (void)USBD_LL_IsoINIncomplete(hpcd->pData, epnum);
}

void HAL_PCD_ConnectCallback(PCD_HandleTypeDef *hpcd)
{
  (void)USBD_LL_DevConnected(hpcd->pData);
}

void HAL_PCD_DisconnectCallback(PCD_HandleTypeDef *hpcd)
{
  (void)USBD_LL_DevDisconnected(hpcd->pData);
}

USBD_StatusTypeDef USBD_LL_Init(USBD_HandleTypeDef *pdev)
{
  PCD_HandleTypeDef *hpcd = &hpcd_USB_OTG_HS;
  hpcd->Instance = USB_OTG_HS;
  hpcd->Init.dev_endpoints = 9U;
  hpcd->Init.speed = PCD_SPEED_FULL;
  hpcd->Init.dma_enable = DISABLE;
  hpcd->Init.phy_itface = PCD_PHY_EMBEDDED;
  hpcd->Init.Sof_enable = DISABLE;
  hpcd->Init.low_power_enable = DISABLE;
  hpcd->Init.lpm_enable = DISABLE;
  hpcd->Init.vbus_sensing_enable = DISABLE;
  hpcd->Init.use_dedicated_ep1 = DISABLE;
  hpcd->pData = pdev;
  pdev->pData = hpcd;

  if (HAL_PCD_Init(hpcd) != HAL_OK ||
      HAL_PCDEx_SetRxFiFo(hpcd, 0x100U) != HAL_OK ||
      HAL_PCDEx_SetTxFiFo(hpcd, 0U, 0x40U) != HAL_OK ||
      HAL_PCDEx_SetTxFiFo(hpcd, 1U, 0x80U) != HAL_OK ||
      HAL_PCDEx_SetTxFiFo(hpcd, 2U, 0x40U) != HAL_OK) {
    return USBD_FAIL;
  }
  return USBD_OK;
}

USBD_StatusTypeDef USBD_LL_DeInit(USBD_HandleTypeDef *pdev)
{
  return USB_Status(HAL_PCD_DeInit(pdev->pData));
}

USBD_StatusTypeDef USBD_LL_Start(USBD_HandleTypeDef *pdev)
{
  return USB_Status(HAL_PCD_Start(pdev->pData));
}

USBD_StatusTypeDef USBD_LL_Stop(USBD_HandleTypeDef *pdev)
{
  return USB_Status(HAL_PCD_Stop(pdev->pData));
}

USBD_StatusTypeDef USBD_LL_OpenEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr,
                                  uint8_t ep_type, uint16_t ep_mps)
{
  return USB_Status(HAL_PCD_EP_Open(pdev->pData, ep_addr, ep_mps, ep_type));
}

USBD_StatusTypeDef USBD_LL_CloseEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr)
{
  return USB_Status(HAL_PCD_EP_Close(pdev->pData, ep_addr));
}

USBD_StatusTypeDef USBD_LL_FlushEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr)
{
  return USB_Status(HAL_PCD_EP_Flush(pdev->pData, ep_addr));
}

USBD_StatusTypeDef USBD_LL_StallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr)
{
  return USB_Status(HAL_PCD_EP_SetStall(pdev->pData, ep_addr));
}

USBD_StatusTypeDef USBD_LL_ClearStallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr)
{
  return USB_Status(HAL_PCD_EP_ClrStall(pdev->pData, ep_addr));
}

uint8_t USBD_LL_IsStallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr)
{
  PCD_HandleTypeDef *hpcd = pdev->pData;
  uint8_t endpoint = ep_addr & 0x7FU;
  return ((ep_addr & 0x80U) != 0U) ? hpcd->IN_ep[endpoint].is_stall :
                                      hpcd->OUT_ep[endpoint].is_stall;
}

USBD_StatusTypeDef USBD_LL_SetUSBAddress(USBD_HandleTypeDef *pdev, uint8_t dev_addr)
{
  return USB_Status(HAL_PCD_SetAddress(pdev->pData, dev_addr));
}

USBD_StatusTypeDef USBD_LL_Transmit(USBD_HandleTypeDef *pdev, uint8_t ep_addr,
                                    uint8_t *buffer, uint32_t size)
{
  pdev->ep_in[ep_addr & 0x7FU].total_length = size;
  return USB_Status(HAL_PCD_EP_Transmit(pdev->pData, ep_addr, buffer, size));
}

USBD_StatusTypeDef USBD_LL_PrepareReceive(USBD_HandleTypeDef *pdev, uint8_t ep_addr,
                                          uint8_t *buffer, uint32_t size)
{
  return USB_Status(HAL_PCD_EP_Receive(pdev->pData, ep_addr, buffer, size));
}

uint32_t USBD_LL_GetRxDataSize(USBD_HandleTypeDef *pdev, uint8_t ep_addr)
{
  return HAL_PCD_EP_GetRxCount(pdev->pData, ep_addr);
}

USBD_StatusTypeDef USBD_LL_SetTestMode(USBD_HandleTypeDef *pdev, uint8_t testmode)
{
  return USB_Status(HAL_PCD_SetTestMode(pdev->pData, testmode));
}

void *USBD_static_malloc(uint32_t size)
{
  static uint32_t memory[(sizeof(USBD_CDC_HandleTypeDef) + 3U) / 4U];
  return (size <= sizeof(memory)) ? memory : NULL;
}

void USBD_static_free(void *ptr)
{
  (void)ptr;
}

void USBD_LL_Delay(uint32_t delay)
{
  HAL_Delay(delay);
}
