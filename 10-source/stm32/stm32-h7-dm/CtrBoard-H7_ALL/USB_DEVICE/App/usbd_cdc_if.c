/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : usbd_cdc_if.c
  * @version        : v1.0_Cube
  * @brief          : Usb device for Virtual Com Port.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "usbd_cdc_if.h"

/* USER CODE BEGIN INCLUDE */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "dmusb_task.h"
#include "usbd_core.h"

/* USER CODE END INCLUDE */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* Private variables ---------------------------------------------------------*/
#define USB_LOG_LINE_MAX (192U)

/* USER CODE END PV */

/** @addtogroup STM32_USB_OTG_DEVICE_LIBRARY
  * @brief Usb device library.
  * @{
  */

/** @addtogroup USBD_CDC_IF
  * @{
  */

/** @defgroup USBD_CDC_IF_Private_TypesDefinitions USBD_CDC_IF_Private_TypesDefinitions
  * @brief Private types.
  * @{
  */

/* USER CODE BEGIN PRIVATE_TYPES */

/* USER CODE END PRIVATE_TYPES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Defines USBD_CDC_IF_Private_Defines
  * @brief Private defines.
  * @{
  */

/* USER CODE BEGIN PRIVATE_DEFINES */
/* USER CODE END PRIVATE_DEFINES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Macros USBD_CDC_IF_Private_Macros
  * @brief Private macros.
  * @{
  */

/* USER CODE BEGIN PRIVATE_MACRO */

/* USER CODE END PRIVATE_MACRO */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Variables USBD_CDC_IF_Private_Variables
  * @brief Private variables.
  * @{
  */

/* Create buffer for reception and transmission           */
/* It's up to user to redefine and/or remove those define */
/** Received data over USB are stored in this buffer      */
uint8_t UserRxBufferHS[APP_RX_DATA_SIZE];

/** Data to send over USB CDC are stored in this buffer   */
uint8_t UserTxBufferHS[APP_TX_DATA_SIZE];

/* USER CODE BEGIN PRIVATE_VARIABLES */
static volatile uint8_t g_usbLogBusy = 0U;
#if !defined(DMUSB_BINARY_PROTOCOL) || (DMUSB_BINARY_PROTOCOL == 0)
static char g_usbLogLine[USB_LOG_LINE_MAX];
#endif

/* USER CODE END PRIVATE_VARIABLES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Exported_Variables USBD_CDC_IF_Exported_Variables
  * @brief Public variables.
  * @{
  */

extern USBD_HandleTypeDef hUsbDeviceHS;

/* USER CODE BEGIN EXPORTED_VARIABLES */

/* USER CODE END EXPORTED_VARIABLES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_FunctionPrototypes USBD_CDC_IF_Private_FunctionPrototypes
  * @brief Private functions declaration.
  * @{
  */

static int8_t CDC_Init_HS(void);
static int8_t CDC_DeInit_HS(void);
static int8_t CDC_Control_HS(uint8_t cmd, uint8_t* pbuf, uint16_t length);
static int8_t CDC_Receive_HS(uint8_t* pbuf, uint32_t *Len);
static int8_t CDC_TransmitCplt_HS(uint8_t *pbuf, uint32_t *Len, uint8_t epnum);

/* USER CODE BEGIN PRIVATE_FUNCTIONS_DECLARATION */
#if !defined(DMUSB_BINARY_PROTOCOL) || (DMUSB_BINARY_PROTOCOL == 0)
static uint8_t USB_LogStartTransmit(const char *text);
#endif

/* USER CODE END PRIVATE_FUNCTIONS_DECLARATION */

/**
  * @}
  */

USBD_CDC_ItfTypeDef USBD_Interface_fops_HS =
{
  CDC_Init_HS,
  CDC_DeInit_HS,
  CDC_Control_HS,
  CDC_Receive_HS,
  CDC_TransmitCplt_HS
};

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Initializes the CDC media low layer over the USB HS IP
  * @retval USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Init_HS(void)
{
  /* USER CODE BEGIN 8 */
  /* Set Application Buffers */
  USBD_CDC_SetTxBuffer(&hUsbDeviceHS, UserTxBufferHS, 0);
  USBD_CDC_SetRxBuffer(&hUsbDeviceHS, UserRxBufferHS);
  return (USBD_OK);
  /* USER CODE END 8 */
}

/**
  * @brief  DeInitializes the CDC media low layer
  * @param  None
  * @retval USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_DeInit_HS(void)
{
  /* USER CODE BEGIN 9 */
  return (USBD_OK);
  /* USER CODE END 9 */
}

/**
  * @brief  Manage the CDC class requests
  * @param  cmd: Command code
  * @param  pbuf: Buffer containing command data (request parameters)
  * @param  length: Number of data to be sent (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Control_HS(uint8_t cmd, uint8_t* pbuf, uint16_t length)
{
  /* USER CODE BEGIN 10 */
  switch(cmd)
  {
  case CDC_SEND_ENCAPSULATED_COMMAND:

    break;

  case CDC_GET_ENCAPSULATED_RESPONSE:

    break;

  case CDC_SET_COMM_FEATURE:

    break;

  case CDC_GET_COMM_FEATURE:

    break;

  case CDC_CLEAR_COMM_FEATURE:

    break;

  /*******************************************************************************/
  /* Line Coding Structure                                                       */
  /*-----------------------------------------------------------------------------*/
  /* Offset | Field       | Size | Value  | Description                          */
  /* 0      | dwDTERate   |   4  | Number |Data terminal rate, in bits per second*/
  /* 4      | bCharFormat |   1  | Number | Stop bits                            */
  /*                                        0 - 1 Stop bit                       */
  /*                                        1 - 1.5 Stop bits                    */
  /*                                        2 - 2 Stop bits                      */
  /* 5      | bParityType |  1   | Number | Parity                               */
  /*                                        0 - None                             */
  /*                                        1 - Odd                              */
  /*                                        2 - Even                             */
  /*                                        3 - Mark                             */
  /*                                        4 - Space                            */
  /* 6      | bDataBits  |   1   | Number Data bits (5, 6, 7, 8 or 16).          */
  /*******************************************************************************/
  case CDC_SET_LINE_CODING:

    break;

  case CDC_GET_LINE_CODING:

    break;

  case CDC_SET_CONTROL_LINE_STATE:

    break;

  case CDC_SEND_BREAK:

    break;

  default:
    break;
  }

  return (USBD_OK);
  /* USER CODE END 10 */
}

/**
  * @brief Data received over USB OUT endpoint are sent over CDC interface
  *         through this function.
  *
  *         @note
  *         This function will issue a NAK packet on any OUT packet received on
  *         USB endpoint until exiting this function. If you exit this function
  *         before transfer is complete on CDC interface (ie. using DMA controller)
  *         it will result in receiving more data while previous ones are still
  *         not sent.
  *
  * @param  Buf: Buffer of data to be received
  * @param  Len: Number of data received (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAILL
  */
static int8_t CDC_Receive_HS(uint8_t* Buf, uint32_t *Len)
{
  /* USER CODE BEGIN 11 */
  if (Len != 0 && *Len <= 0xFFFFU) {
    DMUSB_OnReceive(Buf, (uint16_t)*Len);
  }
  USBD_CDC_SetRxBuffer(&hUsbDeviceHS, &Buf[0]);
  USBD_CDC_ReceivePacket(&hUsbDeviceHS);
  return (USBD_OK);
  /* USER CODE END 11 */
}

/**
  * @brief  Data to send over USB IN endpoint are sent over CDC interface
  *         through this function.
  * @param  Buf: Buffer of data to be sent
  * @param  Len: Number of data to be sent (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL or USBD_BUSY
  */
uint8_t CDC_Transmit_HS(uint8_t* Buf, uint16_t Len)
{
  uint8_t result = USBD_OK;
  /* USER CODE BEGIN 12 */
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceHS.pClassData;
  if (hcdc == 0) {
    return USBD_FAIL;
  }
  if (hcdc->TxState != 0){
    return USBD_BUSY;
  }
  USBD_CDC_SetTxBuffer(&hUsbDeviceHS, Buf, Len);
  result = USBD_CDC_TransmitPacket(&hUsbDeviceHS);
  /* USER CODE END 12 */
  return result;
}

uint8_t USB_CDC_TxReady(void)
{
  USBD_CDC_HandleTypeDef *hcdc;
  if (hUsbDeviceHS.dev_state != USBD_STATE_CONFIGURED) {
    return 0U;
  }
  hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceHS.pClassData;
  return (uint8_t)((hcdc != 0 && hcdc->TxState == 0U) ? 1U : 0U);
}

void USB_CDC_GetTxState(uint8_t *dev_state, uint32_t *tx_state)
{
  USBD_CDC_HandleTypeDef *hcdc =
      (USBD_CDC_HandleTypeDef*)hUsbDeviceHS.pClassData;
  if (dev_state != 0) {
    *dev_state = (uint8_t)hUsbDeviceHS.dev_state;
  }
  if (tx_state != 0) {
    *tx_state = (hcdc != 0) ? hcdc->TxState : 0xFFFFFFFFU;
  }
}

uint8_t USB_CDC_RecoverTx(void)
{
  USBD_CDC_HandleTypeDef *hcdc =
      (USBD_CDC_HandleTypeDef*)hUsbDeviceHS.pClassData;
  uint32_t primask;
  uint8_t result;

  if (hUsbDeviceHS.dev_state != USBD_STATE_CONFIGURED || hcdc == 0) {
    return USBD_FAIL;
  }
  if (hcdc->TxState == 0U) {
    return USBD_OK;
  }

  /* A lost DataIn completion leaves TxState set forever. Flush only the CDC IN
   * endpoint and release the software state after the caller has observed it
   * stuck for several state periods. Masking interrupts closes the race with a
   * late completion callback while the endpoint bookkeeping is repaired. */
  primask = __get_PRIMASK();
  __disable_irq();
  result = (uint8_t)USBD_LL_FlushEP(&hUsbDeviceHS, CDC_IN_EP);
  if (result == USBD_OK) {
    hUsbDeviceHS.ep_in[CDC_IN_EP & 0x0FU].total_length = 0U;
    hcdc->TxState = 0U;
  }
  if (primask == 0U) {
    __enable_irq();
  }
  return result;
}

int USB_LogPrintf(const char *fmt, ...)
{
  va_list ap;
  int len;
  char line[USB_LOG_LINE_MAX];

  va_start(ap, fmt);
  len = vsnprintf(line, sizeof(line), fmt, ap);
  va_end(ap);

  if (len <= 0) {
    return len;
  }
  USB_LogWrite(line);
  return len;
}

void USB_LogWrite(const char *text)
{
#if defined(DMUSB_BINARY_PROTOCOL) && (DMUSB_BINARY_PROTOCOL != 0)
  /* Text and binary frames cannot share one CDC byte stream. */
  (void)text;
#else
  if (text == 0) {
    return;
  }
  (void)USB_LogStartTransmit(text);
#endif
}

void USB_LogTask_Init(void)
{
  /* Direct CDC transmit is used during USB bring-up. */
}

/**
  * @brief  CDC_TransmitCplt_HS
  *         Data transmitted callback
  *
  *         @note
  *         This function is IN transfer complete callback used to inform user that
  *         the submitted Data is successfully sent over USB.
  *
  * @param  Buf: Buffer of data to be received
  * @param  Len: Number of data received (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_TransmitCplt_HS(uint8_t *Buf, uint32_t *Len, uint8_t epnum)
{
  uint8_t result = USBD_OK;
  /* USER CODE BEGIN 14 */
  UNUSED(Buf);
  UNUSED(Len);
  UNUSED(epnum);
  g_usbLogBusy = 0U;
  /* USER CODE END 14 */
  return result;
}

/* USER CODE BEGIN PRIVATE_FUNCTIONS_IMPLEMENTATION */
#if !defined(DMUSB_BINARY_PROTOCOL) || (DMUSB_BINARY_PROTOCOL == 0)
static uint8_t USB_LogStartTransmit(const char *text)
{
  USBD_CDC_HandleTypeDef *hcdc;
  uint16_t len;
  uint8_t result;

  if (text == 0) {
    return USBD_FAIL;
  }
  if (hUsbDeviceHS.dev_state != USBD_STATE_CONFIGURED) {
    return USBD_FAIL;
  }
  hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceHS.pClassData;
  if (hcdc != 0 && hcdc->TxState == 0U) {
    g_usbLogBusy = 0U;
  }
  if (g_usbLogBusy != 0U) {
    return USBD_BUSY;
  }

  len = (uint16_t)strnlen(text, sizeof(g_usbLogLine) - 2U);
  if (len == 0U) {
    return USBD_OK;
  }

  memcpy(g_usbLogLine, text, len);
  g_usbLogLine[len++] = '\r';
  g_usbLogLine[len++] = '\n';

  g_usbLogBusy = 1U;
  result = CDC_Transmit_HS((uint8_t *)g_usbLogLine, len);
  if (result != USBD_OK) {
    g_usbLogBusy = 0U;
  }
  return result;
}
#endif


/* USER CODE END PRIVATE_FUNCTIONS_IMPLEMENTATION */

/**
  * @}
  */

/**
  * @}
  */
