/**
  ******************************************************************************
  * File Name          : BSP_CAN.cpp
  * Description        : FDCAN wrapper for motor control
  ******************************************************************************
  */

#include "BSP_CAN.h"

#include <string.h>

#define BSP_CAN_CALLBACK_MAX      (8U)
static BSP_CAN_FunCallBack_t g_canCallbacks[BSP_CAN_CALLBACK_MAX];
static unsigned char g_canCallbackCount = 0U;
volatile unsigned int can_mit_tx_completed[32];
volatile unsigned int can_raw_rx_count[32];
volatile unsigned int can_tx_event_lost[3];

static FDCAN_HandleTypeDef *BSP_CAN_GetHandleByPort(unsigned char Port)
{
  if (Port == 1U) {
    return &hfdcan1;
  }
  if (Port == 2U) {
    return &hfdcan2;
  }
  if (Port == 3U) {
    return &hfdcan3;
  }
  return 0;
}

static unsigned char BSP_CAN_GetPortByHandle(FDCAN_HandleTypeDef *CanHandle)
{
  if (CanHandle == &hfdcan1) {
    return 1U;
  }
  if (CanHandle == &hfdcan2) {
    return 2U;
  }
  if (CanHandle == &hfdcan3) {
    return 3U;
  }
  return 0U;
}

static void BSP_CAN_ConfigFilterAndStart(FDCAN_HandleTypeDef *CanHandle)
{
  FDCAN_FilterTypeDef FilterConfig;

  memset(&FilterConfig, 0, sizeof(FilterConfig));
  FilterConfig.IdType = FDCAN_STANDARD_ID;
  FilterConfig.FilterIndex = 0U;
  FilterConfig.FilterType = FDCAN_FILTER_MASK;
  FilterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
  FilterConfig.FilterID1 = 0x000U;
  FilterConfig.FilterID2 = 0x000U;

  if (HAL_FDCAN_ConfigFilter(CanHandle, &FilterConfig) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_FDCAN_ConfigGlobalFilter(CanHandle,
                                   FDCAN_ACCEPT_IN_RX_FIFO0,
                                   FDCAN_REJECT,
                                   FDCAN_REJECT_REMOTE,
                                   FDCAN_REJECT_REMOTE) != HAL_OK) {
    Error_Handler();
  }
  if (CanHandle->Init.FrameFormat == FDCAN_FRAME_FD_BRS) {
    /* Secondary sample point follows the measured transceiver loop delay. */
    if (HAL_FDCAN_ConfigTxDelayCompensation(CanHandle,
          CanHandle->Init.DataPrescaler * CanHandle->Init.DataTimeSeg1, 0U) != HAL_OK ||
        HAL_FDCAN_EnableTxDelayCompensation(CanHandle) != HAL_OK) {
      Error_Handler();
    }
  }
  if (HAL_FDCAN_Start(CanHandle) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_FDCAN_ActivateNotification(CanHandle,
                                     FDCAN_IT_RX_FIFO0_NEW_MESSAGE | FDCAN_IT_TX_EVT_FIFO_NEW_DATA |
                                     FDCAN_IT_TX_EVT_FIFO_ELT_LOST,
                                     0U) != HAL_OK) {
    Error_Handler();
  }
}

void BSP_CAN_Init(void)
{
  unsigned char Index;

  g_canCallbackCount = 0U;
  for (Index = 0U; Index < BSP_CAN_CALLBACK_MAX; Index++) {
    g_canCallbacks[Index] = 0;
  }

  BSP_CAN_ConfigFilterAndStart(&hfdcan1);
  BSP_CAN_ConfigFilterAndStart(&hfdcan2);
  BSP_CAN_ConfigFilterAndStart(&hfdcan3);
}

void BSP_CAN_ClearTxMailboxes(unsigned char Port)
{
  FDCAN_HandleTypeDef *CanHandle;
  const uint32_t AllTxBuffers = 0xFFFFFFFFU;

  CanHandle = BSP_CAN_GetHandleByPort(Port);
  if (CanHandle == 0) {
    return;
  }

  if (HAL_FDCAN_AbortTxRequest(CanHandle, AllTxBuffers) != HAL_OK) {
    Error_Handler();
  }
}

unsigned char BSP_CAN_TrySendStandardDataMessage(unsigned char Port, unsigned int ID, unsigned char *Data)
{
  FDCAN_HandleTypeDef *CanHandle;
  FDCAN_TxHeaderTypeDef TxHeader;
  unsigned char TxData[8];

  CanHandle = BSP_CAN_GetHandleByPort(Port);
  if (CanHandle == 0 || Data == 0) {
    return 0U;
  }

  memset(&TxHeader, 0, sizeof(TxHeader));
  TxHeader.Identifier = ID & 0x7FFU;
  TxHeader.IdType = FDCAN_STANDARD_ID;
  TxHeader.TxFrameType = FDCAN_DATA_FRAME;
  TxHeader.DataLength = FDCAN_DLC_BYTES_8;
  TxHeader.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
  TxHeader.BitRateSwitch = FDCAN_BRS_OFF;
  TxHeader.FDFormat = FDCAN_CLASSIC_CAN;
  TxHeader.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
  TxHeader.MessageMarker = 0U;

  memcpy(TxData, Data, sizeof(TxData));

  if (HAL_FDCAN_GetTxFifoFreeLevel(CanHandle) == 0U) {
    return 0U;
  }

  if (HAL_FDCAN_AddMessageToTxFifoQ(CanHandle, &TxHeader, TxData) != HAL_OK) {
    return 0U;
  }
  return 1U;
}

unsigned char BSP_CAN_TrySendStandardFdDataMessage(unsigned char Port,
                                                   unsigned int ID,
                                                   unsigned char *Data,
                                                   unsigned char BitRateSwitch)
{
  return BSP_CAN_TrySendStandardFdFrame(Port, ID, Data, 8U, BitRateSwitch);
}

unsigned char BSP_CAN_TrySendStandardFdFrame(unsigned char Port, unsigned int ID,
                                            unsigned char *Data, unsigned char Length,
                                            unsigned char BitRateSwitch)
{
  FDCAN_HandleTypeDef *CanHandle;
  FDCAN_TxHeaderTypeDef TxHeader;
  unsigned char TxData[8];

  CanHandle = BSP_CAN_GetHandleByPort(Port);
  if (CanHandle == 0 || Data == 0 || (Length != 4U && Length != 8U)) {
    return 0U;
  }

  memset(&TxHeader, 0, sizeof(TxHeader));
  TxHeader.Identifier = ID & 0x7FFU;
  TxHeader.IdType = FDCAN_STANDARD_ID;
  TxHeader.TxFrameType = FDCAN_DATA_FRAME;
  TxHeader.DataLength = Length == 4U ? FDCAN_DLC_BYTES_4 : FDCAN_DLC_BYTES_8;
  TxHeader.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
  TxHeader.BitRateSwitch = (BitRateSwitch != 0U) ? FDCAN_BRS_ON : FDCAN_BRS_OFF;
  TxHeader.FDFormat = FDCAN_FD_CAN;
  TxHeader.TxEventFifoControl = FDCAN_STORE_TX_EVENTS;
  TxHeader.MessageMarker = 0U;
  if (ID >= 1U && ID <= 14U && Length == 8U) {
    unsigned char special = 1U;
    for (unsigned char i = 0U; i < 7U; ++i) if (Data[i] != 0xFFU) special = 0U;
    TxHeader.MessageMarker = special == 0U ? 1U : 0U;
  }

  memset(TxData, 0, sizeof(TxData));
  memcpy(TxData, Data, Length);

  if (HAL_FDCAN_GetTxFifoFreeLevel(CanHandle) == 0U) {
    return 0U;
  }

  if (HAL_FDCAN_AddMessageToTxFifoQ(CanHandle, &TxHeader, TxData) != HAL_OK) {
    return 0U;
  }
  return 1U;
}

void BSP_CAN_SendStandardDataMessage(unsigned char Port, unsigned int ID, unsigned char *Data)
{
  (void)BSP_CAN_TrySendStandardDataMessage(Port, ID, Data);
}

extern "C" void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *CanHandle, uint32_t RxFifo0ITs)
{
  FDCAN_RxHeaderTypeDef RxHeader;
  unsigned char RxData[64];
  unsigned char Port;
  unsigned char Index;

  if ((RxFifo0ITs & FDCAN_IT_RX_FIFO0_NEW_MESSAGE) == 0U) {
    return;
  }

  while (HAL_FDCAN_GetRxFifoFillLevel(CanHandle, FDCAN_RX_FIFO0) > 0U) {
    if (HAL_FDCAN_GetRxMessage(CanHandle, FDCAN_RX_FIFO0, &RxHeader, RxData) != HAL_OK) {
      Error_Handler();
    }

    if (RxHeader.IdType == FDCAN_STANDARD_ID && RxHeader.Identifier < 32U) {
      can_raw_rx_count[RxHeader.Identifier]++;
    }
    if (RxHeader.IdType != FDCAN_STANDARD_ID || RxHeader.RxFrameType != FDCAN_DATA_FRAME ||
        RxHeader.DataLength != FDCAN_DLC_BYTES_8) {
      continue;
    }

    Port = BSP_CAN_GetPortByHandle(CanHandle);
    if (Port == 0U) {
      continue;
    }

    for (Index = 0U; Index < g_canCallbackCount; Index++) {
      if (g_canCallbacks[Index] != 0) {
        g_canCallbacks[Index](Port, RxHeader.Identifier, RxData);
      }
    }
  }
}

unsigned char BSP_CAN_ReconfigurePortTiming(unsigned char Port,
                                            unsigned int FrameFormat,
                                            unsigned int NominalPrescaler,
                                            unsigned int NominalSyncJumpWidth,
                                            unsigned int NominalTimeSeg1,
                                            unsigned int NominalTimeSeg2,
                                            unsigned int DataPrescaler,
                                            unsigned int DataSyncJumpWidth,
                                            unsigned int DataTimeSeg1,
                                            unsigned int DataTimeSeg2,
                                            unsigned int RxElementSize)
{
  FDCAN_HandleTypeDef *CanHandle;

  CanHandle = BSP_CAN_GetHandleByPort(Port);
  if (CanHandle == 0) {
    return 0U;
  }

  (void)HAL_FDCAN_Stop(CanHandle);
  (void)HAL_FDCAN_DeInit(CanHandle);

  CanHandle->Init.FrameFormat = FrameFormat;
  CanHandle->Init.AutoRetransmission = ENABLE;
  CanHandle->Init.ProtocolException = ENABLE;
  CanHandle->Init.NominalPrescaler = NominalPrescaler;
  CanHandle->Init.NominalSyncJumpWidth = NominalSyncJumpWidth;
  CanHandle->Init.NominalTimeSeg1 = NominalTimeSeg1;
  CanHandle->Init.NominalTimeSeg2 = NominalTimeSeg2;
  CanHandle->Init.DataPrescaler = DataPrescaler;
  CanHandle->Init.DataSyncJumpWidth = DataSyncJumpWidth;
  CanHandle->Init.DataTimeSeg1 = DataTimeSeg1;
  CanHandle->Init.DataTimeSeg2 = DataTimeSeg2;
  CanHandle->Init.RxFifo0ElmtSize = RxElementSize;
  CanHandle->Init.RxFifo1ElmtSize = RxElementSize;
  CanHandle->Init.RxBufferSize = RxElementSize;

  if (HAL_FDCAN_Init(CanHandle) != HAL_OK) {
    return 0U;
  }
  BSP_CAN_ConfigFilterAndStart(CanHandle);
  return 1U;
}

void BSP_CAN_AddRxCallBackFunction(BSP_CAN_FunCallBack_t Function)
{
  unsigned char Index;

  if (Function == 0) {
    return;
  }

  for (Index = 0U; Index < g_canCallbackCount; Index++) {
    if (g_canCallbacks[Index] == Function) {
      return;
    }
  }

  if (g_canCallbackCount < BSP_CAN_CALLBACK_MAX) {
    g_canCallbacks[g_canCallbackCount] = Function;
    g_canCallbackCount++;
  }
}

/**********************************END OF FILE***********************************/

extern "C" void HAL_FDCAN_TxEventFifoCallback(FDCAN_HandleTypeDef *handle, uint32_t events)
{
  unsigned char port = BSP_CAN_GetPortByHandle(handle);
  if (port == 0U) return;
  if ((events & FDCAN_IT_TX_EVT_FIFO_ELT_LOST) != 0U) can_tx_event_lost[port-1U]++;
  while ((handle->Instance->TXEFS & FDCAN_TXEFS_EFFL) != 0U) {
    FDCAN_TxEventFifoTypeDef event;
    if (HAL_FDCAN_GetTxEvent(handle, &event) != HAL_OK) { can_tx_event_lost[port-1U]++; break; }
    if (event.IdType == FDCAN_STANDARD_ID && event.Identifier < 32U && event.MessageMarker == 1U) {
      can_mit_tx_completed[event.Identifier]++;
    }
  }
}
