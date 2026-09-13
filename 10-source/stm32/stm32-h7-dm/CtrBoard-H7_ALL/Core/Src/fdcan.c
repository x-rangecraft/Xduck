/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    fdcan.c
  * @brief   This file provides code for the configuration
  *          of the FDCAN instances.
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
#include "fdcan.h"

/* USER CODE BEGIN 0 */

/*
 * The three FDCAN instances share the 10 KiB Message RAM.  Size each port for
 * the installed motors: clear-error + enable + initial MIT command in TX,
 * plus one in-flight idle feedback poll. RX holds two feedback batches.
 * A single TX batch overflows during the synchronous enable operation.
 * TX event FIFOs count actual completed transmissions (two words/event).
 * Offsets and element sizes are expressed in 32-bit words by the STM32 HAL.
 */
#define FDCAN1_MOTOR_COUNT              5U
#define FDCAN2_MOTOR_COUNT              5U
#define FDCAN3_MOTOR_COUNT              4U
#define FDCAN_RX_BATCH_COUNT            2U
#define FDCAN_STD_FILTER_WORDS          1U
#define FDCAN_TX_EVENT_ELEMENTS         16U
#define FDCAN_TX_EVENT_WORDS            (2U * FDCAN_TX_EVENT_ELEMENTS)
#define FDCAN_TX_STARTUP_BATCH_COUNT    3U
#define FDCAN1_TX_FIFO_ELEMENTS         (FDCAN1_MOTOR_COUNT * FDCAN_TX_STARTUP_BATCH_COUNT + 1U)
#define FDCAN2_TX_FIFO_ELEMENTS         (FDCAN2_MOTOR_COUNT * FDCAN_TX_STARTUP_BATCH_COUNT + 1U)
#define FDCAN3_TX_FIFO_ELEMENTS         (FDCAN3_MOTOR_COUNT * FDCAN_TX_STARTUP_BATCH_COUNT + 1U)

#define FDCAN1_RX_FIFO0_ELEMENTS        (FDCAN1_MOTOR_COUNT * FDCAN_RX_BATCH_COUNT)
#define FDCAN2_RX_FIFO0_ELEMENTS        (FDCAN2_MOTOR_COUNT * FDCAN_RX_BATCH_COUNT)
#define FDCAN3_RX_FIFO0_ELEMENTS        (FDCAN3_MOTOR_COUNT * FDCAN_RX_BATCH_COUNT)

#define FDCAN1_MESSAGE_RAM_WORDS        (FDCAN_STD_FILTER_WORDS + FDCAN_TX_EVENT_WORDS + \
                                         (FDCAN1_RX_FIFO0_ELEMENTS * FDCAN_DATA_BYTES_8) + \
                                         (FDCAN1_TX_FIFO_ELEMENTS * FDCAN_DATA_BYTES_8))
#define FDCAN2_MESSAGE_RAM_WORDS        (FDCAN_STD_FILTER_WORDS + FDCAN_TX_EVENT_WORDS + \
                                         (FDCAN2_RX_FIFO0_ELEMENTS * FDCAN_DATA_BYTES_8) + \
                                         (FDCAN2_TX_FIFO_ELEMENTS * FDCAN_DATA_BYTES_8))
#define FDCAN3_MESSAGE_RAM_WORDS        (FDCAN_STD_FILTER_WORDS + FDCAN_TX_EVENT_WORDS + \
                                         (FDCAN3_RX_FIFO0_ELEMENTS * FDCAN_DATA_BYTES_8) + \
                                         (FDCAN3_TX_FIFO_ELEMENTS * FDCAN_DATA_BYTES_8))

#define FDCAN1_MESSAGE_RAM_OFFSET       0U
#define FDCAN2_MESSAGE_RAM_OFFSET       (FDCAN1_MESSAGE_RAM_OFFSET + FDCAN1_MESSAGE_RAM_WORDS)
#define FDCAN3_MESSAGE_RAM_OFFSET       (FDCAN2_MESSAGE_RAM_OFFSET + FDCAN2_MESSAGE_RAM_WORDS)

_Static_assert(FDCAN1_TX_FIFO_ELEMENTS <= 32U && FDCAN2_TX_FIFO_ELEMENTS <= 32U &&
               FDCAN3_TX_FIFO_ELEMENTS <= 32U, "FDCAN TX FIFO exceeds hardware capacity");
_Static_assert(FDCAN3_MESSAGE_RAM_OFFSET + FDCAN3_MESSAGE_RAM_WORDS <= 2560U,
               "FDCAN instances overlap the end of the shared 10 KiB message RAM");

/* USER CODE END 0 */

FDCAN_HandleTypeDef hfdcan1;
FDCAN_HandleTypeDef hfdcan2;
FDCAN_HandleTypeDef hfdcan3;

/*
 * PLL1Q = 120 MHz, CCU /1. All three GF43X40-10 buses use the same timing.
 * Nominal: 120/(1*120)=1 Mbps, sample 75%, SJW 250 ns.
 * Data:    120/(2*12)=5 Mbps, sample 83.33%, SJW 33.33 ns.
 * The old 6/15/4/4 nominal profile (80%) produced sustained protocol errors
 * on all three buses. Changing only nominal timing cleared the errors in
 * the 2026-09-13 disabled 14-motor A/B test; see the hardware audit document.
 * Match the DM FD example's nominal 75% point using our 120 MHz clock:
 * do not copy its 80 MHz prescaler/segment numbers directly.
 */
/* FDCAN1 init function */
void MX_FDCAN1_Init(void)
{

  /* USER CODE BEGIN FDCAN1_Init 0 */

  /* USER CODE END FDCAN1_Init 0 */

  /* USER CODE BEGIN FDCAN1_Init 1 */

  /* USER CODE END FDCAN1_Init 1 */
  hfdcan1.Instance = FDCAN1;
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  hfdcan1.Init.FrameFormat = FDCAN_FRAME_FD_BRS;
#else
  hfdcan1.Init.FrameFormat = FDCAN_FRAME_CLASSIC;
#endif
  hfdcan1.Init.Mode = FDCAN_MODE_NORMAL;
  hfdcan1.Init.AutoRetransmission = ENABLE;
  /* Two nominal-bit idle times after TX let lower-priority motor feedback
   * arbitrate before another queued command. Required by this GF43 network. */
  hfdcan1.Init.TransmitPause = ENABLE;
  hfdcan1.Init.ProtocolException = ENABLE;
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  hfdcan1.Init.NominalPrescaler = 1;
  hfdcan1.Init.NominalSyncJumpWidth = 30;
  hfdcan1.Init.NominalTimeSeg1 = 89;
  hfdcan1.Init.NominalTimeSeg2 = 30;
#else
  hfdcan1.Init.NominalPrescaler = 12;
  hfdcan1.Init.NominalSyncJumpWidth = 4;
  hfdcan1.Init.NominalTimeSeg1 = 15;
  hfdcan1.Init.NominalTimeSeg2 = 4;
#endif
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  hfdcan1.Init.DataPrescaler = 2;
  hfdcan1.Init.DataSyncJumpWidth = 2;
  hfdcan1.Init.DataTimeSeg1 = 9;
  hfdcan1.Init.DataTimeSeg2 = 2;
#else
  hfdcan1.Init.DataPrescaler = 1;
  hfdcan1.Init.DataSyncJumpWidth = 1;
  hfdcan1.Init.DataTimeSeg1 = 1;
  hfdcan1.Init.DataTimeSeg2 = 1;
#endif
  hfdcan1.Init.MessageRAMOffset = FDCAN1_MESSAGE_RAM_OFFSET;
  hfdcan1.Init.StdFiltersNbr = 1;
  hfdcan1.Init.ExtFiltersNbr = 0;
  hfdcan1.Init.RxFifo0ElmtsNbr = FDCAN1_RX_FIFO0_ELEMENTS;
  hfdcan1.Init.RxFifo0ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan1.Init.RxFifo1ElmtsNbr = 0;
  hfdcan1.Init.RxFifo1ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan1.Init.RxBuffersNbr = 0;
  hfdcan1.Init.RxBufferSize = FDCAN_DATA_BYTES_8;
  hfdcan1.Init.TxEventsNbr = FDCAN_TX_EVENT_ELEMENTS;
  hfdcan1.Init.TxBuffersNbr = 0;
  hfdcan1.Init.TxFifoQueueElmtsNbr = FDCAN1_TX_FIFO_ELEMENTS;
  hfdcan1.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
  hfdcan1.Init.TxElmtSize = FDCAN_DATA_BYTES_8;
  if (HAL_FDCAN_Init(&hfdcan1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN FDCAN1_Init 2 */

  /* USER CODE END FDCAN1_Init 2 */

}
/* FDCAN2 init function */
void MX_FDCAN2_Init(void)
{

  /* USER CODE BEGIN FDCAN2_Init 0 */

  /* USER CODE END FDCAN2_Init 0 */

  /* USER CODE BEGIN FDCAN2_Init 1 */

  /* USER CODE END FDCAN2_Init 1 */
  hfdcan2.Instance = FDCAN2;
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  hfdcan2.Init.FrameFormat = FDCAN_FRAME_FD_BRS;
#else
  hfdcan2.Init.FrameFormat = FDCAN_FRAME_CLASSIC;
#endif
  hfdcan2.Init.Mode = FDCAN_MODE_NORMAL;
  hfdcan2.Init.AutoRetransmission = ENABLE;
  hfdcan2.Init.TransmitPause = ENABLE;
  hfdcan2.Init.ProtocolException = ENABLE;
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  hfdcan2.Init.NominalPrescaler = 1;
  hfdcan2.Init.NominalSyncJumpWidth = 30;
  hfdcan2.Init.NominalTimeSeg1 = 89;
  hfdcan2.Init.NominalTimeSeg2 = 30;
#else
  hfdcan2.Init.NominalPrescaler = 6;
  hfdcan2.Init.NominalSyncJumpWidth = 4;
  hfdcan2.Init.NominalTimeSeg1 = 15;
  hfdcan2.Init.NominalTimeSeg2 = 4;
#endif
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  hfdcan2.Init.DataPrescaler = 2;
  hfdcan2.Init.DataSyncJumpWidth = 2;
  hfdcan2.Init.DataTimeSeg1 = 9;
  hfdcan2.Init.DataTimeSeg2 = 2;
#else
  hfdcan2.Init.DataPrescaler = 1;
  hfdcan2.Init.DataSyncJumpWidth = 1;
  hfdcan2.Init.DataTimeSeg1 = 1;
  hfdcan2.Init.DataTimeSeg2 = 1;
#endif
  hfdcan2.Init.MessageRAMOffset = FDCAN2_MESSAGE_RAM_OFFSET;
  hfdcan2.Init.StdFiltersNbr = 1;
  hfdcan2.Init.ExtFiltersNbr = 0;
  hfdcan2.Init.RxFifo0ElmtsNbr = FDCAN2_RX_FIFO0_ELEMENTS;
  hfdcan2.Init.RxFifo0ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan2.Init.RxFifo1ElmtsNbr = 0;
  hfdcan2.Init.RxFifo1ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan2.Init.RxBuffersNbr = 0;
  hfdcan2.Init.RxBufferSize = FDCAN_DATA_BYTES_8;
  hfdcan2.Init.TxEventsNbr = FDCAN_TX_EVENT_ELEMENTS;
  hfdcan2.Init.TxBuffersNbr = 0;
  hfdcan2.Init.TxFifoQueueElmtsNbr = FDCAN2_TX_FIFO_ELEMENTS;
  hfdcan2.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
  hfdcan2.Init.TxElmtSize = FDCAN_DATA_BYTES_8;
  if (HAL_FDCAN_Init(&hfdcan2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN FDCAN2_Init 2 */

  /* USER CODE END FDCAN2_Init 2 */

}
/* FDCAN3 init function */
void MX_FDCAN3_Init(void)
{

  /* USER CODE BEGIN FDCAN3_Init 0 */

  /* USER CODE END FDCAN3_Init 0 */

  /* USER CODE BEGIN FDCAN3_Init 1 */

  /* USER CODE END FDCAN3_Init 1 */
  hfdcan3.Instance = FDCAN3;
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  hfdcan3.Init.FrameFormat = FDCAN_FRAME_FD_BRS;
#else
  hfdcan3.Init.FrameFormat = FDCAN_FRAME_CLASSIC;
#endif
  hfdcan3.Init.Mode = FDCAN_MODE_NORMAL;
  hfdcan3.Init.AutoRetransmission = ENABLE;
  hfdcan3.Init.TransmitPause = ENABLE;
  hfdcan3.Init.ProtocolException = ENABLE;
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  hfdcan3.Init.NominalPrescaler = 1;
  hfdcan3.Init.NominalSyncJumpWidth = 30;
  hfdcan3.Init.NominalTimeSeg1 = 89;
  hfdcan3.Init.NominalTimeSeg2 = 30;
  hfdcan3.Init.DataPrescaler = 2;
  hfdcan3.Init.DataSyncJumpWidth = 2;
  hfdcan3.Init.DataTimeSeg1 = 9;
  hfdcan3.Init.DataTimeSeg2 = 2;
#else
  hfdcan3.Init.NominalPrescaler = 6;
  hfdcan3.Init.NominalSyncJumpWidth = 4;
  hfdcan3.Init.NominalTimeSeg1 = 15;
  hfdcan3.Init.NominalTimeSeg2 = 4;
  hfdcan3.Init.DataPrescaler = 1;
  hfdcan3.Init.DataSyncJumpWidth = 1;
  hfdcan3.Init.DataTimeSeg1 = 1;
  hfdcan3.Init.DataTimeSeg2 = 1;
#endif
  hfdcan3.Init.MessageRAMOffset = FDCAN3_MESSAGE_RAM_OFFSET;
  hfdcan3.Init.StdFiltersNbr = 1;
  hfdcan3.Init.ExtFiltersNbr = 0;
  hfdcan3.Init.RxFifo0ElmtsNbr = FDCAN3_RX_FIFO0_ELEMENTS;
  hfdcan3.Init.RxFifo0ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan3.Init.RxFifo1ElmtsNbr = 0;
  hfdcan3.Init.RxFifo1ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan3.Init.RxBuffersNbr = 0;
  hfdcan3.Init.RxBufferSize = FDCAN_DATA_BYTES_8;
  hfdcan3.Init.TxEventsNbr = FDCAN_TX_EVENT_ELEMENTS;
  hfdcan3.Init.TxBuffersNbr = 0;
  hfdcan3.Init.TxFifoQueueElmtsNbr = FDCAN3_TX_FIFO_ELEMENTS;
  hfdcan3.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
  hfdcan3.Init.TxElmtSize = FDCAN_DATA_BYTES_8;
  if (HAL_FDCAN_Init(&hfdcan3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN FDCAN3_Init 2 */

  /* USER CODE END FDCAN3_Init 2 */

}

static uint32_t HAL_RCC_FDCAN_CLK_ENABLED=0;

void HAL_FDCAN_MspInit(FDCAN_HandleTypeDef* fdcanHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  if(fdcanHandle->Instance==FDCAN1)
  {
  /* USER CODE BEGIN FDCAN1_MspInit 0 */

  /* USER CODE END FDCAN1_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    PeriphClkInitStruct.FdcanClockSelection = RCC_FDCANCLKSOURCE_PLL;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* FDCAN1 clock enable */
    HAL_RCC_FDCAN_CLK_ENABLED++;
    if(HAL_RCC_FDCAN_CLK_ENABLED==1){
      __HAL_RCC_FDCAN_CLK_ENABLE();
    }

    __HAL_RCC_GPIOD_CLK_ENABLE();
    /**FDCAN1 GPIO Configuration
    PD0     ------> FDCAN1_RX
    PD1     ------> FDCAN1_TX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF9_FDCAN1;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

    /* FDCAN1 interrupt Init */
    HAL_NVIC_SetPriority(FDCAN1_IT0_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN1_IT0_IRQn);
    HAL_NVIC_SetPriority(FDCAN1_IT1_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN1_IT1_IRQn);
  /* USER CODE BEGIN FDCAN1_MspInit 1 */

  /* USER CODE END FDCAN1_MspInit 1 */
  }
  else if(fdcanHandle->Instance==FDCAN2)
  {
  /* USER CODE BEGIN FDCAN2_MspInit 0 */

  /* USER CODE END FDCAN2_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    PeriphClkInitStruct.FdcanClockSelection = RCC_FDCANCLKSOURCE_PLL;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* FDCAN2 clock enable */
    HAL_RCC_FDCAN_CLK_ENABLED++;
    if(HAL_RCC_FDCAN_CLK_ENABLED==1){
      __HAL_RCC_FDCAN_CLK_ENABLE();
    }

    __HAL_RCC_GPIOB_CLK_ENABLE();
    /**FDCAN2 GPIO Configuration
    PB5     ------> FDCAN2_RX
    PB6     ------> FDCAN2_TX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_5|GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF9_FDCAN2;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /* FDCAN2 interrupt Init */
    HAL_NVIC_SetPriority(FDCAN2_IT0_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN2_IT0_IRQn);
    HAL_NVIC_SetPriority(FDCAN2_IT1_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN2_IT1_IRQn);
  /* USER CODE BEGIN FDCAN2_MspInit 1 */

  /* USER CODE END FDCAN2_MspInit 1 */
  }
  else if(fdcanHandle->Instance==FDCAN3)
  {
  /* USER CODE BEGIN FDCAN3_MspInit 0 */

  /* USER CODE END FDCAN3_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    PeriphClkInitStruct.FdcanClockSelection = RCC_FDCANCLKSOURCE_PLL;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* FDCAN3 clock enable */
    HAL_RCC_FDCAN_CLK_ENABLED++;
    if(HAL_RCC_FDCAN_CLK_ENABLED==1){
      __HAL_RCC_FDCAN_CLK_ENABLE();
    }

    __HAL_RCC_GPIOD_CLK_ENABLE();
    /**FDCAN3 GPIO Configuration
    PD12     ------> FDCAN3_RX
    PD13     ------> FDCAN3_TX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_12|GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF5_FDCAN3;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

    /* FDCAN3 interrupt Init */
    HAL_NVIC_SetPriority(FDCAN3_IT0_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN3_IT0_IRQn);
    HAL_NVIC_SetPriority(FDCAN3_IT1_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN3_IT1_IRQn);
  /* USER CODE BEGIN FDCAN3_MspInit 1 */

  /* USER CODE END FDCAN3_MspInit 1 */
  }
}

void HAL_FDCAN_MspDeInit(FDCAN_HandleTypeDef* fdcanHandle)
{

  if(fdcanHandle->Instance==FDCAN1)
  {
  /* USER CODE BEGIN FDCAN1_MspDeInit 0 */

  /* USER CODE END FDCAN1_MspDeInit 0 */
    /* Peripheral clock disable */
    HAL_RCC_FDCAN_CLK_ENABLED--;
    if(HAL_RCC_FDCAN_CLK_ENABLED==0){
      __HAL_RCC_FDCAN_CLK_DISABLE();
    }

    /**FDCAN1 GPIO Configuration
    PD0     ------> FDCAN1_RX
    PD1     ------> FDCAN1_TX
    */
    HAL_GPIO_DeInit(GPIOD, GPIO_PIN_0|GPIO_PIN_1);

    /* FDCAN1 interrupt Deinit */
    HAL_NVIC_DisableIRQ(FDCAN1_IT0_IRQn);
    HAL_NVIC_DisableIRQ(FDCAN1_IT1_IRQn);
  /* USER CODE BEGIN FDCAN1_MspDeInit 1 */

  /* USER CODE END FDCAN1_MspDeInit 1 */
  }
  else if(fdcanHandle->Instance==FDCAN2)
  {
  /* USER CODE BEGIN FDCAN2_MspDeInit 0 */

  /* USER CODE END FDCAN2_MspDeInit 0 */
    /* Peripheral clock disable */
    HAL_RCC_FDCAN_CLK_ENABLED--;
    if(HAL_RCC_FDCAN_CLK_ENABLED==0){
      __HAL_RCC_FDCAN_CLK_DISABLE();
    }

    /**FDCAN2 GPIO Configuration
    PB5     ------> FDCAN2_RX
    PB6     ------> FDCAN2_TX
    */
    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_5|GPIO_PIN_6);

    /* FDCAN2 interrupt Deinit */
    HAL_NVIC_DisableIRQ(FDCAN2_IT0_IRQn);
    HAL_NVIC_DisableIRQ(FDCAN2_IT1_IRQn);
  /* USER CODE BEGIN FDCAN2_MspDeInit 1 */

  /* USER CODE END FDCAN2_MspDeInit 1 */
  }
  else if(fdcanHandle->Instance==FDCAN3)
  {
  /* USER CODE BEGIN FDCAN3_MspDeInit 0 */

  /* USER CODE END FDCAN3_MspDeInit 0 */
    /* Peripheral clock disable */
    HAL_RCC_FDCAN_CLK_ENABLED--;
    if(HAL_RCC_FDCAN_CLK_ENABLED==0){
      __HAL_RCC_FDCAN_CLK_DISABLE();
    }

    /**FDCAN3 GPIO Configuration
    PD12     ------> FDCAN3_RX
    PD13     ------> FDCAN3_TX
    */
    HAL_GPIO_DeInit(GPIOD, GPIO_PIN_12|GPIO_PIN_13);

    /* FDCAN3 interrupt Deinit */
    HAL_NVIC_DisableIRQ(FDCAN3_IT0_IRQn);
    HAL_NVIC_DisableIRQ(FDCAN3_IT1_IRQn);
  /* USER CODE BEGIN FDCAN3_MspDeInit 1 */

  /* USER CODE END FDCAN3_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */
