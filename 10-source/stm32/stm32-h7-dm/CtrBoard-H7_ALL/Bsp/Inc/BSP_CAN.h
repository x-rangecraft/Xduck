/**
  ******************************************************************************
  * File Name          : BSP_CAN.h
  * Description        : FDCAN wrapper for motor control
  ******************************************************************************
  */

#ifndef __BSP_CAN_H
#define __BSP_CAN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include "fdcan.h"
#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"

typedef void (*BSP_CAN_FunCallBack_t)(unsigned char Port, unsigned int ID, unsigned char *Data);

void BSP_CAN_Init(void);
void BSP_CAN_ClearTxMailboxes(unsigned char Port);
unsigned char BSP_CAN_TrySendStandardDataMessage(unsigned char Port, unsigned int ID, unsigned char *Data);
unsigned char BSP_CAN_TrySendStandardFdDataMessage(unsigned char Port, unsigned int ID, unsigned char *Data, unsigned char BitRateSwitch);
void BSP_CAN_SendStandardDataMessage(unsigned char Port, unsigned int ID, unsigned char *Data);
void BSP_CAN_AddRxCallBackFunction(BSP_CAN_FunCallBack_t Function);
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
                                            unsigned int RxElementSize);

#ifdef __cplusplus
}
#endif

#endif

/**********************************END OF FILE***********************************/
