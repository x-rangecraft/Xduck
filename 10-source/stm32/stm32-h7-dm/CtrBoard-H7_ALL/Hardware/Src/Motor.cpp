/**
  ******************************************************************************
  * File Name          : Motor.cpp
  * Description        : DAMIAO  MIT mode motor interface
  ******************************************************************************
  */

#include "Motor.h"
#include "BSP_CAN.h"
#if H7DM_CAN_CAPTURE
#include "can_capture.h"
#include "motor_app.h"
#endif
#include "usbd_cdc_if.h"

#include <string.h>

Motor_t Motor[MOTOR_NUM];
/* Read-only SWD diagnostics; counters wrap naturally and never drive control. */
volatile unsigned int motor_mit_tx_count[MOTOR_NUM];
volatile unsigned int motor_feedback_rx_count[MOTOR_NUM];
static void MotorCanCallBack(unsigned char Port, unsigned int ID, unsigned char *Data);
static unsigned char g_motorTxFailure;

#define MOTOR_SPECIAL_NONE       (0x00U)
#define MOTOR_SPECIAL_ENABLE     (0xFCU)
#define MOTOR_SPECIAL_DISABLE    (0xFDU)
#define MOTOR_SPECIAL_SAVE_ZERO  (0xFEU)
#define MOTOR_SPECIAL_CLEAR_ERR  (0xFBU)

#define MOTOR_REGISTER_FRAME_ID          (0x7FFU)
#define MOTOR_REGISTER_CMD_READ          (0x33U)
#define MOTOR_REGISTER_CMD_WRITE         (0x55U)
#define MOTOR_REGISTER_CMD_SAVE          (0xAAU)
#define MOTOR_REGISTER_SAVE_ARG          (0x01U)
#define MOTOR_REGISTER_TIMEOUT_ID        (9U)
#define MOTOR_REGISTER_RESPONSE_WAIT_MS  (20U)
#define MOTOR_REGISTER_MAX_RETRIES       (20U)
#define MOTOR_REGISTER_RETRY_DELAY_MS    (2U)
#define MOTOR_REGISTER_SAVE_DELAY_MS     (100U)

static volatile unsigned char g_motorRegWaitActive;
static volatile unsigned char g_motorRegResponseSeen;
static unsigned char g_motorRegExpectedPort;
static unsigned short g_motorRegExpectedResponseId;
static unsigned char g_motorRegExpectedMotorId;
static unsigned char g_motorRegExpectedCmd;
static unsigned char g_motorRegExpectedReg;
static unsigned char g_motorRegResponseData[8];
static volatile unsigned char g_motorRegLastRxValid;
static volatile unsigned int g_motorRegLastRxId;
static volatile unsigned char g_motorRegLastRxData[8];

static void Motor_U32ToLe(unsigned int Value, unsigned char Out[4])
{
  Out[0] = (unsigned char)Value;
  Out[1] = (unsigned char)(Value >> 8);
  Out[2] = (unsigned char)(Value >> 16);
  Out[3] = (unsigned char)(Value >> 24);
}

static unsigned int Motor_LeToU32(const unsigned char Data[4])
{
  return ((unsigned int)Data[0]) |
         ((unsigned int)Data[1] << 8) |
         ((unsigned int)Data[2] << 16) |
         ((unsigned int)Data[3] << 24);
}

static void Motor_DelayMs(unsigned int DelayMs)
{
  if (xTaskGetSchedulerState() == taskSCHEDULER_RUNNING) {
    osDelay(DelayMs);
  } else {
    HAL_Delay(DelayMs);
  }
}

static unsigned char Motor_PortUsesFd(unsigned char Port)
{
#if defined(DM_CAN_FD_MOTOR) && (DM_CAN_FD_MOTOR != 0)
  return (unsigned char)((Port >= 1U && Port <= 3U) ? 1U : 0U);
#else
  (void)Port;
  return 0U;
#endif
}

static void Motor_RegisterPrepareWait(unsigned char Port,
                                      unsigned short ResponseId,
                                      unsigned char MotorId,
                                      unsigned char Cmd,
                                      unsigned char Reg)
{
  __disable_irq();
  g_motorRegExpectedPort = Port;
  g_motorRegExpectedResponseId = ResponseId;
  g_motorRegExpectedMotorId = MotorId;
  g_motorRegExpectedCmd = Cmd;
  g_motorRegExpectedReg = Reg;
  g_motorRegResponseSeen = 0U;
  g_motorRegWaitActive = 1U;
  __enable_irq();
}

static unsigned char Motor_RegisterFetchResponse(unsigned char Response[8])
{
  unsigned char seen;

  __disable_irq();
  seen = g_motorRegResponseSeen;
  if (seen != 0U && Response != 0) {
    memcpy(Response, g_motorRegResponseData, 8U);
  }
  g_motorRegWaitActive = 0U;
  g_motorRegResponseSeen = 0U;
  __enable_irq();

  return seen;
}

static void Motor_RegisterClearLastRx(void)
{
  __disable_irq();
  g_motorRegLastRxValid = 0U;
  g_motorRegLastRxId = 0U;
  __enable_irq();
}

static void Motor_RegisterStoreLastRx(unsigned char Port, unsigned int ID, unsigned char *Data)
{
  unsigned char index;

  if (Port != g_motorRegExpectedPort || Data == 0) {
    return;
  }

  g_motorRegLastRxId = ID & 0x7FFU;
  for (index = 0U; index < 8U; index++) {
    g_motorRegLastRxData[index] = Data[index];
  }
  g_motorRegLastRxValid = 1U;
}

static unsigned char Motor_RegisterCopyLastRx(unsigned int *ID, unsigned char Data[8])
{
  unsigned char index;
  unsigned char valid;

  __disable_irq();
  valid = g_motorRegLastRxValid;
  if (valid != 0U) {
    if (ID != 0) {
      *ID = g_motorRegLastRxId;
    }
    if (Data != 0) {
      for (index = 0U; index < 8U; index++) {
        Data[index] = g_motorRegLastRxData[index];
      }
    }
  }
  __enable_irq();

  return valid;
}

static unsigned char Motor_RegisterOnRx(unsigned char Port, unsigned int ID, unsigned char *Data)
{
  if (Data == 0 || g_motorRegWaitActive == 0U) {
    return 0U;
  }
  Motor_RegisterStoreLastRx(Port, ID, Data);
  if (Port != g_motorRegExpectedPort ||
      (unsigned short)(ID & 0x7FFU) != g_motorRegExpectedResponseId ||
      Data[0] != g_motorRegExpectedMotorId ||
      Data[2] != g_motorRegExpectedCmd ||
      Data[3] != g_motorRegExpectedReg) {
    return 0U;
  }

  memcpy(g_motorRegResponseData, Data, 8U);
  g_motorRegResponseSeen = 1U;
  return 1U;
}

static void Motor_RegisterLogTimeout(unsigned char Port,
                                     unsigned short CANID,
                                     unsigned short MasterID,
                                     unsigned char Cmd,
                                     unsigned char Reg)
{
  unsigned int last_id = 0U;
  unsigned char last_data[8];

  if (Motor_RegisterCopyLastRx(&last_id, last_data) != 0U) {
    (void)USB_LogPrintf("[DM_REG_TIMEOUT] p=%u motor=0x%02X exp=0x%03X cmd=0x%02X reg=0x%02X last_id=0x%03X last=%02X %02X %02X %02X %02X %02X %02X %02X",
                        (unsigned int)Port,
                        (unsigned int)CANID,
                        (unsigned int)MasterID,
                        (unsigned int)Cmd,
                        (unsigned int)Reg,
                        last_id,
                        (unsigned int)last_data[0],
                        (unsigned int)last_data[1],
                        (unsigned int)last_data[2],
                        (unsigned int)last_data[3],
                        (unsigned int)last_data[4],
                        (unsigned int)last_data[5],
                        (unsigned int)last_data[6],
                        (unsigned int)last_data[7]);
  } else {
    (void)USB_LogPrintf("[DM_REG_TIMEOUT] p=%u motor=0x%02X exp=0x%03X cmd=0x%02X reg=0x%02X last=none",
                        (unsigned int)Port,
                        (unsigned int)CANID,
                        (unsigned int)MasterID,
                        (unsigned int)Cmd,
                        (unsigned int)Reg);
  }
}

static unsigned char Motor_RegisterSendFrame(unsigned char Port,
                                             unsigned char Retry,
                                             unsigned char TxData[8])
{
  if (Motor_PortUsesFd(Port) == 0U) {
    return BSP_CAN_TrySendStandardDataMessage(Port, MOTOR_REGISTER_FRAME_ID, TxData);
  }

  (void)Retry;
  const unsigned char length = TxData[2] == MOTOR_REGISTER_CMD_WRITE ? 8U : 4U;
  return BSP_CAN_TrySendStandardFdFrame(Port, MOTOR_REGISTER_FRAME_ID, TxData, length, 1U);
}

static float Motor_LimitFloat(float Value, float Min, float Max)
{
  if (Value < Min) {
    return Min;
  }
  if (Value > Max) {
    return Max;
  }
  return Value;
}

static unsigned int Motor_FloatToUint(float x, float x_min, float x_max, unsigned char bits)
{
  const unsigned int MaxInt = (1U << bits) - 1U;
  float span;

  if (x_max <= x_min) {
    return 0U;
  }

  span = x_max - x_min;
  x = Motor_LimitFloat(x, x_min, x_max);
  return (unsigned int)((x - x_min) * ((float)MaxInt) / span);
}

static float Motor_UintToFloat(unsigned int x_int, float x_min, float x_max, unsigned char bits)
{
  const unsigned int MaxInt = (1U << bits) - 1U;
  float span;

  if (x_max <= x_min) {
    return x_min;
  }

  span = x_max - x_min;
  return ((float)x_int) * span / ((float)MaxInt) + x_min;
}

static unsigned char Motor_SendSpecialFrame(unsigned char Port, unsigned short CANID, unsigned char Tail)
{
  unsigned char TxData[8];

  memset(TxData, 0xFF, sizeof(TxData));
  TxData[7] = Tail;
  if (((Motor_PortUsesFd(Port) != 0U) &&
       (BSP_CAN_TrySendStandardFdDataMessage(Port, CANID, TxData, 1U) == 0U)) ||
      ((Motor_PortUsesFd(Port) == 0U) &&
       (BSP_CAN_TrySendStandardDataMessage(Port, CANID, TxData) == 0U))) {
    g_motorTxFailure = 1U;
    return 0U;
  }
  return 1U;
}

static int Motor_FindCodeByFeedback(unsigned char Port, unsigned int MasterID, unsigned char FeedbackID)
{
  int AnyCode = -1;
  int ExactCode = -1;
  unsigned char AnyCount = 0U;
  unsigned char ExactCount = 0U;
  unsigned char Code;

  for (Code = 0U; Code < MOTOR_NUM; Code++) {
    if (Motor[Code].IsConfigured() == 0U) {
      continue;
    }
    if (Motor[Code].GetPort() != Port) {
      continue;
    }
    if (Motor[Code].GetMasterID() != MasterID) {
      continue;
    }

    AnyCode = Code;
    AnyCount++;

    if ((Motor[Code].GetCANID() & 0x0FU) == FeedbackID) {
      ExactCode = Code;
      ExactCount++;
    }
  }

  if (ExactCount == 1U) {
    return ExactCode;
  }
  if (ExactCount > 1U) {
    return -1;
  }
  if (AnyCount == 1U) {
    return AnyCode;
  }
  return -1;
}

static unsigned char Motor_NormalizeFeedbackState(unsigned char Code, unsigned int ID, unsigned char RawState, const unsigned char *Data)
{
  /* I2RT states are identical for classic and FD frames. Never promote a
   * disabled or undocumented state (including captured state 5) to enabled. */
  (void)Code;
  (void)ID;
  (void)Data;
  return RawState;
}

Motor_t::Motor_t()
{
  Private_Port = 0U;
  Private_CANID = 0U;
  Private_MasterID = 0U;
  Private_UseFlag = 0U;
  Private_RunFlag = 0U;
  Private_SpecialCommand = 0U;

  Private_Position = 0.0f;
  Private_Speed = 0.0f;
  Private_Torque = 0.0f;
  Private_MosTemp = 0U;
  Private_RotorTemp = 0U;
  Private_State = Motor_State_Disable;
  Private_LastRxTick = 0U;

  Private_Position_Set = 0.0f;
  Private_Speed_Set = 0.0f;
  Private_Kp_Set = 0.0f;
  Private_Kd_Set = 0.0f;
  Private_Torque_Set = 0.0f;

  Private_Position_Min = MOTOR_MIT_P_MIN_DEFAULT;
  Private_Position_Max = MOTOR_MIT_P_MAX_DEFAULT;
  Private_Speed_Min = MOTOR_MIT_V_MIN_DEFAULT;
  Private_Speed_Max = MOTOR_MIT_V_MAX_DEFAULT;
  Private_Torque_Min = MOTOR_MIT_T_MIN_DEFAULT;
  Private_Torque_Max = MOTOR_MIT_T_MAX_DEFAULT;
}

void Motor_t::SetConfig(unsigned char Port, unsigned short CANID, unsigned short MasterID)
{
  SetPort(Port);
  SetCANID(CANID);
  SetMasterID(MasterID);
  if ((Private_Port >= 1U && Private_Port <= 3U) && Private_CANID <= 0x7FFU && Private_MasterID <= 0x7FFU) {
    Private_UseFlag = 1U;
  } else {
    Private_UseFlag = 0U;
  }
}

void Motor_t::SetPort(unsigned char Port)
{
  Private_Port = Port;
}

void Motor_t::SetCANID(unsigned short CANID)
{
  Private_CANID = CANID;
}

void Motor_t::SetMasterID(unsigned short MasterID)
{
  Private_MasterID = MasterID;
}

unsigned char Motor_t::GetPort(void)
{
  return Private_Port;
}

unsigned short Motor_t::GetCANID(void)
{
  return Private_CANID;
}

unsigned short Motor_t::GetMasterID(void)
{
  return Private_MasterID;
}

unsigned char Motor_t::IsConfigured(void)
{
  return Private_UseFlag;
}

void Motor_t::SetMITRange(float PositionMin, float PositionMax, float SpeedMin, float SpeedMax, float TorqueMin, float TorqueMax)
{
  if (PositionMax > PositionMin) {
    Private_Position_Min = PositionMin;
    Private_Position_Max = PositionMax;
  }
  if (SpeedMax > SpeedMin) {
    Private_Speed_Min = SpeedMin;
    Private_Speed_Max = SpeedMax;
  }
  if (TorqueMax > TorqueMin) {
    Private_Torque_Min = TorqueMin;
    Private_Torque_Max = TorqueMax;
  }
}

void Motor_t::SetMITFullRange(float PositionMin, float PositionMax,
                              float SpeedMin, float SpeedMax,
                              float TorqueMin, float TorqueMax,
                              float KpMin, float KpMax,
                              float KdMin, float KdMax)
{
  SetMITRange(PositionMin, PositionMax, SpeedMin, SpeedMax, TorqueMin, TorqueMax);
  if (KpMax > KpMin) {
    Private_Kp_Min = KpMin;
    Private_Kp_Max = KpMax;
  }
  if (KdMax > KdMin) {
    Private_Kd_Min = KdMin;
    Private_Kd_Max = KdMax;
  }
}

void Motor_t::SetMITCommand(float Position, float Speed, float Kp, float Kd, float Torque)
{
  Private_Position_Set = Position;
  Private_Speed_Set = Speed;
  Private_Kp_Set = Kp;
  Private_Kd_Set = Kd;
  Private_Torque_Set = Torque;
}

void Motor_t::SetPosition(float Position)
{
  Private_Position_Set = Position;
}

void Motor_t::SetSpeed(float Speed)
{
  Private_Speed_Set = Speed;
}

void Motor_t::SetKp(float Kp)
{
  Private_Kp_Set = Kp;
}

void Motor_t::SetKd(float Kd)
{
  Private_Kd_Set = Kd;
}

void Motor_t::SetTorque(float Torque)
{
  Private_Torque_Set = Torque;
}

void Motor_t::ClearCommand(void)
{
  Private_Position_Set = 0.0f;
  Private_Speed_Set = 0.0f;
  Private_Kp_Set = 0.0f;
  Private_Kd_Set = 0.0f;
  Private_Torque_Set = 0.0f;
}

void Motor_t::Enable(void)
{
  if (Private_UseFlag == 0U) {
    return;
  }

  Private_SpecialCommand = MOTOR_SPECIAL_ENABLE;
  Private_RunFlag = 0U;
}

void Motor_t::EnablePure(void)
{
  if (Private_UseFlag == 0U) {
    return;
  }
  if (Private_Port < 1U || Private_Port > 3U) {
    return;
  }

  Private_RunFlag = 0U;
  Private_SpecialCommand = MOTOR_SPECIAL_NONE;

  BSP_CAN_ClearTxMailboxes(Private_Port);
  Motor_SendSpecialFrame(Private_Port, Private_CANID, MOTOR_SPECIAL_ENABLE);
}

void Motor_t::SendEnableFrame(void)
{
  if (Private_UseFlag == 0U) {
    return;
  }
  if (Private_Port < 1U || Private_Port > 3U) {
    return;
  }
  (void)Motor_SendSpecialFrame(Private_Port, Private_CANID, MOTOR_SPECIAL_ENABLE);
}

void Motor_t::SendDisableFrame(void)
{
  if (Private_UseFlag == 0U) {
    return;
  }
  if (Private_Port < 1U || Private_Port > 3U) {
    return;
  }
  (void)Motor_SendSpecialFrame(Private_Port, Private_CANID, MOTOR_SPECIAL_DISABLE);
}

void Motor_t::SendClearErrorFrame(void)
{
  if (Private_UseFlag == 0U) {
    return;
  }
  if (Private_Port < 1U || Private_Port > 3U) {
    return;
  }
  (void)Motor_SendSpecialFrame(Private_Port, Private_CANID, MOTOR_SPECIAL_CLEAR_ERR);
}

void Motor_t::SetRunFlag(unsigned char RunFlag)
{
  Private_RunFlag = (unsigned char)((RunFlag != 0U) ? 1U : 0U);
}

void Motor_t::Disable(void)
{
  Private_RunFlag = 0U;
  Private_SpecialCommand = MOTOR_SPECIAL_DISABLE;
}

void Motor_t::SaveZero(void)
{
  Private_SpecialCommand = MOTOR_SPECIAL_SAVE_ZERO;
}

void Motor_t::ClearError(void)
{
  Private_SpecialCommand = MOTOR_SPECIAL_CLEAR_ERR;
}

float Motor_t::GetPosition(void)
{
  return Private_Position;
}

float Motor_t::GetSpeed(void)
{
  return Private_Speed;
}

float Motor_t::GetTorque(void)
{
  return Private_Torque;
}

unsigned char Motor_t::GetMosTemp(void)
{
  return Private_MosTemp;
}

unsigned char Motor_t::GetRotorTemp(void)
{
  return Private_RotorTemp;
}

unsigned char Motor_t::GetState(void)
{
  return Private_State;
}

unsigned char Motor_t::IsEnable(void)
{
  return (unsigned char)(Private_State == Motor_State_Enable);
}

unsigned char Motor_t::IsOnline(unsigned int Timeout_ms)
{
  TickType_t NowTick;
  TickType_t TimeoutTick;

  if (Private_LastRxTick == 0U) {
    return 0U;
  }

  NowTick = xTaskGetTickCount();
  TimeoutTick = pdMS_TO_TICKS(Timeout_ms);
  return (unsigned char)((NowTick - Private_LastRxTick) <= TimeoutTick);
}

static unsigned char Motor_RegisterRequest(Motor_t *MotorObj,
                                           unsigned char Cmd,
                                           unsigned char Reg,
                                           unsigned int Value,
                                           unsigned char Response[8])
{
  unsigned char retry;
  unsigned char TxData[8];
  unsigned char Port;
  unsigned short CANID;
  unsigned short MasterID;
  unsigned char UseFlag;

  if (MotorObj == 0 || Response == 0) {
    return 0U;
  }

  Port = MotorObj->GetPort();
  CANID = MotorObj->GetCANID();
  MasterID = MotorObj->GetMasterID();
  UseFlag = MotorObj->IsConfigured();
  if (UseFlag == 0U || Port < 1U || Port > 3U || CANID > 0xFFU) {
    return 0U;
  }

  TxData[0] = (unsigned char)CANID;
  TxData[1] = 0U;
  TxData[2] = Cmd;
  TxData[3] = Reg;
  Motor_U32ToLe(Value, &TxData[4]);
  Motor_RegisterClearLastRx();

  for (retry = 0U; retry < MOTOR_REGISTER_MAX_RETRIES; retry++) {
    TickType_t start_tick;

    Motor_RegisterPrepareWait(Port, MasterID, (unsigned char)CANID, Cmd, Reg);
    if (Motor_RegisterSendFrame(Port, retry, TxData) == 0U) {
      (void)Motor_RegisterFetchResponse(0);
      g_motorTxFailure = 1U;
      return 0U;
    }

    start_tick = xTaskGetTickCount();
    while ((xTaskGetTickCount() - start_tick) <= pdMS_TO_TICKS(MOTOR_REGISTER_RESPONSE_WAIT_MS)) {
      if (g_motorRegResponseSeen != 0U) {
        if (Motor_RegisterFetchResponse(Response) != 0U) {
          if (Cmd == MOTOR_REGISTER_CMD_WRITE && Motor_LeToU32(&Response[4]) != Value) {
            Motor_DelayMs(MOTOR_REGISTER_RETRY_DELAY_MS);
            break;
          }
          return 1U;
        }
      }
      Motor_DelayMs(1U);
    }

    (void)Motor_RegisterFetchResponse(0);
    Motor_DelayMs(MOTOR_REGISTER_RETRY_DELAY_MS);
  }

  Motor_RegisterLogTimeout(Port, CANID, MasterID, Cmd, Reg);
  return 0U;
}

static unsigned char Motor_RegisterRead(Motor_t *MotorObj, unsigned char Reg, unsigned int *OutValue)
{
  unsigned char Response[8];

  if (OutValue == 0) {
    return 0U;
  }
  if (Motor_RegisterRequest(MotorObj, MOTOR_REGISTER_CMD_READ, Reg, 0U, Response) == 0U) {
    return 0U;
  }
  *OutValue = Motor_LeToU32(&Response[4]);
  return 1U;
}

static unsigned char Motor_RegisterWriteVerified(Motor_t *MotorObj,
                                                 unsigned char Reg,
                                                 unsigned int Value,
                                                 unsigned char SaveAfterWrite)
{
  unsigned char Response[8];
  unsigned int VerifiedValue = 0U;

  if (Motor_RegisterRequest(MotorObj, MOTOR_REGISTER_CMD_WRITE, Reg, Value, Response) == 0U) {
    return 0U;
  }
  if (Motor_RegisterRead(MotorObj, Reg, &VerifiedValue) == 0U || VerifiedValue != Value) {
    return 0U;
  }
  if (SaveAfterWrite != 0U) {
    if (Motor_RegisterRequest(MotorObj,
                              MOTOR_REGISTER_CMD_SAVE,
                              MOTOR_REGISTER_SAVE_ARG,
                              0U,
                              Response) == 0U) {
      return 0U;
    }
    Motor_DelayMs(MOTOR_REGISTER_SAVE_DELAY_MS);
    if (Motor_RegisterRead(MotorObj, Reg, &VerifiedValue) == 0U || VerifiedValue != Value) {
      return 0U;
    }
  }
  return 1U;
}

unsigned char Motor_t::SetupTimeout(unsigned int TimeoutMs)
{
  unsigned int CurrentValue = 0U;
  unsigned int VerifiedValue = 0U;

  if (Motor_RegisterRead(this, MOTOR_REGISTER_TIMEOUT_ID, &CurrentValue) == 0U) {
    return 0U;
  }
  if (CurrentValue == TimeoutMs) {
    return 1U;
  }
  if (Motor_RegisterWriteVerified(this, MOTOR_REGISTER_TIMEOUT_ID, TimeoutMs, 1U) == 0U) {
    return 0U;
  }
  if (Motor_RegisterRead(this, MOTOR_REGISTER_TIMEOUT_ID, &VerifiedValue) == 0U) {
    return 0U;
  }
  return (unsigned char)((VerifiedValue == TimeoutMs) ? 1U : 0U);
}

unsigned char Motor_t::ReadRegister(unsigned char Reg, unsigned int *OutValue)
{
  return Motor_RegisterRead(this, Reg, OutValue);
}

unsigned char Motor_t::WriteRegister(unsigned char Reg, unsigned int Value, unsigned char SaveAfterWrite)
{
  return Motor_RegisterWriteVerified(this, Reg, Value, SaveAfterWrite);
}

TickType_t Motor_t::GetLastRxTick(void)
{
  return Private_LastRxTick;
}

void Motor_CAN_Back(unsigned char Code, unsigned char Err, unsigned char *RxData)
{
  unsigned short PositionInt;
  unsigned short SpeedInt;
  unsigned short TorqueInt;
  TickType_t now_tick;

  motor_feedback_rx_count[Code]++;

  PositionInt = ((unsigned short)RxData[1] << 8) | RxData[2];
  SpeedInt = ((unsigned short)RxData[3] << 4) | ((unsigned short)RxData[4] >> 4);
  TorqueInt = (((unsigned short)RxData[4] & 0x000FU) << 8) | RxData[5];

  Motor[Code].Private_State = Err;
  Motor[Code].Private_Position = Motor_UintToFloat(PositionInt, Motor[Code].Private_Position_Min, Motor[Code].Private_Position_Max, 16U);
  Motor[Code].Private_Speed = Motor_UintToFloat(SpeedInt, Motor[Code].Private_Speed_Min, Motor[Code].Private_Speed_Max, 12U);
  Motor[Code].Private_Torque = Motor_UintToFloat(TorqueInt, Motor[Code].Private_Torque_Min, Motor[Code].Private_Torque_Max, 12U);
  Motor[Code].Private_MosTemp = RxData[6];
  Motor[Code].Private_RotorTemp = RxData[7];
  now_tick = xTaskGetTickCountFromISR();
  Motor[Code].Private_LastRxTick = now_tick;
}

unsigned char Motor_CAN_Send(unsigned char Code)
{
  unsigned char TxData[8];
  unsigned int PositionInt;
  unsigned int SpeedInt;
  unsigned int KpInt;
  unsigned int KdInt;
  unsigned int TorqueInt;
  unsigned char Port;
  unsigned short CANID;
  unsigned char UseFlag;
  unsigned char RunFlag;
  unsigned char State;
  unsigned char SpecialCommand;
  float PositionSet;
  float SpeedSet;
  float KpSet;
  float KdSet;
  float TorqueSet;
  float PositionMin;
  float PositionMax;
  float SpeedMin;
  float SpeedMax;
  float TorqueMin;
  float TorqueMax;

  taskENTER_CRITICAL();
  UseFlag = Motor[Code].Private_UseFlag;
  Port = Motor[Code].Private_Port;
  CANID = Motor[Code].Private_CANID;
  SpecialCommand = Motor[Code].Private_SpecialCommand;
  RunFlag = Motor[Code].Private_RunFlag;
  State = Motor[Code].Private_State;
  PositionSet = Motor[Code].Private_Position_Set;
  SpeedSet = Motor[Code].Private_Speed_Set;
  KpSet = Motor[Code].Private_Kp_Set;
  KdSet = Motor[Code].Private_Kd_Set;
  TorqueSet = Motor[Code].Private_Torque_Set;
  PositionMin = Motor[Code].Private_Position_Min;
  PositionMax = Motor[Code].Private_Position_Max;
  SpeedMin = Motor[Code].Private_Speed_Min;
  SpeedMax = Motor[Code].Private_Speed_Max;
  TorqueMin = Motor[Code].Private_Torque_Min;
  TorqueMax = Motor[Code].Private_Torque_Max;
  taskEXIT_CRITICAL();

  if (UseFlag == 0U) {
    return 0U;
  }
  if (Port < 1U || Port > 3U) {
    return 0U;
  }

  if (SpecialCommand != MOTOR_SPECIAL_NONE) {
    taskENTER_CRITICAL();
    if (Motor[Code].Private_SpecialCommand == SpecialCommand) {
      Motor[Code].Private_SpecialCommand = MOTOR_SPECIAL_NONE;
    }
    taskEXIT_CRITICAL();
    return Motor_SendSpecialFrame(Port, CANID, SpecialCommand);
  }

  if (RunFlag == 0U) {
    return 0U;
  }
  if (State != Motor_State_Enable) {
    return 0U;
  }

  PositionInt = Motor_FloatToUint(PositionSet, PositionMin, PositionMax, 16U);
  SpeedInt = Motor_FloatToUint(SpeedSet, SpeedMin, SpeedMax, 12U);
  KpInt = Motor_FloatToUint(Motor_LimitFloat(KpSet, MOTOR_MIT_KP_MIN, MOTOR_MIT_KP_MAX), MOTOR_MIT_KP_MIN, MOTOR_MIT_KP_MAX, 12U);
  KdInt = Motor_FloatToUint(Motor_LimitFloat(KdSet, MOTOR_MIT_KD_MIN, MOTOR_MIT_KD_MAX), MOTOR_MIT_KD_MIN, MOTOR_MIT_KD_MAX, 12U);
  TorqueInt = Motor_FloatToUint(TorqueSet, TorqueMin, TorqueMax, 12U);

  TxData[0] = (unsigned char)(PositionInt >> 8);
  TxData[1] = (unsigned char)(PositionInt);
  TxData[2] = (unsigned char)(SpeedInt >> 4);
  TxData[3] = (unsigned char)(((SpeedInt & 0x0FU) << 4) | (KpInt >> 8));
  TxData[4] = (unsigned char)(KpInt);
  TxData[5] = (unsigned char)(KdInt >> 4);
  TxData[6] = (unsigned char)(((KdInt & 0x0FU) << 4) | (TorqueInt >> 8));
  TxData[7] = (unsigned char)(TorqueInt);

  if (((Motor_PortUsesFd(Port) != 0U) &&
       (BSP_CAN_TrySendStandardFdDataMessage(Port, CANID, TxData, 1U) == 0U)) ||
      ((Motor_PortUsesFd(Port) == 0U) &&
       (BSP_CAN_TrySendStandardDataMessage(Port, CANID, TxData) == 0U))) {
    g_motorTxFailure = 1U;
    return 0U;
  }
  motor_mit_tx_count[Code]++;
#if H7DM_CAN_CAPTURE
  CanCapture_Tx(Code, MotorApp_GetMotorLastCommandSeq(Code), TxData);
#endif
  return 1U;
}

unsigned char Motor_SendAllOnce(void)
{
  unsigned char Code;
  unsigned char SendCount = 0U;

  for (Code = 0U; Code < MOTOR_NUM; Code++) {
    if (Motor_CAN_Send(Code) != 0U) {
      SendCount++;
    }
  }
  return SendCount;
}

unsigned char Motor_SendListOnce(const unsigned char *Codes, unsigned char Count)
{
  unsigned char index;
  unsigned char send_count = 0U;

  if (Codes == 0) {
    return 0U;
  }

  for (index = 0U; index < Count; index++) {
    unsigned char code = Codes[index];
    if (code >= MOTOR_NUM) {
      continue;
    }
    if (Motor_CAN_Send(code) != 0U) {
      send_count++;
    }
  }
  return send_count;
}

unsigned char Motor_SendPortBudgeted(unsigned char Port, unsigned char *Cursor, unsigned char Budget)
{
  unsigned char scanned = 0U;
  unsigned char sent = 0U;
  unsigned char code;

  if (Cursor == 0 || Budget == 0U || Port < 1U || Port > 3U) {
    return 0U;
  }

  code = *Cursor;
  if (code >= MOTOR_NUM) {
    code = 0U;
  }

  while (scanned < MOTOR_NUM && sent < Budget) {
    unsigned char this_code = code;
    code++;
    if (code >= MOTOR_NUM) {
      code = 0U;
    }
    scanned++;

    if (Motor[this_code].IsConfigured() == 0U || Motor[this_code].GetPort() != Port) {
      continue;
    }
    if (Motor_CAN_Send(this_code) != 0U) {
      sent++;
    }
  }

  *Cursor = code;
  return sent;
}

void Motor_ClearTxFailure(void)
{
  g_motorTxFailure = 0U;
}

unsigned char Motor_HasTxFailure(void)
{
  return g_motorTxFailure;
}

void Motor_Task(void const *argument)
{
  unsigned char Code;
  unsigned char SendCount;

  (void)argument;
  osDelay(3500U);

  for (;;) {
    SendCount = 0U;
    for (Code = 0U; Code < MOTOR_NUM; Code++) {
      if (Motor_CAN_Send(Code) != 0U) {
        SendCount++;
        if ((SendCount & 0x03U) == 0U) {
          osDelay(1U);
        }
      }
    }

    if (SendCount == 0U || (SendCount & 0x03U) != 0U) {
      osDelay(1U);
    }
  }
}

static void MotorCanCallBack(unsigned char Port, unsigned int ID, unsigned char *Data)
{
  int Code;
  unsigned char FeedbackID;
  unsigned char Err;
#if H7DM_CAN_CAPTURE
  uint32_t received_cycle = DWT->CYCCNT;
#endif

  if (Motor_RegisterOnRx(Port, ID, Data) != 0U) {
    return;
  }

  FeedbackID = Data[0] & 0x0FU;
  Err = (Data[0] >> 4) & 0x0FU;
  Code = Motor_FindCodeByFeedback(Port, ID, FeedbackID);
  if (Code >= 0) {
#if H7DM_CAN_CAPTURE
    CanCapture_Rx((uint8_t)Code, received_cycle, Data);
#endif
    Err = Motor_NormalizeFeedbackState((unsigned char)Code, ID, Err, Data);
    Motor_CAN_Back((unsigned char)Code, Err, Data);
  }
}

void Motor_Init(void)
{
  BSP_CAN_AddRxCallBackFunction(MotorCanCallBack);
}

Motor_t *MotorPoint(unsigned char Code)
{
  if (Code >= MOTOR_NUM) {
    return &Motor[0];
  }
  return &Motor[Code];
}

/**********************************END OF FILE***********************************/
