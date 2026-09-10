/**
  ******************************************************************************
  * File Name          : Motor.h
  * Description        : DAMIAO DM-J10010-2EC MIT mode motor interface
  ******************************************************************************
  */

#ifndef __MOTOR_H
#define __MOTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include "cmsis_os.h"
#include "FreeRTOS.h"
#include "task.h"

#define MOTOR_NUM                 24

#define MOTOR_MIT_P_MIN_DEFAULT   (-12.5f)
#define MOTOR_MIT_P_MAX_DEFAULT   ( 12.5f)
#define MOTOR_MIT_V_MIN_DEFAULT   (-45.0f)
#define MOTOR_MIT_V_MAX_DEFAULT   ( 45.0f)
#define MOTOR_MIT_T_MIN_DEFAULT   (-18.0f)
#define MOTOR_MIT_T_MAX_DEFAULT   ( 18.0f)
#define MOTOR_MIT_KP_MIN          (  0.0f)
#define MOTOR_MIT_KP_MAX          (500.0f)
#define MOTOR_MIT_KD_MIN          (  0.0f)
#define MOTOR_MIT_KD_MAX          (  5.0f)
#define MOTOR_FEEDBACK_ID_P16_OFFSET (0x10U)

#ifdef __cplusplus
typedef enum {
  Motor_State_Disable       = 0x0,
  Motor_State_Enable        = 0x1,
  Motor_State_OverVoltage   = 0x8,
  Motor_State_UnderVoltage  = 0x9,
  Motor_State_OverCurrent   = 0xA,
  Motor_State_MosOverTemp   = 0xB,
  Motor_State_RotorOverTemp = 0xC,
  Motor_State_CommLost      = 0xD,
  Motor_State_OverLoad      = 0xE,
} Motor_State_e;

class Motor_t {
private:
  unsigned char Private_Port;
  unsigned short Private_CANID;
  unsigned short Private_MasterID;
  unsigned char Private_UseFlag;
  unsigned char Private_RunFlag;
  unsigned char Private_SpecialCommand;

  float Private_Position;
  float Private_Speed;
  float Private_Torque;
  unsigned char Private_MosTemp;
  unsigned char Private_RotorTemp;
  unsigned char Private_State;
  TickType_t Private_LastRxTick;

  float Private_Position_Set;
  float Private_Speed_Set;
  float Private_Kp_Set;
  float Private_Kd_Set;
  float Private_Torque_Set;

  float Private_Position_Min;
  float Private_Position_Max;
  float Private_Speed_Min;
  float Private_Speed_Max;
  float Private_Torque_Min;
  float Private_Torque_Max;
  float Private_Kp_Min;
  float Private_Kp_Max;
  float Private_Kd_Min;
  float Private_Kd_Max;

public:
  Motor_t();

  void SetConfig(unsigned char Port, unsigned short CANID, unsigned short MasterID = 0);
  void SetPort(unsigned char Port);
  void SetCANID(unsigned short CANID);
  void SetMasterID(unsigned short MasterID);
  unsigned char GetPort(void);
  unsigned short GetCANID(void);
  unsigned short GetMasterID(void);
  unsigned char IsConfigured(void);

  void SetMITRange(float PositionMin, float PositionMax, float SpeedMin, float SpeedMax, float TorqueMin, float TorqueMax);
  void SetMITFullRange(float PositionMin, float PositionMax,
                       float SpeedMin, float SpeedMax,
                       float TorqueMin, float TorqueMax,
                       float KpMin, float KpMax,
                       float KdMin, float KdMax);
  void SetMITCommand(float Position, float Speed, float Kp, float Kd, float Torque);
  void SetPosition(float Position);
  void SetSpeed(float Speed);
  void SetKp(float Kp);
  void SetKd(float Kd);
  void SetTorque(float Torque);
  void ClearCommand(void);

  void Enable(void);
  void EnablePure(void);
  void SendEnableFrame(void);
  void SendDisableFrame(void);
  void SendClearErrorFrame(void);
  void SetRunFlag(unsigned char RunFlag);
  unsigned char SetupTimeout(unsigned int TimeoutMs);
  unsigned char ReadRegister(unsigned char Reg, unsigned int *OutValue);
  unsigned char WriteRegister(unsigned char Reg, unsigned int Value, unsigned char SaveAfterWrite);
  void Disable(void);
  void SaveZero(void);
  void ClearError(void);

  float GetPosition(void);
  float GetSpeed(void);
  float GetTorque(void);
  unsigned char GetMosTemp(void);
  unsigned char GetRotorTemp(void);
  unsigned char GetState(void);
  unsigned char IsEnable(void);
  unsigned char IsOnline(unsigned int Timeout_ms);
  TickType_t GetLastRxTick(void);

  friend void Motor_CAN_Back(unsigned char Code, unsigned char Err, unsigned char *RxData);
  friend unsigned char Motor_CAN_Send(unsigned char Code);
};

Motor_t *MotorPoint(unsigned char Code);
#endif

void Motor_Init(void);
void Motor_Task(void const *argument);
unsigned char Motor_SendAllOnce(void);
unsigned char Motor_SendListOnce(const unsigned char *Codes, unsigned char Count);
unsigned char Motor_SendPortBudgeted(unsigned char Port, unsigned char *Cursor, unsigned char Budget);
void Motor_ClearTxFailure(void);
unsigned char Motor_HasTxFailure(void);

#ifdef __cplusplus
}
#endif

#endif

/**********************************END OF FILE***********************************/
