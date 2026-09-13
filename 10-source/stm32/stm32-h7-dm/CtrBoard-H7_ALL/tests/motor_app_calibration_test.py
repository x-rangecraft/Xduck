"""Run production calibration, shaping and fault functions against simulated motors.
No hardware access; invokes the host C++ compiler.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'App/Src/motor_app.cpp').read_text()

def function(signature):
    start = source.index(signature + '\n{')
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'

constants = '\n'.join(line for line in source.splitlines() if line.startswith('#define MOTOR_APP_'))
faults = '\n'.join(line for line in (root/'App/Inc/motor_app.h').read_text().splitlines()
                   if line.startswith('#define MOTOR_APP_'))
tables = source[source.index('typedef struct {'):source.index('static const unsigned char kMotorAppCanPort1MitList')]
pre = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include "h7spi_protocol.h"
#include "dmusb_protocol.h"
#include "imu_task.h"
using TickType_t = unsigned;
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
#define Motor_State_CommLost 13
'''
stubs = r'''
unsigned now_ms = 1000;
float g_positionMin[24], g_positionMax[24];
unsigned char g_positionLimitsReady, g_motorAppEnabled, g_motorAppMode=MOTOR_APP_MODE_ADMIN;
unsigned char g_motorAppAdminOp, g_motorAppRegisterBusy, g_motorAppRequestedCount;
unsigned char g_motorAppSuccessCount, g_motorAppFault, g_motorAppFailedMotorID;
unsigned char g_motorAppFailedRouteIndex, g_motorAppFailedPhase, g_motorAppPhase, g_motorAppRouteIndex;
unsigned char g_motorAppLastFaultMotorID, g_motorAppLastFaultRouteIndex;
unsigned char g_motorAppFeedbackRecovering, g_motorAppPendingSpiCommandValid, g_motorAppCanPort1Cursor;
uint32_t g_motorAppFeedbackRecoverySinceMs;
unsigned long g_motorAppRequestedMask, g_motorAppSucceededMask, g_motorAppFaultFlags;
unsigned g_motorAppProtectionStartTick, g_motorAppLastHostCommandTickMs;
unsigned tx_failure, command_clears;
imu_snapshot_t snapshot;
struct Motor_t {
 float position=0, speed=0;
 unsigned state=0, online=1, mos=30, rotor=30, disabled=0, run=0, clears=0;
 uint32_t last_rx=now_ms;
 float GetPosition(){return position;} float GetSpeed(){return speed;}
 unsigned GetState(){return state;}
 unsigned IsOnline(unsigned timeout){return online && uint32_t(now_ms-last_rx)<=timeout;}
 unsigned GetMosTemp(){return mos;} unsigned GetRotorTemp(){return rotor;}
 void SetRunFlag(unsigned f){run=f;} void SendDisableFrame(){disabled++;}
 void ClearCommand(){clears++;}
} motors[14];
unsigned HAL_GetTick(){return now_ms;} unsigned xTaskGetTickCount(){return now_ms;}
unsigned MotorApp_GetDefaultCount(){return 14;}
unsigned MotorApp_IsActiveMitRoute(unsigned i){return kMotorAppDefaultConfig[i].port!=0;}
Motor_t* MotorApp_GetRouteMotor(unsigned i){return &motors[i];}
unsigned MotorApp_GetRouteIndexByMotorID(unsigned id,unsigned char*route){
 for(unsigned i=0;i<14;i++) if(id && kMotorAppDefaultConfig[i].can_id==id){*route=i;return 1;}
 return 0;
}
void MotorApp_ClearAdminStats(){g_motorAppRequestedMask=g_motorAppSucceededMask=0;g_motorAppRequestedCount=g_motorAppSuccessCount=0;}
void MotorApp_RecordAdminSuccess(unsigned i){g_motorAppSucceededMask|=1ul<<i;g_motorAppSuccessCount++;}
float MotorApp_AbsFloat(float x){return std::fabs(x);}
float MotorApp_MaxFloat(float a,float b){return a>b?a:b;}
float MotorApp_LimitFloat(float x,float a,float b){return x<a?a:x>b?b:x;}
unsigned MotorApp_IsTickExpired(unsigned now,unsigned deadline){return (int)(now-deadline)>=0;}
uint8_t IMU_GetSnapshot(imu_snapshot_t *out){*out=snapshot;return 1;}
void Motor_ClearTxFailure(){tx_failure=0;}
unsigned Motor_HasTxFailure(){return tx_failure;}
void MotorApp_ClearCommandTracking(){command_clears++;}
static void MotorApp_EnterFault(unsigned char,unsigned long);
'''
code = ''.join(function(signature) for signature in [
    'static unsigned char MotorApp_CalibrationAllowed(void)',
    'unsigned char MotorApp_SetPositionLimits(const dmusb_position_limit_t *limits, unsigned char count)',
    'static unsigned char MotorApp_ImuHealthy(void)',
    'static void MotorApp_PrepareFallbackCommands(void)',
    'static void MotorApp_DisableAllRoutes(void)',
    'static void MotorApp_EnterFault(unsigned char RouteIndex, unsigned long FaultFlag)',
    'static unsigned char MotorApp_IsHardwareFaultState(unsigned char State)',
    'static void MotorApp_CheckRuntimeProtection(void)',
    'static void MotorApp_CheckHostWatchdog(void)',
    'static void MotorApp_CheckFeedbackRecovery(void)',
    'unsigned char MotorApp_GetLastFaultMotorID(void)',
    'unsigned char MotorApp_GetLastFaultRouteIndex(void)',
])
# Multiline signature, but extract using the exact prefix in the source.
start = source.index('static unsigned char MotorApp_ShapeMitCommand(')
signature = source[start:source.index('\n{', start)]
code += function(signature)
post = r'''
void reset() {
 g_motorAppEnabled=g_motorAppFault=g_motorAppAdminOp=g_motorAppRegisterBusy=0;
 g_motorAppMode=MOTOR_APP_MODE_ADMIN;g_motorAppFaultFlags=0;g_positionLimitsReady=0;
 g_motorAppProtectionStartTick=now_ms;
 g_motorAppFeedbackRecovering=0;g_motorAppPendingSpiCommandValid=0;
 g_motorAppLastFaultMotorID=0;g_motorAppLastFaultRouteIndex=0xff;
 for(unsigned i=0;i<14;i++){motors[i]=Motor_t{};g_positionMin[i]=-3;g_positionMax[i]=3;}
 snapshot={};snapshot.flags=IMU_FLAG_SENSOR_OK|IMU_FLAG_CALIBRATED;
 snapshot.sample_tick_ms=now_ms;snapshot.quaternion[0]=1;snapshot.projected_gravity[2]=-1;
}
void all_disabled(){for(unsigned i=0;i<14;i++)assert(motors[i].disabled==(MotorApp_IsActiveMitRoute(i)?1u:0u));}
void timeout_fault(){
 reset();g_motorAppEnabled=1;g_motorAppMode=MOTOR_APP_MODE_CONTROL_MIT;
 for(auto&m:motors){m.state=1;m.run=1;}
 motors[5].online=0;MotorApp_CheckRuntimeProtection();
 assert(g_motorAppFaultFlags==MOTOR_APP_FAULT_FEEDBACK_STALE && !g_motorAppEnabled);
 assert(MotorApp_GetLastFaultMotorID()==6 && MotorApp_GetLastFaultRouteIndex()==5);
 // Simulate actual disable acknowledgements, not just the gateway's request.
 for(auto&m:motors){m.state=0;m.online=1;}
 g_motorAppPendingSpiCommandValid=1;
 MotorApp_CheckFeedbackRecovery();
}
void recover_ticks(unsigned duration){
 for(unsigned i=0;i<duration;i++){
  now_ms++;snapshot.sample_tick_ms=now_ms;
  for(auto&m:motors)if(m.online)m.last_rx=now_ms;
  MotorApp_CheckFeedbackRecovery();
 }
}
void recovered_but_disabled(){
 assert(!g_motorAppFault && !g_motorAppEnabled && !g_motorAppPendingSpiCommandValid);
 assert(g_motorAppMode==MOTOR_APP_MODE_ADMIN && g_motorAppAdminOp==MOTOR_APP_ADMIN_NONE);
 for(auto&m:motors)assert(!m.run && !m.state);
 assert(MotorApp_GetLastFaultMotorID()==6 && MotorApp_GetLastFaultRouteIndex()==5);
}
int main(){
 assert(MotorApp_IsHardwareFaultState(13));
 reset();
 dmusb_position_limit_t limits[14]{};
 unsigned n=0;
 for(unsigned i=0;i<14;i++) if(MotorApp_IsActiveMitRoute(i)){
  limits[n].motor_id=kMotorAppDefaultConfig[i].can_id;
  limits[n].min_mrad=-4000;limits[n++].max_mrad=4000;
 }
 assert(MotorApp_SetPositionLimits(limits,14)==0 && g_positionLimitsReady);
 assert(g_motorAppSuccessCount==14 && g_motorAppSucceededMask==0x3fff);
 // Real enabled feedback refuses calibration even if gateway thinks it is disabled.
 motors[5].state=1;assert(!MotorApp_CalibrationAllowed());
 assert(MotorApp_SetPositionLimits(limits,14)==H7SPI_RESULT_BUSY);
 motors[5].state=0;motors[0].online=0;assert(!MotorApp_CalibrationAllowed());motors[0].online=1;
 g_motorAppEnabled=1;assert(!MotorApp_CalibrationAllowed());g_motorAppEnabled=0;
 limits[4].min_mrad=4000;assert(MotorApp_SetPositionLimits(limits,14)==H7SPI_RESULT_LIMIT_EXCEEDED);
 assert(g_positionMin[0]==-4 && g_positionMax[0]==4); // no partial application
 limits[4].min_mrad=-4000;limits[4].motor_id=2;
 assert(MotorApp_SetPositionLimits(limits,14)==H7SPI_RESULT_DUPLICATE_MOTOR_ID);
 // The exact 5 -> 4 contract survives torque shaping, even outside the boundary.
 h7spi_motor_cmd_t cmd{};cmd.p_mrad=5000;cmd.kp_centi=65535;cmd.kd_milli=65535;
 cmd.v_mrad_s=100000;cmd.torque_mnm=100000;
 MotorApp_ShapedCommand_t out;
 for(unsigned i=0;i<14;i++)if(MotorApp_IsActiveMitRoute(i)){
  motors[i].position=6;motors[i].speed=100;
  MotorApp_ShapeMitCommand(&cmd,i,&out);
  assert(out.position==4);assert(out.speed==kMotorAppDefaultConfig[i].v_max);
  assert(out.kp>=0 && out.kp<=500 && out.kd>=0 && out.kd<=5);
  float total=out.kp*(out.position-motors[i].position)+out.kd*(out.speed-motors[i].speed)+out.torque;
  assert(std::fabs(total)<=kMotorAppDefaultConfig[i].t_max+0.01f);
 }
 cmd.p_mrad=-5000;MotorApp_ShapeMitCommand(&cmd,0,&out);assert(out.position==-4);
 // No 8-second grace: first enabled tick catches a lost motor.
 reset();g_motorAppEnabled=1;for(auto &m:motors)m.state=1;motors[5].online=0;MotorApp_CheckRuntimeProtection();
 assert(g_motorAppFaultFlags==MOTOR_APP_FAULT_FEEDBACK_STALE && !g_motorAppEnabled);all_disabled();
 reset();g_motorAppEnabled=1;snapshot.flags=0;MotorApp_CheckRuntimeProtection();
 assert(g_motorAppFaultFlags==MOTOR_APP_FAULT_IMU_INVALID);all_disabled();
 reset();g_motorAppEnabled=1;snapshot.sample_tick_ms=now_ms-101;MotorApp_CheckRuntimeProtection();
 assert(g_motorAppFaultFlags==MOTOR_APP_FAULT_IMU_INVALID);all_disabled();
 reset();g_motorAppEnabled=1;snapshot.quaternion[0]=NAN;MotorApp_CheckRuntimeProtection();
 assert(g_motorAppFaultFlags==MOTOR_APP_FAULT_IMU_INVALID);all_disabled();
 reset();g_motorAppEnabled=1;for(auto &m:motors)m.state=1;motors[9].mos=75;MotorApp_CheckRuntimeProtection();
 assert(g_motorAppFaultFlags==MOTOR_APP_FAULT_MOS_OVER_TEMP);all_disabled();
 reset();g_motorAppEnabled=1;g_motorAppLastHostCommandTickMs=now_ms-100;
 MotorApp_CheckHostWatchdog();assert(g_motorAppEnabled);now_ms++;
 MotorApp_CheckHostWatchdog();assert(!g_motorAppEnabled && (g_motorAppFaultFlags & MOTOR_APP_FAULT_HOST_COMMAND_STALE));all_disabled();
 // Recovery clears only the current timeout, after a full healthy second.
 timeout_fault();recover_ticks(999);assert(g_motorAppFault);
 recover_ticks(1);recovered_but_disabled();assert(!g_motorAppFaultFlags);
 const unsigned clears=motors[5].clears;recover_ticks(2000);assert(motors[5].clears==clears);
 // A second fault starts a new window; recovery is not sticky across episodes.
 timeout_fault();recover_ticks(500);motors[9].online=0;recover_ticks(2000);assert(g_motorAppFault);
 motors[9].online=1;recover_ticks(1000);assert(g_motorAppFault);
 recover_ticks(1);recovered_but_disabled();
 // Every configured motor must acknowledge disabled and remain healthy.
 for(unsigned route=0;route<14;route++)if(MotorApp_IsActiveMitRoute(route)){
  for(unsigned problem=0;problem<5;problem++){
   timeout_fault();recover_ticks(500);
   switch(problem){case 0:motors[route].online=0;break;case 1:motors[route].state=1;break;
    case 2:motors[route].state=8;break;case 3:motors[route].mos=75;break;case 4:motors[route].rotor=80;break;}
   recover_ticks(2000);assert(g_motorAppFault);
   motors[route]=Motor_t{};recover_ticks(1001);recovered_but_disabled();
  }
 }
 // IMU, management activity and non-recoverable faults reset/prevent recovery.
 timeout_fault();recover_ticks(500);snapshot.flags=0;recover_ticks(2000);assert(g_motorAppFault);
 snapshot.flags=IMU_FLAG_SENSOR_OK|IMU_FLAG_CALIBRATED;recover_ticks(1001);recovered_but_disabled();
 timeout_fault();recover_ticks(500);g_motorAppAdminOp=MOTOR_APP_ADMIN_CLEAR_ERROR;
 recover_ticks(2000);assert(g_motorAppFault);g_motorAppAdminOp=MOTOR_APP_ADMIN_NONE;
 recover_ticks(1001);recovered_but_disabled();
 timeout_fault();g_motorAppRegisterBusy=1;recover_ticks(2000);assert(g_motorAppFault);
 g_motorAppRegisterBusy=0;recover_ticks(1001);recovered_but_disabled();
 for(unsigned long bit=1;bit<=0x200;bit<<=1){
  if(bit==MOTOR_APP_FAULT_FEEDBACK_STALE || bit==MOTOR_APP_FAULT_HOST_COMMAND_STALE)continue;
  timeout_fault();g_motorAppFaultFlags|=bit;recover_ticks(2000);
  assert(g_motorAppFault && (g_motorAppFaultFlags&bit) && (g_motorAppFaultFlags&MOTOR_APP_FAULT_FEEDBACK_STALE));
 }
 timeout_fault();g_motorAppFaultFlags|=MOTOR_APP_FAULT_HOST_COMMAND_STALE;
 recover_ticks(1000);recovered_but_disabled();assert(g_motorAppFaultFlags==MOTOR_APP_FAULT_HOST_COMMAND_STALE);
 // Unsigned elapsed time works at boot tick zero and across the 32-bit wrap.
 now_ms=0;timeout_fault();recover_ticks(1000);recovered_but_disabled();
 now_ms=UINT32_MAX-500;timeout_fault();recover_ticks(1000);recovered_but_disabled();
 puts("production calibration/protection/recovery passed: 1s stability, all-motor disable acknowledgements, intermittent loss, mixed faults, IMU/admin guards, no enable, retained diagnostics, tick wrap");
}
'''
with tempfile.TemporaryDirectory() as temp:
    p = Path(temp) / 'test.cpp'
    p.write_text(pre + constants + '\n' + faults + '\n' + tables + stubs + code + post)
    subprocess.run(['c++', '-std=c++11', '-Wall', '-Wextra', '-Werror', '-I', str(root/'App/Inc'), str(p), '-o', str(p.with_suffix(''))], check=True)
    subprocess.run([str(p.with_suffix(''))], check=True)
