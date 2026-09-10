from pathlib import Path
import subprocess,tempfile
# Compile the production function against simulated CAN feedback; never opens hardware.
source=(Path(__file__).resolve().parents[1]/'App/Src/motor_app.cpp').read_text()
a=source.index('static unsigned char MotorApp_StartAllMitFast(void)');b=source.index('static void MotorApp_DisableAllRoutes(void)',a)
body=source[a:b]
pre=r'''
#include <cassert>
#include <cstdio>
using TickType_t=unsigned;
#define H7SPI_MAX_MOTORS 14
#define MOTOR_APP_PHASE_FAST_START 4
#define MOTOR_APP_FAULT_CAN_TX_DROP 32
#define MOTOR_APP_FAULT_MOTOR_HW_FAULT 16
#define MOTOR_APP_FAULT_ENABLE_FAILED 1
#define MOTOR_APP_FAULT_HOST_COMMAND_STALE 128
#define H7SPI_RESULT_FAULT 8
#define H7SPI_RESULT_OK 0
#define MOTOR_APP_MODE_CONTROL_MIT 33
#define MOTOR_APP_MAX_RETRIES 5
#define Motor_State_OverVoltage 8
#define MOTOR_APP_HOST_COMMAND_TIMEOUT_MS 100
#define MOTOR_APP_DM_TIMEOUT_MS 8000
#define MOTOR_APP_ADMIN_NONE 0
#define MOTOR_APP_PHASE_IDLE 0
#define pdMS_TO_TICKS(x) (x)
unsigned now_ms, g_motorAppPhase, g_motorAppEnabled,g_motorAppMode,g_motorAppLastHostCommandTickMs,g_motorAppSucceededMask,g_motorAppRequestedMask,g_motorAppSuccessCount,g_motorAppRequestedCount,g_motorAppProtectionStartTick,g_motorAppAdminOp,g_motorAppRouteCursor,g_motorAppRouteIndex,g_motorAppRetry;
unsigned motor_enable_last_failure[4];
unsigned fail_id, fault, txfail, delayed_id, stuck_id, stale_id, inject_tx, delays, delayed_scheduler;
const unsigned ids[14]={2,0,0,0,0,1,0,0,0,13,14,15,0,0};
struct Motor_t {
 unsigned id, clear_at, enables, enabled, rx, run;
 void SetRunFlag(unsigned r){run=r;}
 unsigned GetLastRxTick(){return rx;}
 void SendClearErrorFrame(){clear_at=now_ms;enabled=0;}
 void SendEnableFrame(){assert(id);assert(now_ms>=clear_at+5);enables++;if(id==inject_tx)txfail=1;if(id!=stuck_id&&(id!=delayed_id||enables>=3))enabled=1;if(id!=stale_id)rx=now_ms;}
 unsigned GetCANID(){return id;}
 unsigned GetState(){return enabled;}
 unsigned IsEnable(){return enabled;}
} motors[14];
unsigned MotorApp_GetDefaultCount(){return 14;}
unsigned MotorApp_IsActiveMitRoute(unsigned i){return ids[i]!=0;}
Motor_t* MotorApp_GetRouteMotor(unsigned i){return &motors[i];}
void Motor_ClearTxFailure(){txfail=0;}
unsigned Motor_HasTxFailure(){return txfail;}
void MotorApp_EnterFault(unsigned i,unsigned f){fault=f;fail_id=i<14?ids[i]:255;g_motorAppEnabled=0;for(auto &m:motors){m.enabled=0;m.run=0;}}
void MotorApp_RecordAdminSuccess(unsigned i){g_motorAppSucceededMask|=1u<<i;g_motorAppSuccessCount++;}
void MotorApp_PrepareFallbackCommands(){for(auto&m:motors)m.run=0;}
void MotorApp_SendActiveMitOnce(){for(auto&m:motors)if(m.id&&m.id!=stale_id)m.rx=now_ms;}
unsigned HAL_GetTick(){return now_ms;}
unsigned xTaskGetTickCount(){return now_ms;}
void osDelay(unsigned ms){assert(g_motorAppPhase==MOTOR_APP_PHASE_FAST_START);now_ms+=(delayed_scheduler && ms==10 ? 101 : ms);delays++;}
'''
post=r'''
void reset(){now_ms=10;delays=0;delayed_scheduler=0;txfail=fault=fail_id=0;g_motorAppSucceededMask=g_motorAppSuccessCount=g_motorAppEnabled=0;g_motorAppAdminOp=1;g_motorAppRequestedCount=5;g_motorAppRequestedMask=0xe21;delayed_id=stuck_id=stale_id=inject_tx=0;for(unsigned i=0;i<14;i++)motors[i]={ids[i],0,0,0,1,0};}
int main(){
 reset();assert(MotorApp_StartAllMitFast()==0);assert(g_motorAppSuccessCount==5);assert(now_ms==30);for(auto&m:motors)assert(m.id?m.enables==1:m.enables==0);
 reset();delayed_id=2;assert(MotorApp_StartAllMitFast()==0);assert(g_motorAppSuccessCount==5);assert(motors[0].enables==3);assert(motors[5].enables==1);assert(now_ms-g_motorAppLastHostCommandTickMs<100);
 reset();stuck_id=2;assert(MotorApp_StartAllMitFast()==8);assert(fault==1&&fail_id==2&&g_motorAppSuccessCount==4);for(auto&m:motors)assert(!m.run&&!m.enabled);assert(motors[0].enables==5);assert(motor_enable_last_failure[0]==2&&motor_enable_last_failure[1]==0);
 reset();stale_id=2;assert(MotorApp_StartAllMitFast()==8);assert(fault==1&&fail_id==2&&g_motorAppSuccessCount==4);
 reset();inject_tx=2;assert(MotorApp_StartAllMitFast()==8);assert(fault==32&&fail_id==2);for(auto&m:motors)assert(!m.enabled);
 reset();delayed_scheduler=1;assert(MotorApp_StartAllMitFast()==8);assert(fault==128);for(auto&m:motors)assert(!m.enabled);
 puts("production enable handshake: immediate success, selective retry, stuck motor shutdown, stale feedback rejection, CAN failure shutdown, watchdog deadline passed");
}
'''
with tempfile.TemporaryDirectory() as temp:
 p=Path(temp)/'test.cpp';p.write_text(pre+body+post)
 subprocess.run(['c++','-std=c++11','-Wall','-Wextra','-Werror',str(p),'-o',str(p.with_suffix(''))],check=True)
 subprocess.run([str(p.with_suffix(''))],check=True)
