"""Compile production route setup and CAN scalar codecs against read-back values.
No hardware access. Requires a host C++ compiler.
"""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'App/Src/motor_app.cpp').read_text()
h=(root/'Hardware/Src/Motor.cpp').read_text()
constants='\n'.join(line for line in s.splitlines() if line.startswith('#define MOTOR_APP_'))
tables=s[s.index('typedef struct {'):s.index('static const unsigned char kMotorAppCanPort1MitList')]
setup=s[s.index('void MotorApp_ConfigDefault(void)'):s.index('unsigned char MotorApp_ConfigureStartupRegisters(void)')]
codec=h[h.index('static float Motor_LimitFloat('):h.index('static unsigned char Motor_SendSpecialFrame(')]
pre=r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#define MOTOR_FEEDBACK_ID_P16_OFFSET 16
'''
stubs=r'''
float g_positionMin[24], g_positionMax[24];
unsigned char g_positionLimitsReady;
struct Motor_t {
 unsigned id=0;
 float values[10]{};
 void SetConfig(unsigned char,unsigned short i,unsigned short master){id=i;assert(master==i+16);}
 void SetMITFullRange(float a,float b,float c,float d,float e,float f,float g,float h,float i,float j){float input[10]={a,b,c,d,e,f,g,h,i,j};for(unsigned k=0;k<10;k++)values[k]=input[k];}
} motors[24];
Motor_t* MotorPoint(unsigned char i){return &motors[i];}
void MotorApp_ClearRuntimeState(){}
void MotorApp_PrepareFallbackCommands(){}
'''
post=r'''
int main(){
 MotorApp_ConfigDefault();
 unsigned configured=0;
 for(auto&m:motors) {
  if(!m.id)continue;
  configured++;
  float expected_v=(m.id==1||m.id==2)?30.0f:10.0f;
  float expected_t=(m.id==1||m.id==2)?10.0f:28.0f;
  assert(m.values[0]==-12.5f&&m.values[1]==12.5f);
  assert(m.values[2]==-expected_v&&m.values[3]==expected_v);
  assert(m.values[4]==-expected_t&&m.values[5]==expected_t);
  assert(Motor_FloatToUint(1.0f,m.values[0],m.values[1],16)==35388);
  float positions[]={-3,-1,0,1,3};
  for(float p:positions){auto raw=Motor_FloatToUint(p,m.values[0],m.values[1],16);assert(std::fabs(Motor_UintToFloat(raw,m.values[0],m.values[1],16)-p)<0.0004f);}
 }
 assert(configured==5);assert(motors[3].id==0);
 for(unsigned i=0;i<14;i++){
  const auto&c=kMotorAppDefaultConfig[i];
  if(!c.port)continue;
  const auto&w=kMotorAppWireRange[i];
  assert(c.p_min==-3.0f&&c.p_max==3.0f);
  assert(c.v_max<=w.v_max&&c.t_max<=w.t_max);
 }
 puts("production motor setup/codecs: all five read-back ranges, 1-rad golden encoding, position round trips, unchanged position guard and capability limits passed");
}
'''
with tempfile.TemporaryDirectory() as temp:
 p=Path(temp)/'test.cpp';p.write_text(pre+constants+'\n'+tables+stubs+setup+codec+post)
 subprocess.run(['c++','-std=c++11','-Wall','-Wextra','-Werror',str(p),'-o',str(p.with_suffix(''))],check=True)
 subprocess.run([str(p.with_suffix(''))],check=True)
