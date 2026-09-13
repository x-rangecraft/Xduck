"""Compile the production FD transmit and feedback-state paths with a fake HAL.

No hardware access. Exercise retry transport, packet lengths, queue failures,
and the captured undocumented state without inferring successful enable.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
motor = (root / 'Hardware/Src/Motor.cpp').read_text()
bsp = (root / 'Bsp/Src/BSP_CAN.cpp').read_text()


def function(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


pre = r'''
#include <cassert>
#include <cstring>
#include <cstdio>
#define DM_CAN_FD_MOTOR 1
#define MOTOR_REGISTER_FRAME_ID 0x7ff
#define MOTOR_REGISTER_CMD_WRITE 0x55
enum { FDCAN_STANDARD_ID, FDCAN_DATA_FRAME, FDCAN_ESI_ACTIVE,
       FDCAN_BRS_OFF, FDCAN_BRS_ON, FDCAN_FD_CAN, FDCAN_NO_TX_EVENTS, FDCAN_STORE_TX_EVENTS };
enum { FDCAN_DLC_BYTES_4=4, FDCAN_DLC_BYTES_8=8, HAL_OK=0 };
struct FDCAN_HandleTypeDef {} handle;
struct FDCAN_TxHeaderTypeDef {
 unsigned Identifier, IdType, TxFrameType, DataLength, ErrorStateIndicator;
 unsigned BitRateSwitch, FDFormat, TxEventFifoControl, MessageMarker;
} last;
unsigned free_level=1, fail_add=0, sends=0;
unsigned char bytes[8];
FDCAN_HandleTypeDef *BSP_CAN_GetHandleByPort(unsigned char p){return p>=1&&p<=3?&handle:nullptr;}
unsigned HAL_FDCAN_GetTxFifoFreeLevel(FDCAN_HandleTypeDef*){return free_level;}
unsigned HAL_FDCAN_AddMessageToTxFifoQ(FDCAN_HandleTypeDef*, FDCAN_TxHeaderTypeDef* h, unsigned char* p){
 last=*h;std::memcpy(bytes,p,8);sends++;return fail_add;
}
unsigned char BSP_CAN_TrySendStandardDataMessage(unsigned char,unsigned,unsigned char*){
 assert(false && "all I2RT retries must use FD+BRS");return 0;
}
'''
body = function(bsp, 'unsigned char BSP_CAN_TrySendStandardFdFrame(')
body += function(motor, 'static unsigned char Motor_PortUsesFd(')
body += function(motor, 'static unsigned char Motor_RegisterSendFrame(')
body += function(motor, 'static unsigned char Motor_NormalizeFeedbackState(')
post = r'''
int main(){
 for(unsigned port=1;port<=3;port++)for(unsigned retry=0;retry<6;retry++){
  for(unsigned cmd: {0x33u,0x55u,0xaau}){
   unsigned char request[8]={1,0,(unsigned char)cmd,0x23,3,0,0,0};
   assert(Motor_RegisterSendFrame(port,retry,request));
   assert(last.Identifier==0x7ff && last.IdType==FDCAN_STANDARD_ID);
   assert(last.FDFormat==FDCAN_FD_CAN && last.BitRateSwitch==FDCAN_BRS_ON);
   assert(last.DataLength==(cmd==0x55?8u:4u));
   assert(std::memcmp(bytes,request,last.DataLength)==0);
  }
 }
 unsigned char captured[8]={0x51,0x66,0x02,0x7f,0xf7,0xff,0x20,0x1f};
 assert(Motor_NormalizeFeedbackState(1,0x11,captured[0]>>4,captured)==5);
 for(unsigned id=1;id<=14;id++)for(unsigned state=0;state<16;state++){
  captured[0]=(state<<4)|id;
  assert(Motor_NormalizeFeedbackState(id,id+16,state,captured)==state);
 }
 free_level=0;assert(!BSP_CAN_TrySendStandardFdFrame(1,1,captured,8,1));
 free_level=1;fail_add=1;assert(!BSP_CAN_TrySendStandardFdFrame(1,1,captured,8,1));
 fail_add=0;assert(!BSP_CAN_TrySendStandardFdFrame(0,1,captured,8,1));
 assert(!BSP_CAN_TrySendStandardFdFrame(1,1,captured,6,1));
 puts("production I2RT FD: all ports/retries FD+BRS, 4/8-byte requests, HAL failures, raw states preserved");
}
'''
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / 'test.cpp'
    path.write_text('#include <initializer_list>\n' + pre + body + post)
    subprocess.run(['c++', '-std=c++11', '-Wall', '-Wextra', '-Werror',
                    str(path), '-o', str(path.with_suffix(''))], check=True)
    subprocess.run([str(path.with_suffix(''))], check=True)
