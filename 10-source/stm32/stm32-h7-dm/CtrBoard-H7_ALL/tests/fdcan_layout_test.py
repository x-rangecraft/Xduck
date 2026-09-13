"""Run production CAN initializers against a host HAL stub; check bus invariants."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'Core/Src/fdcan.c').read_text()
# Keep the production RAM expressions and all three initializers, excluding GPIO.
source = source[source.index('/* USER CODE BEGIN 0 */'):
                source.index('static uint32_t HAL_RCC_FDCAN_CLK_ENABLED')]
fields = sorted(set(re.findall(r'\.Init\.(\w+)', source)))
constants = sorted(set(re.findall(r'\bFDCAN_(?:FRAME_\w+|MODE_NORMAL|TX_FIFO_OPERATION)\b', source)))
stub = '''#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#define DM_CAN_FD_MOTOR 1
#define FDCAN_DATA_BYTES_8 4U
#define ENABLE 1
#define DISABLE 0
#define HAL_OK 0
#define FDCAN1 1
#define FDCAN2 2
#define FDCAN3 3
'''
stub += '\n'.join(f'#define {name} {i+10}' for i, name in enumerate(constants))
stub += '\ntypedef struct { unsigned ' + ','.join(fields) + '; } Init;\n'
stub += '''typedef struct { unsigned Instance; Init Init; } FDCAN_HandleTypeDef;
static int HAL_FDCAN_Init(FDCAN_HandleTypeDef *h) { (void)h; return HAL_OK; }
static void Error_Handler(void) { assert(0); }
'''
checks = r'''
int main(void) {
  MX_FDCAN1_Init(); MX_FDCAN2_Init(); MX_FDCAN3_Init();
  FDCAN_HandleTypeDef *ports[] = {&hfdcan1, &hfdcan2, &hfdcan3};
  const unsigned motors[] = {5, 5, 4};
  unsigned end = 0;
  for (unsigned p = 0; p < 3; ++p) {
    Init c = ports[p]->Init;
    assert(c.FrameFormat == FDCAN_FRAME_FD_BRS);
    assert(c.TransmitPause == ENABLE);
    assert(c.TxEventsNbr >= c.TxFifoQueueElmtsNbr);
    unsigned nt = 1 + c.NominalTimeSeg1 + c.NominalTimeSeg2;
    unsigned dt = 1 + c.DataTimeSeg1 + c.DataTimeSeg2;
    assert(120000000U == 1000000U * c.NominalPrescaler * nt);
    assert(120000000U == 5000000U * c.DataPrescaler * dt);
    assert(4 * (1 + c.NominalTimeSeg1) == 3 * nt); /* 75% nominal */
    assert(c.NominalSyncJumpWidth <= c.NominalTimeSeg2);
    assert(c.DataSyncJumpWidth <= c.DataTimeSeg2);
    assert(c.RxFifo0ElmtsNbr >= 2 * motors[p]);
    assert(c.TxFifoQueueElmtsNbr >= 3 * motors[p] + 1);
    assert(c.TxFifoQueueElmtsNbr <= 32);
    assert(c.MessageRAMOffset >= end);
    end = c.MessageRAMOffset + c.StdFiltersNbr + 2 * c.ExtFiltersNbr
        + c.RxFifo0ElmtsNbr * c.RxFifo0ElmtSize
        + c.RxFifo1ElmtsNbr * c.RxFifo1ElmtSize
        + c.RxBuffersNbr * c.RxBufferSize + 2 * c.TxEventsNbr
        + (c.TxBuffersNbr + c.TxFifoQueueElmtsNbr) * c.TxElmtSize;
    assert(end <= 2560);
  }
  puts("FDCAN 5/5/4: FD+BRS 1/5 Mbps, nominal 75%, bounded nonoverlapping RAM");
}
'''
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / 'fdcan_test.c'
    path.write_text(stub + source + checks)
    subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                    str(path), '-o', str(path.with_suffix(''))], check=True)
    subprocess.run([str(path.with_suffix(''))], check=True)
