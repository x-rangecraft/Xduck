#ifndef __DMUSB_TASK_H
#define __DMUSB_TASK_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void DMUSB_TaskInit(void);
void DMUSB_OnReceive(const uint8_t *data, uint16_t len);
#ifdef __cplusplus
}
#endif
#endif
