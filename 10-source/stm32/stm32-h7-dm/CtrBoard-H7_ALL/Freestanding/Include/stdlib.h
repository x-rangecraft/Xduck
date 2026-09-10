#ifndef FREESTANDING_STDLIB_H
#define FREESTANDING_STDLIB_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *malloc(size_t size);
void free(void *ptr);
void abort(void);
int abs(int value);

#ifdef __cplusplus
}
#endif

#endif

