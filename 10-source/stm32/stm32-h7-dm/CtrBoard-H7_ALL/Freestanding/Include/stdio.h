#ifndef FREESTANDING_STDIO_H
#define FREESTANDING_STDIO_H

#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __freestanding_file FILE;

int printf(const char *fmt, ...);
int snprintf(char *str, size_t size, const char *fmt, ...);
int vsnprintf(char *str, size_t size, const char *fmt, va_list ap);

#ifdef __cplusplus
}
#endif

#endif
