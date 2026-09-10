#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

void *memcpy(void *dest, const void *src, size_t n)
{
  unsigned char *d = (unsigned char *)dest;
  const unsigned char *s = (const unsigned char *)src;
  while (n-- != 0U) {
    *d++ = *s++;
  }
  return dest;
}

void *memmove(void *dest, const void *src, size_t n)
{
  unsigned char *d = (unsigned char *)dest;
  const unsigned char *s = (const unsigned char *)src;
  if (d < s) {
    while (n-- != 0U) {
      *d++ = *s++;
    }
  } else if (d > s) {
    d += n;
    s += n;
    while (n-- != 0U) {
      *--d = *--s;
    }
  }
  return dest;
}

void *memset(void *s, int c, size_t n)
{
  unsigned char *p = (unsigned char *)s;
  while (n-- != 0U) {
    *p++ = (unsigned char)c;
  }
  return s;
}

int memcmp(const void *s1, const void *s2, size_t n)
{
  const unsigned char *a = (const unsigned char *)s1;
  const unsigned char *b = (const unsigned char *)s2;
  while (n-- != 0U) {
    if (*a != *b) {
      return (int)*a - (int)*b;
    }
    a++;
    b++;
  }
  return 0;
}

size_t strlen(const char *s)
{
  const char *p = s;
  while (*p != '\0') {
    p++;
  }
  return (size_t)(p - s);
}

size_t strnlen(const char *s, size_t maxlen)
{
  size_t len = 0U;
  while (len < maxlen && s[len] != '\0') {
    len++;
  }
  return len;
}

static void out_char(char **out, size_t *remaining, int *count, char ch)
{
  if (*remaining > 1U) {
    **out = ch;
    (*out)++;
    (*remaining)--;
  }
  (*count)++;
}

static void out_string(char **out, size_t *remaining, int *count, const char *s)
{
  if (s == (const char *)0) {
    s = "(null)";
  }
  while (*s != '\0') {
    out_char(out, remaining, count, *s++);
  }
}

static void out_unsigned(char **out, size_t *remaining, int *count,
                         unsigned long long value, unsigned int base,
                         unsigned int width, char pad, int upper)
{
  char buf[32];
  unsigned int pos = 0U;
  const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";

  do {
    buf[pos++] = digits[value % base];
    value /= base;
  } while (value != 0ULL && pos < sizeof(buf));

  while (pos < width) {
    out_char(out, remaining, count, pad);
    width--;
  }
  while (pos != 0U) {
    out_char(out, remaining, count, buf[--pos]);
  }
}

int vsnprintf(char *str, size_t size, const char *fmt, va_list ap)
{
  char *out = str;
  size_t remaining = size;
  int count = 0;

  while (*fmt != '\0') {
    if (*fmt != '%') {
      out_char(&out, &remaining, &count, *fmt++);
      continue;
    }

    fmt++;
    char pad = ' ';
    unsigned int width = 0U;
    int long_count = 0;

    if (*fmt == '0') {
      pad = '0';
      fmt++;
    }
    while (*fmt >= '0' && *fmt <= '9') {
      width = (width * 10U) + (unsigned int)(*fmt - '0');
      fmt++;
    }
    while (*fmt == 'l') {
      long_count++;
      fmt++;
    }

    switch (*fmt) {
    case 'c':
      out_char(&out, &remaining, &count, (char)va_arg(ap, int));
      break;
    case 's':
      out_string(&out, &remaining, &count, va_arg(ap, const char *));
      break;
    case 'd':
    case 'i': {
      long long v;
      if (long_count >= 2) {
        v = va_arg(ap, long long);
      } else if (long_count == 1) {
        v = va_arg(ap, long);
      } else {
        v = va_arg(ap, int);
      }
      if (v < 0) {
        out_char(&out, &remaining, &count, '-');
        v = -v;
      }
      out_unsigned(&out, &remaining, &count, (unsigned long long)v, 10U, width, pad, 0);
      break;
    }
    case 'u':
      out_unsigned(&out, &remaining, &count,
                   long_count != 0 ? va_arg(ap, unsigned long) : va_arg(ap, unsigned int),
                   10U, width, pad, 0);
      break;
    case 'x':
    case 'X':
      out_unsigned(&out, &remaining, &count,
                   long_count != 0 ? va_arg(ap, unsigned long) : va_arg(ap, unsigned int),
                   16U, width, pad, (*fmt == 'X'));
      break;
    case 'p':
      out_string(&out, &remaining, &count, "0x");
      out_unsigned(&out, &remaining, &count, (uintptr_t)va_arg(ap, void *), 16U, width, '0', 0);
      break;
    case '%':
      out_char(&out, &remaining, &count, '%');
      break;
    default:
      out_char(&out, &remaining, &count, '%');
      out_char(&out, &remaining, &count, *fmt);
      break;
    }
    if (*fmt != '\0') {
      fmt++;
    }
  }

  if (size != 0U) {
    *out = '\0';
  }
  return count;
}

int snprintf(char *str, size_t size, const char *fmt, ...)
{
  va_list ap;
  int ret;
  va_start(ap, fmt);
  ret = vsnprintf(str, size, fmt, ap);
  va_end(ap);
  return ret;
}

int printf(const char *fmt, ...)
{
  (void)fmt;
  return 0;
}

void *malloc(size_t size)
{
  (void)size;
  return (void *)0;
}

void free(void *ptr)
{
  (void)ptr;
}

void abort(void)
{
  for (;;) {
  }
}

int abs(int value)
{
  return (value < 0) ? -value : value;
}

double sqrt(double x)
{
  double guess = x;
  if (x <= 0.0) {
    return 0.0;
  }
  for (int i = 0; i < 16; i++) {
    guess = 0.5 * (guess + (x / guess));
  }
  return guess;
}

float sqrtf(float x)
{
  return (float)sqrt((double)x);
}

void __libc_init_array(void)
{
}

void __libc_fini_array(void)
{
}

void __cxa_pure_virtual(void)
{
  abort();
}

int __aeabi_atexit(void *object, void (*destructor)(void *), void *dso_handle)
{
  (void)object;
  (void)destructor;
  (void)dso_handle;
  return 0;
}

