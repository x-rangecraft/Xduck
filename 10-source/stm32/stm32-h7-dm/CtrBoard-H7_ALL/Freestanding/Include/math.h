#ifndef FREESTANDING_MATH_H
#define FREESTANDING_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef float float_t;
typedef double double_t;

double sqrt(double x);
float sqrtf(float x);

#ifdef __cplusplus
}
#endif

#endif
