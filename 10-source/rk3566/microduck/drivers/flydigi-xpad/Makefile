CC ?= cc
CFLAGS ?= -O2
CPPFLAGS ?=

WARNINGS := -Wall -Wextra -Werror -Wformat=2 -Wshadow -Wconversion -Wstrict-prototypes

.PHONY: all clean test

all: xpad-usbd

xpad-usbd: xpad-usbd.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNINGS) -std=c11 -o $@ $<

test: xpad-usbd
	./xpad-usbd --self-test

clean:
	rm -f xpad-usbd
