export ARCH = arm64
export CROSS_COMPILE=aarch64-linux-gnu-

TARGET := hello-kernel
WARN := -W -Wall -Wstrict-prototypes -Wmissing-prototypes
INCLUDE := -isystem /usr/src/linux-source-3.16/include
CFLAGS := -C -DMODULE -D__KERNEL__ ${WARN} ${INCLUDE}
CC := /usr/bin/aarch64-linux-gnu-gcc
LD := /usr/bin/aarch64-linux-gnu-ld

${TARGET}.o: ${TARGET}.c

.PHONY: clean

clean:
	rm -rf {TARGET}.o
