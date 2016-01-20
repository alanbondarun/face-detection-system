export ARCH=arm64
export CROSS_COMPILE=/usr/bin/aarch64-linux-gnu-

obj-m += hello-kernel.o
KDIR := /lib/modules/$(shell uname -r)/build

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean
