obj-m += pir/pir-irq.o

KDIR := /lib/modules/$(shell uname -r)/build

ccflags-y += -Wall

all:
ifeq ($(TARGET_OS),LINUX)
	# we build kernel modules only in Linux environment
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules
endif

clean:
ifeq ($(TARGET_OS),LINUX)
	# we clean kernel modules only in Linux environment
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean
endif
