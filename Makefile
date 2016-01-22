obj-m += hello-kernel.o
obj-m += vol-up-irq.o

KDIR := /lib/modules/$(shell uname -r)/build
USR_TARGET = vol-up-user

.PHONY: all clean

all:
	make -C $(KDIR) M=$(PWD) modules
	gcc -o $(USR_TARGET) $(USR_TARGET).c

clean:
	make -C $(KDIR) M=$(PWD) clean
	rm -rf ./$(USR_TARGET)
