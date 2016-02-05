SRC_DIR := src
BUILD_DIR := build

KDIR := /lib/modules/$(shell uname -r)/build

obj-m := pir-irq.o

ALGO_SUBDIR := algorithm

.PHONY: all clean

all:
	[ -e $(BUILD_DIR) ] || mkdir $(BUILD_DIR)
	[ -e $(BUILD_DIR)/$(ALGO_SUBDIR) ] || mkdir $(BUILD_DIR)/$(ALGO_SUBDIR)
	$(MAKE) -C $(SRC_DIR) all

clean:
	$(MAKE) -C $(SRC_DIR) clean
