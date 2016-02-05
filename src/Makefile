SRC_DIR := src
BUILD_DIR := build

obj-m += modules/pir-irq.o

OBJECTS := led-user algorithm/test

KDIR := /lib/modules/$(shell uname -r)/build

ccflags-y += -Wall
cxxflags-y += -Wall

.PHONY: all clean

all: $(OBJECTS)
	$(MAKE) -C $(KDIR) SUBDIRS=$(shell pwd) modules

led-user:
	$(CXX) -o ../$(BUILD_DIR)/led-user led-user.c

algorithm/test:
	$(CXX) -o ../$(BUILD_DIR)/algorithm/test algorithm/test.cpp

clean:
	$(MAKE) -C $(KDIR) SUBDIRS=$(shell pwd) clean
	rm -rf ../$(BUILD_DIR)
