SRC_DIR := src
BUILD_DIR := build

obj-m += modules/pir-irq.o

OBJECTS := led-user algorithm/test

KDIR := /lib/modules/$(shell uname -r)/build

ccflags-y += -Wall
cxxflags-y += -Wall

# in what kind of OS are these sources built?
ifeq ($(OS),Windows_NT)
	TARGET_OS := WIN32
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		TARGET_OS := LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		TARGET_OS := OSX
	endif
endif

.PHONY: all clean

all: $(OBJECTS)
ifeq ($(TARGET_OS),LINUX)
	# we build kernel modules only in Linux environment
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules
endif

led-user:
	$(CXX) -o ../$(BUILD_DIR)/led-user led-user.c

algorithm/test:
	$(CXX) -o ../$(BUILD_DIR)/algorithm/test algorithm/test.cpp

clean:
ifeq ($(TARGET_OS),LINUX)
	# we clean kernel modules only in Linux environment
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean
endif
	rm -rf ../$(BUILD_DIR)
