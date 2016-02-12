SRC_DIR := src
BUILD_DIR := build
MODULE_DIR := modules

ALGO_SUBDIR := algorithm

OBJECTS := led-user $(ALGO_SUBDIR)/test

.PHONY: all clean directory program

# in what kind of OS are these sources built?
ifeq "$(OS)" "Windows_NT"
TARGET_OS := WIN32
else
UNAME_S := $(shell uname -s)
ifeq "$(UNAME_S)" "Linux"
TARGET_OS := LINUX
endif
ifeq "$(UNAME_S)" "Darwin"
TARGET_OS := OSX
endif
endif
export TARGET_OS

all: directory program

directory:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(ALGO_SUBDIR)
    
program: $(OBJECTS)
ifeq ($(TARGET_OS),LINUX)
	# we build kernel modules only in Linux environment
	$(MAKE) -C $(MODULE_DIR) all
endif

clean:
ifeq ($(TARGET_OS),LINUX)
	# we clean kernel modules only in Linux environment
	$(MAKE) -C $(MODULE_DIR) clean
endif
	rm -rf $(BUILD_DIR)

led-user:
	$(CXX) -o $(BUILD_DIR)/led-user $(SRC_DIR)/led-user.c

$(ALGO_SUBDIR)/test:
	$(CXX) -o $(BUILD_DIR)/$(ALGO_SUBDIR)/test $(SRC_DIR)/$(ALGO_SUBDIR)/test.cpp
