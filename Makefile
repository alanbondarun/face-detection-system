SRC_DIR := src
BUILD_DIR := build
MODULE_DIR := modules
INCLUDE_DIR := include
OBJ_DIR := obj

ALGO_SUBDIR := algorithm
LAYER_SUBDIR := layers
CALC_SUBDIR := calc
EXTLIB_SUBDIR := extlib

TARGETS := $(BUILD_DIR)/led-user
MIDDLE_OBJS := $(addprefix $(OBJ_DIR)/, $(EXTLIB_SUBDIR)/jsoncpp.o \
	$(CALC_SUBDIR)/calc-cpu.o $(LAYER_SUBDIR)/layer_data.o \
	$(LAYER_SUBDIR)/sigmoid_layer.o \
	$(LAYER_SUBDIR)/max_pool_layer.o \
	$(LAYER_SUBDIR)/conv_layer.o \
	$(LAYER_SUBDIR)/layer_factory.o \
	network.o led-user.o )

CXXFLAGS := -std=c++0x -I$(INCLUDE_DIR) -Wall

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
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/$(EXTLIB_SUBDIR)
	mkdir -p $(OBJ_DIR)/$(LAYER_SUBDIR)
	mkdir -p $(OBJ_DIR)/$(CALC_SUBDIR)
    
program: $(MIDDLE_OBJS) $(TARGETS)
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
	rm -rf $(OBJ_DIR)

# default rule for object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
	
$(OBJ_DIR)/led-user.o: $(SRC_DIR)/led-user.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<
	
$(BUILD_DIR)/led-user: $(OBJ_DIR)/led-user.o
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/led-user $(OBJ_DIR)/led-user.o
