SRC_DIR := src
BUILD_DIR := build
MODULE_DIR := modules
INCLUDE_DIR := include
OBJ_DIR := obj

LAYER_SUBDIR := layers
CALC_SUBDIR := calc
EXTLIB_SUBDIR := extlib
UTIL_SUBDIR := utils
TEST_SUBDIR := test
IMAGE_SUBDIR := image

TARGETS := $(BUILD_DIR)/led-user $(BUILD_DIR)/test_nn $(BUILD_DIR)/test_load_image \
	$(BUILD_DIR)/test_search_face
EXTLIB_OBJS := $(addprefix $(OBJ_DIR)/, $(EXTLIB_SUBDIR)/jsoncpp.o)
NEURAL_NET_OBJS := $(EXTLIB_OBJS) $(addprefix $(OBJ_DIR)/, $(CALC_SUBDIR)/calc-cpu.o \
	$(CALC_SUBDIR)/util-functions.o \
	$(LAYER_SUBDIR)/layer_data.o \
	$(LAYER_SUBDIR)/sigmoid_layer.o \
	$(LAYER_SUBDIR)/max_pool_layer.o \
	$(LAYER_SUBDIR)/conv_layer.o \
	$(LAYER_SUBDIR)/layer_factory.o \
	$(LAYER_SUBDIR)/layer_merger.o \
	$(IMAGE_SUBDIR)/image.o \
	$(IMAGE_SUBDIR)/image_util.o \
	network.o)
MIDDLE_OBJS := $(NEURAL_NET_OBJS) $(addprefix $(OBJ_DIR)/, led-user.o \
	$(TEST_SUBDIR)/test_load_image.o $(TEST_SUBDIR)/test_nn.o)

MIDDLE_OBJS_DEP = $(MIDDLE_OBJS:.o=.d)

.PHONY: all clean directory program

# in what kind of OS are these sources built?
ifeq ($(OS),Windows_NT)
TARGET_OS := WIN32
TARGET_ARCH := x86
ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
TARGET_ARCH := x64
endif
ifeq ($(PROCESSOR_ARCHITEW6432),AMD64)
TARGET_ARCH := x64
endif
ifeq ($(TARGET_ARCH),AMD64)
LIBRARY_DIR := lib/x86_64
endif
else
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Linux)
TARGET_OS := LINUX
endif
ifeq ($(UNAME_S),Darwin)
TARGET_OS := OSX
endif
ifeq ($(UNAME_M),x86_64)
TARGET_ARCH := x64
LIBRARY_DIR := lib/x86_64
endif
ifeq ($(UNAME_M),aarch64)
TARGET_ARCH := aarch64
LIBRARY_DIR := lib/aarch64
endif
endif
export TARGET_OS

CXXFLAGS := -std=c++0x -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lnetpbm -Wl,-rpath=$(shell pwd)/$(LIBRARY_DIR) -Wall -g
DEPEND_FLAGS := -MMD -MP 

all: directory program

directory:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/$(EXTLIB_SUBDIR)
	mkdir -p $(OBJ_DIR)/$(LAYER_SUBDIR)
	mkdir -p $(OBJ_DIR)/$(CALC_SUBDIR)
	mkdir -p $(OBJ_DIR)/$(UTIL_SUBDIR)
	mkdir -p $(OBJ_DIR)/$(TEST_SUBDIR)
	mkdir -p $(OBJ_DIR)/$(IMAGE_SUBDIR)

program: $(MIDDLE_OBJS) $(TARGETS)
ifeq ($(TARGET_OS),LINUX)
ifneq ($(INCLUDE_MODULE),)
	# we build kernel modules only in Linux environment
	$(MAKE) -C $(MODULE_DIR) all
endif
endif

clean:
ifeq ($(TARGET_OS),LINUX)
ifneq ($(INCLUDE_MODULE),)
	# we clean kernel modules only in Linux environment
	$(MAKE) -C $(MODULE_DIR) clean
endif
endif
	rm -rf $(TARGETS) $(MIDDLE_OBJS) $(MIDDLE_OBJS_DEP)

# default rule for object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(DEPEND_FLAGS) -MT $@ -MF $(patsubst %.o,%.d,$@) $<

$(OBJ_DIR)/led-user.o: $(SRC_DIR)/led-user.c
	$(CXX) -c -o $@ $(CXXFLAGS) $(DEPEND_FLAGS) -MT $@ -MF $(patsubst %.o,%.d,$@) $<

$(BUILD_DIR)/led-user: $(OBJ_DIR)/led-user.o
	$(CXX) -o $@ $(CXXFLAGS) $(DEPEND_FLAGS) -MT $@ -MF $(patsubst %.o,%.d,$@) $^

$(BUILD_DIR)/test_nn: $(OBJ_DIR)/$(TEST_SUBDIR)/test_nn.o $(NEURAL_NET_OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(DEPEND_FLAGS) -MT $@ -MF $(patsubst %.o,%.d,$@) $^

$(BUILD_DIR)/test_load_image: $(OBJ_DIR)/$(IMAGE_SUBDIR)/image.o $(OBJ_DIR)/$(TEST_SUBDIR)/test_load_image.o
	$(CXX) -o $@ $(CXXFLAGS) $(DEPEND_FLAGS) -MT $@ -MF $(patsubst %.o,%.d,$@) $^

$(BUILD_DIR)/test_search_face: $(OBJ_DIR)/$(TEST_SUBDIR)/test_search_face.o \
		$(NEURAL_NET_OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(DEPEND_FLAGS) -MT $@ -MF $(patsubst %.o,%.d,$@) $^

-include $(MIDDLE_OBJS_DEP)
