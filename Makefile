CC      := gcc
NVCC    := nvcc

CUDA_HOME ?= /usr/local/cuda
SM        ?= 86
CINCLUDE  := -Iinclude

CFLAGS  := -O2 -DTS=512 -DLOCAL_SIZE=256 -march=native -mavx2 -Wall -Wextra -fopenmp -pthread $(CINCLUDE) -MMD -MP
NVFLAGS := -O2 -Xcompiler "-fopenmp -pthread" $(CINCLUDE) \
           -gencode arch=compute_$(SM),code=sm_$(SM) -MMD -MP

LDFLAGS := -L$(CUDA_HOME)/lib64
LDLIBS  := -lcudart -lgomp

SRCDIR := src
OBJDIR := build

CPU_SRCS  := $(wildcard $(SRCDIR)/cpu/*.c)
CORE_SRCS := $(wildcard $(SRCDIR)/core/*.c)
MAIN_SRCS := $(wildcard $(SRCDIR)/*.c)
GPU_SRCS  := $(wildcard $(SRCDIR)/gpu/*.cu)

CPU_OBJS  := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(CPU_SRCS))
CORE_OBJS := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(CORE_SRCS))
MAIN_OBJS := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(MAIN_SRCS))
GPU_OBJS  := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(GPU_SRCS))
OBJS      := $(CPU_OBJS) $(CORE_OBJS) $(MAIN_OBJS) $(GPU_OBJS)

BIN_SORT := sort

.PHONY: all clean debug

all: $(BIN_SORT)

$(BIN_SORT): $(OBJS)
	$(NVCC) -o $@ $^ $(LDFLAGS) $(LDLIBS) -cudart static

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c $< -o $@

-include $(OBJS:.o=.d)

debug: CFLAGS  := -g -O0 -DDEBUG -march=native -mavx2 -Wall -Wextra -fopenmp -pthread $(CINCLUDE) -MMD -MP
debug: NVFLAGS := -g -O0 -arch=sm_$(SM) -Xcompiler "-g -O0 -fopenmp -pthread" $(CINCLUDE) -MMD -MP
debug: clean all

clean:
	rm -rf $(OBJDIR) $(BIN_SORT)