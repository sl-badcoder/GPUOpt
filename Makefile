CC      := gcc
NVCC    := nvcc

CUDA_HOME ?= /usr/local/cuda

# List the SMs you want *native* cubins for (adjust to your fleet)
CUDA_ARCHS ?= 80;86;89;90

# Generate -gencode lines for native cubins
GENCODE_NATIVE := $(foreach A,$(subst ;, ,$(CUDA_ARCHS)),-gencode arch=compute_$(A),code=sm_$(A))

# Keep one PTX target for forward JIT on newer GPUs (pick the highest you can compile)
GENCODE_PTX := -gencode arch=compute_90,code=compute_90

CINCLUDE  := -Iinclude

CFLAGS  := -O3 -DTS=512 -DLOCAL_SIZE=256 -march=native -mavx512f -mavx512bw -mavx512vl -Wall -Wextra -fopenmp -pthread $(CINCLUDE) -MMD -MP
NVFLAGS := -O3 -std=c++14 --expt-extended-lambda \
           -Xcompiler "-fopenmp -pthread" $(CINCLUDE) \
           $(GENCODE_NATIVE) $(GENCODE_PTX) -MMD -MP

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
debug: NVFLAGS := -g -O0 -std=c++14 --expt-extended-lambda \
                  -Xcompiler "-g -O0 -fopenmp -pthread" $(CINCLUDE) \
                  $(GENCODE_NATIVE) $(GENCODE_PTX) -MMD -MP
debug: clean all

clean:
	rm -rf $(OBJDIR) $(BIN_SORT)
