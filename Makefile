CC      := gcc
NVCC    := nvcc

CUDA_HOME ?= /usr/local/cuda
SM        ?= 86                 
CINCLUDE := -Iinclude

CFLAGS   := -O2 -DTS=512 -DLOCAL_SIZE=256 -march=native -mavx2 -Wall -Wextra -fopenmp -pthread $(CINCLUDE)
NVFLAGS := -O2 -Xcompiler "-fopenmp -pthread" $(CINCLUDE) \
           -gencode arch=compute_90,code=compute_90


LDFLAGS := -L$(CUDA_HOME)/lib64
LDLIBS   := -lcudart -lgomp

SRCDIR := src
OBJDIR := build

CPU_SRCS := bitonic.c bitonic_simd_merge.c bitonic_cellsort.c helper.c main.c
CPU_SRCS := $(addprefix $(SRCDIR)/,$(CPU_SRCS))


GPU_SRC := $(SRCDIR)/bitonic_gpu.cu    


CPU_OBJS := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(CPU_SRCS))
GPU_OBJ  := $(OBJDIR)/$(notdir $(GPU_SRC:.c=.o))
GPU_OBJ  := $(GPU_OBJ:.cu=.o)

SORTNET_SRC := sort_network/sort_network.c
SORTNET_OBJ := $(OBJDIR)/sort_network.o

BIN_SORT             := sort
BIN_GENERATE_NETWORK := generate_network
BIN_TEST_NETWORK     := test_network


.PHONY: all clean debug dirs
all: dirs $(BIN_SORT) $(BIN_GENERATE_NETWORK) $(BIN_TEST_NETWORK)

dirs:
	@mkdir -p $(OBJDIR)


$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(GPU_OBJ): $(GPU_SRC)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(SORTNET_OBJ): $(SORTNET_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_SORT): $(CPU_OBJS) $(GPU_OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS) $(LDLIBS) -cudart static


$(BIN_GENERATE_NETWORK): $(SRCDIR)/generate_network.c
	$(CC) $(CFLAGS) $^ -o $@

$(BIN_TEST_NETWORK): $(SRCDIR)/sortnet_test.c $(SORTNET_OBJ) $(OBJDIR)/helper.o
	$(CC) $(CFLAGS) $^ -o $@

debug: CFLAGS := -g -O0 -DDEBUG -march=native -mavx2 -Wall -Wextra -fopenmp -pthread $(CINCLUDE)
debug: NVFLAGS := -g -O0 -arch=sm_$(SM) -Xcompiler "-g -O0 -fopenmp -pthread" $(CINCLUDE)
debug: clean all

clean:
	rm -rf $(OBJDIR) $(BIN_SORT) $(BIN_GENERATE_NETWORK) $(BIN_TEST_NETWORK)
