NVCC := nvcc
NVCC_FLAGS := -dlto -arch=native -std=c++17 -O3
LIBS := -L../lib/ -lcupqc
INCLUDES := -I../include/ -I../include/cupqc

SRC := $(wildcard *.cu)
BIN := $(patsubst %.cu,%,$(SRC))

all: $(BIN)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(LIBS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BIN)

.PHONY: all clean
