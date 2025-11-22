NVCC := nvcc
NVCC_FLAGS := -dlto -arch=native -std=c++17 -O3
LIBS := -L../dependencies/cupqc-pkg-0.2.0/lib/ -lcupqc -lnvidia-ml
INCLUDES := -I../dependencies/cupqc-pkg-0.2.0/include/ -I../dependencies/cupqc-pkg-0.2.0/include/cupqc

SRC := $(wildcard *.cu)
BIN := $(patsubst %.cu,%,$(SRC))

all: $(BIN)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(LIBS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BIN)

.PHONY: all clean