
maca_cc=$(MACA_PATH)tools/cu-bridge/bin/cucc
cpp=/usr/bin/g++
$(info ${maca_cc})
cuda_cc=/usr/local/cuda/bin/nvcc


include= -I $(MACA_PATH)tools/cu-bridge/include/ \
		-I ./inc/

cudaInclude= -I ./inc/

LIBS=-lmcruntime -lmxc-runtime64 -lcurand

CFLAG=-D_MACACC_RTC__ -D__FAST_HALF_CVT__ -O3 -arch=sm_80 -use-dim-intrinsic -mllvm -metaxgpu-new-streg-abi=true --fast-bfloat16-math rz -DUSE_MACA
CUDAFLAG=-arch=sm_80
# objs=$(wildcard ./src/*.cu)
# 
# objs=./src/concat_3d_layer_test.cu 
objs=./src/tmp_test.cu 

asm:$(objs)
	$(maca_cc) $(CFLAG)  $(include) $^ -maca-device-only $(LIBS) -aop -S -o test.asm
test_maca:$(objs)
	$(maca_cc) $(CFLAG) $(include) $^ -o test_maca $(LIBS)
#main.o:main.cu
#	$(cc) $(include) $^ -o $@
test_cuda:$(objs)
	$(cuda_cc) $(CUDAFLAG) $(cudaInclude) $^ -o test_cuda
test:
	$(cpp) ./src/test.cpp -o test
clean:
	$(RM) *.o *.asm *.s *.bc *.macai *.out *.pftrace *.mcfb *.dat_dpc_0 *.json test_maca test_cuda test -r


