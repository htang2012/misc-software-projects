.PHONY: all clean

all: test
test:
	g++ -I ../../nccl/build/include -I /usr/local/cuda-10.0/include -g test.cc  -L/home/htang/git/misc-software-projects/nccl/build/lib  -lnccl -o test

clean:
	rm test
     
