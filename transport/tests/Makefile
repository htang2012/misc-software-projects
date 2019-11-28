.PHONY: all clean


all: test
test:
	g++ -I /usr/local/cuda-10.0/include -I ../include -g test.cc  -L../build/lib  -ltransport -ldl -lpthread -o test

clean:
	rm test    
