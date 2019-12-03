.PHONY: all clean


run: bootstrap #test bootstrap
	g++ -L../build/lib -ltransport -ldl -lpthread -g run

test.o:
	g++ -I /usr/local/cuda-10.0/include -I ../include -c test.cc 

bootstrap.o:
	g++ -I /usr/local/cuda-10.0/include -I ../include -c bootstrap.cc
clean:
	rm run    
