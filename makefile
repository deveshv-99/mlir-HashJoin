
all: clean shared test

shared:
	@ clang++ -c ./shared_stuff/shared.cpp -g -O2 -fPIC -o ./shared_stuff/shared.o
	@ clang++ -shared ./shared_stuff/shared.o -o ./shared_stuff/shared.so


join_v1:
	./run_test.sh join_v1.mlir

join_v2:
	./run_test.sh join_v2.mlir


join_v1_ll:
	./run_test_ll.sh join_v1.mlir > join_v1.ll


clean:
	@ rm -f ./shared_stuff/shared.o
	@ rm -f ./shared_stuff/shared.so