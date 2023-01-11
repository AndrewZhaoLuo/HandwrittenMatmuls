benchmark-peak: run run.asm 
	likwid-perfctr -f -m -g FLOPS_SP -C N:0-1 ./run 0

run: matvec.cc 
	clang matvec.cc -O3 -march=znver2 -llikwid -o run 

run.asm: matvec.cc 
	clang matvec.cc -march=znver2 -O3 -S -o run.asm
