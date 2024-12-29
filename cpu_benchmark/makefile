CXX = g++
CXXFLAGS = -std=c++17 -O3 -ftree-vectorize#for debugging purposes
OUTFLAGS = -o
OUTNAME = pass.out
INCLUDE = -Ieigen-3.4.0/ 
OMP = -fopenmp
LDFLAGS = -shared

INS = pass.cpp



run: 
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(INS) $(OMP) -o $(OUTNAME) 
	./$(OUTNAME)

clean:
	rm $(OUTNAME)
