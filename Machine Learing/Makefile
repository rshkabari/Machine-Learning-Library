CXX = g++
CXXFLAGS = -std=c++11 -Wall
SRC = $(wildcard *.cpp)
OBJ = $(SRC:%.cpp=%.o)
DEP = $(OBJ:%.o=%.d)

all: library

-include $(DEP)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

library: $(OBJ)
	ar rcs libml.a $(OBJ)

clean:
	rm -f $(DEP) libml.al
	