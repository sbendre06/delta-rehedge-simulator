CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
TARGET   = hedger
SOURCES  = main.cpp simulator.cpp
HEADERS  = greeks.hpp

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) -lm

clean:
	rm -f $(TARGET)
