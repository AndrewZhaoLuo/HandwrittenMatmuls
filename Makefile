
CXX := clang++

INCLUDES := -I./lib 
OPT := -O3
CXXFLAGS := $(OPT) $(DEBUG) $(INCLUDES) -std=c++17
LDFLAGS := -O3

OUTPUT_DIR = build
OBJECTS = $(patsubst %.cpp, $(OUTPUT_DIR)/%.o, $(ALL_SRC)) 

default : all

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OBJECTS): | $(OUTPUT_DIR)
$(OUTPUT_DIR)/%.o : lib/%.cpp $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) $(DEFINES) -c $< -o $@ 

main: main.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(DEFINES) $< -o build/$@

main.asm: main.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) -g -S $(DEFINES) $< -o build/$@

all : main
clean :
	rm -rf $(OUTPUT_DIR) 
.PHONY : all clean main