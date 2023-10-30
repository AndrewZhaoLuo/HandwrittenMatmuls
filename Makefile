
CXX := clang++

INCLUDES := -I./lib 
OPT := -O3
CXXFLAGS := $(OPT) $(DEBUG) $(INCLUDES) -std=c++17
LDFLAGS := -O3

OUTPUT_DIR = build

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/%.o : lib/%.cpp $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) $(DEFINES) -c $< -o $@

% : $(OUTPUT_DIR)/%.o
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $(OUTPUT_DIR)/$@