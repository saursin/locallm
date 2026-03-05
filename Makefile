BUILD_DIR := llama.cpp/build
NPROC     := $(shell nproc)

# Optimized for Intel i7-13700H (Raptor Lake):
#   AVX2, FMA, F16C, BMI2 enabled by default via GGML
#   AVX-VNNI: explicitly enabled (off by default, supported on this CPU)
#   NATIVE:   -march=native to catch all microarch optimizations
#   LTO:      link-time optimization for extra speed
CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=Release \
	-DGGML_NATIVE=ON \
	-DGGML_AVX_VNNI=ON \
	-DGGML_LTO=ON \
	-DLLAMA_BUILD_TESTS=OFF

STAMP := $(BUILD_DIR)/.configured

.PHONY: all build reconfigure clean

all: build

$(STAMP):
	cmake -B $(BUILD_DIR) -S llama.cpp $(CMAKE_FLAGS)
	@touch $(STAMP)

build: $(STAMP)
	cmake --build $(BUILD_DIR) --config Release -j$(NPROC)
	@echo ""
	@echo "  llama-server: $(BUILD_DIR)/bin/llama-server"

reconfigure:
	rm -f $(STAMP)
	$(MAKE) build

clean:
	rm -rf $(BUILD_DIR)
