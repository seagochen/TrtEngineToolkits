# Specify the CUDA compiler
NVCC = /usr/local/cuda/bin/nvcc

# Specify the architecture and other flags
CUDA_ARCH = -arch=sm_75  # Adjust based on your target GPU architecture
CUDA_FLAGS = -Xcompiler -fPIC

# Specify source and object files
SRC_FILES = $(wildcard *.cu)
OBJ_FILES = $(SRC_FILES:.cu=.o)

# Specify the output library name
LIB_NAME = libcumatrix.a  # For a static library (.a), or .so for shared library

# Build the library
$(LIB_NAME): $(OBJ_FILES)
	$(AR) rcs $@ $^

# Compile CUDA files into object files
%.o: %.cu
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) -c $< -o $@

# Clean up the build files
clean:
	rm -f $(OBJ_FILES) $(LIB_NAME)
