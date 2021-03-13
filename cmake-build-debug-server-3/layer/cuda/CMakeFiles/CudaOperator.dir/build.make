# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cbd109-3/Users/lifang/tinynn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3

# Include any dependencies generated for this target.
include layer/cuda/CMakeFiles/CudaOperator.dir/depend.make

# Include the progress variables for this target.
include layer/cuda/CMakeFiles/CudaOperator.dir/progress.make

# Include the compile flags for this target's objects.
include layer/cuda/CMakeFiles/CudaOperator.dir/flags.make

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o: layer/cuda/CMakeFiles/CudaOperator.dir/flags.make
layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o: ../layer/cuda/absval_cuda.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o"
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o -c /home/cbd109-3/Users/lifang/tinynn/layer/cuda/absval_cuda.cpp

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CudaOperator.dir/absval_cuda.cpp.i"
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cbd109-3/Users/lifang/tinynn/layer/cuda/absval_cuda.cpp > CMakeFiles/CudaOperator.dir/absval_cuda.cpp.i

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CudaOperator.dir/absval_cuda.cpp.s"
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cbd109-3/Users/lifang/tinynn/layer/cuda/absval_cuda.cpp -o CMakeFiles/CudaOperator.dir/absval_cuda.cpp.s

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o.requires:

.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o.requires

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o.provides: layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o.requires
	$(MAKE) -f layer/cuda/CMakeFiles/CudaOperator.dir/build.make layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o.provides.build
.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o.provides

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o.provides.build: layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o


layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o: layer/cuda/CMakeFiles/CudaOperator.dir/flags.make
layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o: ../layer/cuda/absval_cuda.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o"
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda && /usr/local/cuda-10.2/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/cbd109-3/Users/lifang/tinynn/layer/cuda/absval_cuda.cu -o CMakeFiles/CudaOperator.dir/absval_cuda.cu.o

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/CudaOperator.dir/absval_cuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/CudaOperator.dir/absval_cuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o.requires:

.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o.requires

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o.provides: layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o.requires
	$(MAKE) -f layer/cuda/CMakeFiles/CudaOperator.dir/build.make layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o.provides.build
.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o.provides

layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o.provides.build: layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o


layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o: layer/cuda/CMakeFiles/CudaOperator.dir/flags.make
layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o: ../layer/cuda/test_cuda.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o"
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda && /usr/local/cuda-10.2/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/cbd109-3/Users/lifang/tinynn/layer/cuda/test_cuda.cu -o CMakeFiles/CudaOperator.dir/test_cuda.cu.o

layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/CudaOperator.dir/test_cuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/CudaOperator.dir/test_cuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o.requires:

.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o.requires

layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o.provides: layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o.requires
	$(MAKE) -f layer/cuda/CMakeFiles/CudaOperator.dir/build.make layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o.provides.build
.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o.provides

layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o.provides.build: layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o


# Object files for target CudaOperator
CudaOperator_OBJECTS = \
"CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o" \
"CMakeFiles/CudaOperator.dir/absval_cuda.cu.o" \
"CMakeFiles/CudaOperator.dir/test_cuda.cu.o"

# External object files for target CudaOperator
CudaOperator_EXTERNAL_OBJECTS =

layer/cuda/libCudaOperator.a: layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o
layer/cuda/libCudaOperator.a: layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o
layer/cuda/libCudaOperator.a: layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o
layer/cuda/libCudaOperator.a: layer/cuda/CMakeFiles/CudaOperator.dir/build.make
layer/cuda/libCudaOperator.a: layer/cuda/CMakeFiles/CudaOperator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libCudaOperator.a"
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda && $(CMAKE_COMMAND) -P CMakeFiles/CudaOperator.dir/cmake_clean_target.cmake
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CudaOperator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
layer/cuda/CMakeFiles/CudaOperator.dir/build: layer/cuda/libCudaOperator.a

.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/build

layer/cuda/CMakeFiles/CudaOperator.dir/requires: layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cpp.o.requires
layer/cuda/CMakeFiles/CudaOperator.dir/requires: layer/cuda/CMakeFiles/CudaOperator.dir/absval_cuda.cu.o.requires
layer/cuda/CMakeFiles/CudaOperator.dir/requires: layer/cuda/CMakeFiles/CudaOperator.dir/test_cuda.cu.o.requires

.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/requires

layer/cuda/CMakeFiles/CudaOperator.dir/clean:
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda && $(CMAKE_COMMAND) -P CMakeFiles/CudaOperator.dir/cmake_clean.cmake
.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/clean

layer/cuda/CMakeFiles/CudaOperator.dir/depend:
	cd /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cbd109-3/Users/lifang/tinynn /home/cbd109-3/Users/lifang/tinynn/layer/cuda /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3 /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda /home/cbd109-3/Users/lifang/tinynn/cmake-build-debug-server-3/layer/cuda/CMakeFiles/CudaOperator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : layer/cuda/CMakeFiles/CudaOperator.dir/depend

