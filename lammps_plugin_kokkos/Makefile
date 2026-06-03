# Minimal GNU make build for the CVHD LAMMPS plugin.
#
# Usage:
#   make LAMMPS_SRC=/path/to/lammps/src CXX=mpicxx
#
# Use the same compiler/MPI wrapper and integer-size settings as the LAMMPS
# executable that will load this plugin.

LAMMPS_SRC ?= /path/to/lammps/src
CXX = mpicxx
KOKKOS_DEFS ?=
CUDA_ARCH ?=
CXXFLAGS ?= -O2 -fPIC -std=c++17 $(KOKKOS_DEFS) $(CUDA_ARCH)
LDFLAGS ?= -shared

TARGET = cvhdplugin.so
SRCS = fix_cvhd_global_distortion.cpp fix_cvhd_global_distortion_kokkos.cpp cvhd_global_distortion_plugin.cpp

all: $(TARGET)

$(TARGET): $(SRCS) fix_cvhd_global_distortion.h fix_cvhd_global_distortion_kokkos.h
	$(CXX) $(CXXFLAGS) -I$(LAMMPS_SRC) $(LDFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET) *.o
