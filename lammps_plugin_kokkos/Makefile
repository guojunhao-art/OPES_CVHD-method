# Minimal GNU make build for the CVHD LAMMPS plugin.
#
# Usage:
#   make LAMMPS_SRC=/path/to/lammps/src CXX=mpicxx
#
# Use the same compiler/MPI wrapper and integer-size settings as the LAMMPS
# executable that will load this plugin.

LAMMPS_SRC ?= /home/guojunhao/software/lammps_test/lammps-22Jul2025/src
CXX = /home/guojunhao/software/lammps_test/lammps-22Jul2025/lib/kokkos/bin/nvcc_wrapper  -DLMP_KOKKOS -I/home/guojunhao/software/lammps_test/lammps-22Jul2025/lib/kokkos/core/src -I/home/guojunhao/software/lammps_test/lammps-22Jul2025/lib/kokkos/containers/src -I/home/guojunhao/software/lammps_test/lammps-22Jul2025/lib/kokkos/algorithms/src -I/home/guojunhao/software/lammps_test/lammps-22Jul2025/lib/kokkos/core/src -I/home/guojunhao/software/lammps_test/lammps-22Jul2025/BUILD_PLUMED/lib/kokkos -I/home/guojunhao/software/lammps_test/lammps-22Jul2025/lib/kokkos/tpls/desul/include -I/home/guojunhao/software/lammps_test/lammps-22Jul2025/BUILD_PLUMED/lib/kokkos/core/src -I/home/guojunhao/software/lammps_test/lammps-22Jul2025/lib/kokkos/tpls/mdspan/include
CXXFLAGS ?= -O2 -fPIC -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -arch=sm_70
LDFLAGS ?= -shared

TARGET = cvhdplugin.so
SRCS = fix_cvhd_global_distortion.cpp fix_cvhd_global_distortion_kokkos.cpp cvhd_global_distortion_plugin.cpp

all: $(TARGET)

$(TARGET): $(SRCS) fix_cvhd_global_distortion.h fix_cvhd_global_distortion_kokkos.h
	$(CXX) $(CXXFLAGS) -I$(LAMMPS_SRC) $(LDFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET) *.o
