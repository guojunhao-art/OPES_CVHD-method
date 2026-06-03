cmake_minimum_required(VERSION 3.16)

project(cvhd_lammps_plugin LANGUAGES CXX)

set(LAMMPS_SOURCE_DIR "" CACHE PATH "Path to the LAMMPS src directory, e.g. /path/to/lammps/src")

if(NOT LAMMPS_SOURCE_DIR)
  message(FATAL_ERROR "Please set -DLAMMPS_SOURCE_DIR=/path/to/lammps/src")
endif()

if(NOT EXISTS "${LAMMPS_SOURCE_DIR}/lammps.h")
  message(FATAL_ERROR "LAMMPS_SOURCE_DIR does not look like a LAMMPS src directory: ${LAMMPS_SOURCE_DIR}")
endif()

add_library(cvhdplugin MODULE
  fix_cvhd_global_distortion.cpp
  fix_cvhd_global_distortion_kokkos.cpp
  cvhd_global_distortion_plugin.cpp
)

target_include_directories(cvhdplugin PRIVATE
  "${LAMMPS_SOURCE_DIR}"
)

target_compile_features(cvhdplugin PRIVATE cxx_std_17)

# Produce cvhdplugin.so rather than libcvhdplugin.so on Linux.
set_target_properties(cvhdplugin PROPERTIES
  PREFIX ""
  OUTPUT_NAME "cvhdplugin"
)

# The symbols from LAMMPS are resolved when the plugin is loaded by the
# LAMMPS executable/library. This keeps the plugin independent of a specific
# liblammps path during early debugging.
if(UNIX AND NOT APPLE)
  target_link_options(cvhdplugin PRIVATE "-Wl,--unresolved-symbols=ignore-in-shared-libs")
endif()
