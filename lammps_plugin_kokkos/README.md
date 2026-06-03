# OPES_CVHD LAMMPS Plugin

This directory contains the experimental LAMMPS plugin implementation of
`fix cvhd/global/distortion` and `fix cvhd/global/distortion/kk` used for
large-system OPES_CVHD testing.

The current code corresponds to the **V3k-hybrid-perf** development line.

## Backend design

### Raw CV calculation

```text
ccbb / chbb breaking CVs  -> Kokkos reference-pair cache reductions
ccbf formation CV         -> Kokkos fused neighbor-list reduction
```

Breaking CVs are deliberately **not** computed from the current neighbor-list
traversal.  A broken reference bond may move outside the current neighbor list,
but it must not disappear from the breaking CV before the CVHD reset/wait logic
has handled the event.

### Bias force application

```text
ccbb / chbb / ccbf forces -> one fused neighbor-list Kokkos force kernel
```

The force kernel is local by construction.  It reuses
`cc_form_candidate_rmax` as a local force cutoff for all fused-force
components before calling `sqrt()`:

```cpp
if (rsq <= 0.0 || (cand2 > 0.0 && rsq >= cand2)) continue;
```

This is a force cutoff, not a CV cutoff.  Breaking CVs remain reference-pair
based.

## Important implementation notes

1. **LAMMPS 22Jul2025 Kokkos neighbor-list request**

   This LAMMPS version does not provide `NeighConst::REQ_KOKKOS_DEVICE`.
   The `/kk` subclass requests a Kokkos device neighbor list by calling:

   ```cpp
   FixCvhdGlobalDistortion::init();
   auto *request = neighbor->find_request(this);
   request->set_kokkos_host(0);
   request->set_kokkos_device(1);
   ```

   The startup neighbor-list section should show the CVHD fix using
   `kokkos_device` and `bin: kk/device`.

2. **CUDA extended-lambda access rule**

   `backend_collect_reference_pairs()` contains `KOKKOS_LAMBDA` kernels and is
   therefore declared `public`.  NVCC rejects extended `__host__ __device__`
   lambdas inside protected/private member functions.

3. **Host tag-map caching**

   `build_tag_map()` is no longer called every `compute_all()` in the Kokkos
   backend.  The host tag map is rebuilt only when dirty, for example after a
   neighbor rebuild/local-index change or before reference/pair-cache rebuilds.

4. **Reference-pair building**

   `build_reference_pairs()` first lets the Kokkos backend collect reference
   pairs from the device neighbor list.  If the Kokkos list is unavailable, it
   falls back to a host geometric scan.

5. **Current limitation**

   The device-side minimum-image implementation is currently for orthogonal
   boxes.

## Build

The plugin should be compiled against the same LAMMPS source/build used to run
the simulation.  For the `/kk` style, compile with `-DLMP_KOKKOS` and the CUDA
architecture for the target GPU.

Example for a V100 build:

```bash
make clean
make
```

The provided Makefile is intentionally minimal; update the LAMMPS paths and
CUDA architecture flags if your local build differs.  For V100, the architecture
flag should include:

```bash
-arch=sm_70
```

For RTX 40-series GPUs, use the appropriate architecture, e.g. `sm_89`.

## Minimal LAMMPS usage

```lammps
plugin load ./cvhdplugin.so

atom_modify sort 0 0.0

fix cvhd all cvhd/global/distortion/kk cvhd_global_distortion_v3k.cfg
```

For Kokkos/ReaxFF runs, launch LAMMPS with the usual Kokkos options, for example:

```bash
CUDA_VISIBLE_DEVICES=0 mpirun -n 1 lmp -k on g 1 -sf kk -in in.CHO.lmp
```

## Useful validation checks

At startup, reference counts for 400 C10H22 molecules should be approximately:

```text
C-C reference pairs = 3600
C-H reference pairs = 8800
connected C-C bonds = 3600
```

In timer output for the hybrid Kokkos backend, the expected pattern is:

```text
tag_map     low / not every step
kk_reduce   active for ccbb/chbb reference-pair reductions
fused_cv    active for ccbf formation
form_cpu    0 in /kk fused mode
fused_force active only when bias derivatives are nonzero
```

## Files

```text
fix_cvhd_global_distortion.h/.cpp          CPU/base implementation
fix_cvhd_global_distortion_kokkos.h/.cpp   Kokkos subclass
cvhd_global_distortion_plugin.cpp          plugin registration
cvhd_global_distortion_v3k.cfg             example configuration
Makefile                                   minimal GNU make build
CMakeLists.txt                             CMake build scaffold
```
