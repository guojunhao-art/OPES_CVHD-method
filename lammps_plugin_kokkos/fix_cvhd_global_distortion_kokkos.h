/* ----------------------------------------------------------------------
   USER-CVHD: V3k experimental /kk subclass for LAMMPS

   V3k hybrid performance backend:
     - breaking CVs (ccbb/chbb): Kokkos reference-pair reductions
     - formation CV (ccbf): Kokkos fused neighbor-list reduction
     - bias forces: fused local Kokkos neighbor-list kernel
     - host tag map: dirty/on-demand caching
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(cvhd/global/distortion/kk,FixCvhdGlobalDistortionKokkos);
// clang-format on
#else

#ifndef LMP_FIX_CVHD_GLOBAL_DISTORTION_KOKKOS_H
#define LMP_FIX_CVHD_GLOBAL_DISTORTION_KOKKOS_H

#include "fix_cvhd_global_distortion.h"
#include "atom_masks.h"

#ifdef LMP_KOKKOS
#include "KOKKOS/atom_kokkos.h"
#include "KOKKOS/kokkos_base.h"
#include "KOKKOS/kokkos_type.h"
#include "KOKKOS/neigh_list_kokkos.h"
#include "Kokkos_Core.hpp"
#endif

namespace LAMMPS_NS {

class FixCvhdGlobalDistortionKokkos : public FixCvhdGlobalDistortion {
 public:
  FixCvhdGlobalDistortionKokkos(class LAMMPS *, int, char **);
  ~FixCvhdGlobalDistortionKokkos() override = default;

  void init() override;
  void init_list(int, class NeighList *) override;
  void post_neighbor() override;

  // This function launches KOKKOS_LAMBDA kernels through NVCC extended lambdas.
  // It must be public; otherwise nvcc rejects the enclosing member function.
  bool backend_collect_reference_pairs(const TermConfig &term,
                                       std::vector<PairRef> &local_pairs) override;

 protected:
  bool backend_can_skip_tag_map_each_compute() const override;
  void backend_sync_host_atoms_for_tag_map() override;
  void backend_sync_host_atoms_for_reference() override;
  void backend_on_reference_pairs_changed() override;
  void backend_on_connectivity_changed() override;
  void backend_compute_raw_cvs(double &ccbb, double &chbb, double &ccbf) override;
  void backend_apply_bias_forces() override;

 public:
#ifdef LMP_KOKKOS
  // CUDA extended lambdas cannot be placed inside private/protected member
  // functions.  Keep the kernel-launching helpers public in this validation
  // version.  A later cleanup should replace KOKKOS_LAMBDA with public functor
  // structs so these helpers can be private again.
  using DeviceType = LMPDeviceType;

  AtomKokkos *atomKK = nullptr;
  NeighListKokkos<DeviceType> *k_list_ = nullptr;

  struct DevicePairCache {
    Kokkos::View<int*, DeviceType> i;
    Kokkos::View<int*, DeviceType> j;
    Kokkos::View<double*, DeviceType> ref;

    // count[p] = 1 means this rank owns pair.itag and should include this pair
    // in the global breaking reduction.  Force application uses all cached
    // pairs whose i or j is owned by this rank.
    Kokkos::View<int*, DeviceType> count;
    int n = 0;
  };

  struct FusedNeighborCache {
    Kokkos::View<int*, DeviceType> local_to_cidx;
    Kokkos::View<int*, DeviceType> cc_partner_count;
    Kokkos::View<int**, DeviceType> cc_partner_cidx;
    Kokkos::View<int*, DeviceType> ch_partner_count;
    Kokkos::View<tagint**, DeviceType> ch_partner_tag;
    int nall = 0;
    int ncarbon = 0;
    int max_cc = 0;
    int max_ch = 0;
  };

  DevicePairCache cc_pairs_dev_;
  DevicePairCache ch_pairs_dev_;
  FusedNeighborCache fused_dev_;

  bool pair_cache_dirty_ = true;
  bool fused_cache_dirty_ = true;
  bigint pair_cache_last_build_step_ = -1;
  int pair_cache_last_nlocal_ = -1;
  int pair_cache_last_nghost_ = -1;

  void mark_pair_cache_dirty();
  void mark_fused_cache_dirty();
  void ensure_pair_caches_current();
  void ensure_fused_cache_current();
  void update_fused_cache_from_host();
  void compute_fused_raw_cvs_kokkos(double &ccbb, double &chbb, double &ccbf);
  void apply_fused_forces_kokkos(double ccbb, double chbb, double ccbf,
                                 double dVdccbb, double dVdchbb, double dVdccbf);
  double compute_breaking_value_kokkos(const TermConfig &term, DevicePairCache &cache);
  void apply_breaking_forces_kokkos(const TermConfig &term, double raw_value, double dV_draw,
                                    DevicePairCache &cache);
  void update_pair_cache_from_tags(const TermConfig &term, DevicePairCache &cache,
                                   const char *label);
#endif
};

}    // namespace LAMMPS_NS

#endif
#endif
