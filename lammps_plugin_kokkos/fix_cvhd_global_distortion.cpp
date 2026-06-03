/* ----------------------------------------------------------------------
   USER-CVHD: V3k-hybrid-perf reference implementation of global-distortion CVs for LAMMPS

   CPU validation target:
     - builds C-C and C-H reference-bond lists from the LAMMPS neighbor list
     - computes the C-C formation term from the LAMMPS neighbor list
     - supports MPI by using local sums + MPI_Allreduce
     - computes raw ccbb/chbb/ccbf global-distortion CVs
     - computes CVHDM-style projected cv1/cv2 variables
     - applies a simple harmonic bias on cv1 to validate breaking forces

   V3k-hybrid-perf uses hybrid Kokkos CVs (reference-pair breaking, fused-neighbor formation), fused local force application, dirty tag-map caching, and an optional Kokkos reference-pair collector.
------------------------------------------------------------------------- */

#include "fix_cvhd_global_distortion.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "utils.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {
constexpr double OPES_LOG_WEIGHT_MAX = 350.0;

inline double clamp_value(double x, double lo, double hi)
{
  return std::max(lo, std::min(hi, x));
}
} // namespace


FixCvhdGlobalDistortion::FixCvhdGlobalDistortion(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg)
{
  if (narg < 4) {
    error->all(FLERR,
               "Illegal fix cvhd/global/distortion command. "
               "Usage: fix ID group cvhd/global/distortion CONFIG_FILE");
  }

  set_defaults();

  if (narg == 4) {
    config_file_ = arg[3];
  } else if (narg == 5 && std::strcmp(arg[3], "config") == 0) {
    config_file_ = arg[4];
  } else {
    error->all(FLERR,
               "Illegal fix cvhd/global/distortion command. "
               "Usage: fix ID group cvhd/global/distortion [config] CONFIG_FILE");
  }

  read_config(config_file_.c_str());
  validate_config();

  // Store the user-specified initial barrier.  CVHD event resets restore
  // opes_barrier_ to this value, while adaptive-E updates can still increase
  // the current barrier between resets.
  opes_barrier_initial_ = opes_barrier_;
  initialize_opes_state();

  vector_flag = 1;
  size_vector = 26;
  scalar_flag = 1;
  global_freq = 1;
  extvector = 0;
  extscalar = 0;
  energy_global_flag = 1;

  last_compute_step_ = -1;
  initialized_ = false;
  summary_printed_ = false;
  list_ = nullptr;
  last_cc_form_pair_count_ = 0;

  for (double &v : values_) v = 0.0;
}

FixCvhdGlobalDistortion::~FixCvhdGlobalDistortion()
{
  if (ofs_.is_open()) ofs_.close();
}

void FixCvhdGlobalDistortion::set_defaults()
{
  carbon_type_ = -1;
  hydrogen_type_ = -1;

  output_file_ = "cvhd_v3k.colvar";
  output_stride_ = 100;
  compute_stride_ = 1;

  kk_pair_cache_policy_ = "safe";

  kk_fused_neighbor_enable_ = true;
  kk_max_cc_partners_ = 8;
  kk_max_ch_partners_ = 8;

  tag_map_dirty_ = true;

  cvhd_timer_enable_ = false;
  cvhd_timer_stride_ = 1000;
  cvhd_timer_reset_each_print_ = true;
  cvhd_timer_last_print_step_ = -1;
  cvhd_timer_reset_accumulators();

  cvhd_power_ = 6.0;
  cv1_cutoff_cc_ = 0.4194;
  cv1_cutoff_ch_ = 0.6000;
  cv2_cutoff_ccform_ = 0.3200;

  // Pairs with r > cc_form_ref have zero ccbf contribution.  A candidate
  // upper cutoff larger than cc_form_ref therefore only reduces unnecessary
  // work and does not change the CV value.  Set to 0 to disable.
  cc_form_candidate_rmax_ = 4.0;

  bias_enable_ = false;
  opes_use_cv2_ = true;
  opes_ncv_ = 2;

  opes_temperature_ = 1500.0;
  opes_kbt_ = force->boltz * opes_temperature_;
  opes_barrier_ = 10.0;
  opes_barrier_initial_ = 10.0;
  opes_barrier_increment_ = 0.0;
  opes_barrier_max_ = 0.0; // 0 means no explicit cap
  opes_biasfactor_ = opes_barrier_ / opes_kbt_;
  opes_bias_prefactor_ = 1.0 - 1.0 / opes_biasfactor_;
  opes_epsilon_ = 0.0;
  opes_user_biasfactor_ = false;
  opes_pace_ = 1000;
  opes_adaptive_sigma_stride_ = 0; // 0 -> 10 * opes_pace, as in OPES_METADCV
  opes_adaptive_counter_ = 0;
  opes_counter_ = 1;
  opes_reweight_count_ = 1;
  opes_last_update_step_ = -1;

  opes_ct_window_ = 10;
  opes_ct_min_stable_windows_ = 3;
  opes_ct_sample_count_ = 0;
  opes_ct_stable_windows_ = 0;
  opes_ct_slope_threshold_ = 1.5e-5; // kcal/mol/fs in real units; 0.015 kcal/mol/ps
  opes_ct_slope_ = 0.0;
  opes_dE_damp_enable_ = true;
  opes_dE_damp_slowf_ = 3.0;
  opes_dE_damp_pos_ = {0.8, 0.8};
  opes_dE_damp_grid_ = 10;
  opes_last_dE_ = 0.0;
  opes_last_damp_bias_ = 0.0;
  opes_kernel_cutoff_ = 4.0;
  opes_kernel_cutoff2_ = opes_kernel_cutoff_ * opes_kernel_cutoff_;
  opes_val_at_cutoff_ = std::exp(-0.5 * opes_kernel_cutoff2_);
  opes_compression_threshold2_ = opes_compression_threshold_ * opes_compression_threshold_;
  opes_compression_threshold_ = 1.0;
  opes_compression_threshold2_ = opes_compression_threshold_ * opes_compression_threshold_;
  opes_recursive_merge_ = true;
  opes_fixed_sigma_ = false;
  opes_no_zed_ = true;

  opes_sigma0_ = {0.0, 0.0};
  opes_av_cv_ = {0.0, 0.0};
  opes_av_M2_ = {0.0, 0.0};

  opes_sum_weights_ = 1.0;
  opes_sum_weights2_ = 1.0;
  opes_reweight_sum_ = 1.0;
  opes_KDEnorm_ = 1.0;
  opes_Zed_ = 1.0;
  opes_neff_ = 1.0;
  opes_rct_ = 0.0;
  opes_prob_ = 0.0;
  opes_raw_bias_ = 0.0;
  opes_boost_bias_ = 0.0;
  opes_hypertime_ = 0.0;
  opes_log_hypertime_ = -std::numeric_limits<double>::infinity();
  opes_hypertime_overflowed_ = false;
  opes_last_hypertime_step_ = -1;

  cvhd_reset_enable_ = true;
  cvhd_event_threshold_ = 1.0;
  cvhd_waittime_ = 10000;
  cvhd_wait_counter_ = 0;
  cvhd_event_count_ = 0;

  cc_connectivity_update_ = false;
  cc_break_bond_threshold_ = 0.0; // 0 means derive from cc_break_ref*(1+cc_break_reset_maxdist)
  cc_form_bond_threshold_ = 0.0;  // 0 means derive from cc_form_ref*(1-cv2_cutoff_ccform)
  cc_event_break_count_ = 0;
  cc_event_form_count_ = 0;

  cc_break_.enabled = true;
  cc_break_.do_form = false;
  cc_break_.kind = CC_PAIR;
  cc_break_.ref = 1.55;
  cc_break_.rmin = 0.0;
  cc_break_.rmax = 1.80;
  cc_break_.power = 6;

  ch_break_.enabled = true;
  ch_break_.do_form = false;
  ch_break_.kind = CH_PAIR;
  ch_break_.ref = 1.05;
  ch_break_.rmin = 0.0;
  ch_break_.rmax = 1.30;
  ch_break_.power = 10;

  cc_form_.enabled = true;
  cc_form_.do_form = true;
  cc_form_.kind = CC_PAIR;
  cc_form_.ref = 2.50;
  cc_form_.rmin = 2.0;
  cc_form_.rmax = 0.0;
  cc_form_.power = 10;
  cc_form_.nl_stride = 0;
}

int FixCvhdGlobalDistortion::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_NEIGHBOR;
  mask |= END_OF_STEP;
  return mask;
}

void FixCvhdGlobalDistortion::init()
{
  if (!atom->tag_enable) {
    error->all(FLERR,
               "fix cvhd/global/distortion requires atom IDs/tags. "
               "Use atom_modify map yes if needed.");
  }

  if (cc_break_.enabled || ch_break_.enabled || cc_form_.enabled) {
    neighbor->add_request(this, backend_neighbor_request_flags());
  }

  open_output();
  initialized_ = true;
}

void FixCvhdGlobalDistortion::init_list(int, NeighList *ptr)
{
  list_ = ptr;
}

void FixCvhdGlobalDistortion::setup(int)
{
  build_tag_map();
  build_carbon_index();

  bool reference_changed = false;
  bool connectivity_changed = false;

  if (cc_break_.enabled) {
    build_reference_pairs(cc_break_);
    reference_changed = true;
    rebuild_cc_connectivity_from_cc_break_pairs();
    connectivity_changed = true;
  }
  if (ch_break_.enabled) {
    build_reference_pairs(ch_break_);
    reference_changed = true;
  }

  if (reference_changed) backend_on_reference_pairs_changed();
  if (connectivity_changed) backend_on_connectivity_changed();

  last_compute_step_ = -1;
  compute_all();

  if (bias_enable_) {
    backend_apply_bias_forces();
  }

  write_output();

  if (!summary_printed_) {
    print_summary();
    summary_printed_ = true;
  }
}

void FixCvhdGlobalDistortion::post_neighbor()
{
  // Any neighbor rebuild/migration may change local index mapping.
  mark_tag_map_dirty();
  // Kokkos subclasses use this hook to invalidate local-index device caches.
}

void FixCvhdGlobalDistortion::post_force(int)
{
  if (!bias_enable_) return;

  compute_all();

  const double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
  backend_apply_bias_forces();
  cvhd_timer_add(cvhd_t_apply_total_, cvhd_n_apply_total_, t0);
}

void FixCvhdGlobalDistortion::end_of_step()
{
  const bigint step = update->ntimestep;

  const bool do_compute_output = (compute_stride_ <= 0 || step % compute_stride_ == 0);
  const bool do_output = do_compute_output && output_stride_ > 0 && step % output_stride_ == 0;

  if (!bias_enable_ && !do_compute_output) return;

  compute_all();

  if (bias_enable_) {
    const double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
    opes_accumulate_hypertime();
    cvhd_timer_add(cvhd_t_hypertime_, cvhd_n_hypertime_, t0);
    // compute_all() stores values_[25] before hypertime is accumulated for
    // this MD interval.  Refresh it here so output is not one timestep behind.
    values_[25] = opes_hypertime_;
  }

  if (do_output) {
    const double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
    write_output();
    cvhd_timer_add(cvhd_t_output_, cvhd_n_output_, t0);
  }

  if (bias_enable_) {
    double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
    opes_update_after_step();
    cvhd_timer_add(cvhd_t_opes_update_, cvhd_n_opes_update_, t0);

    t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
    cvhd_check_event_and_reset();
    cvhd_timer_add(cvhd_t_reset_, cvhd_n_reset_, t0);
  }

  cvhd_timer_maybe_print();
}

double FixCvhdGlobalDistortion::compute_vector(int n)
{
  compute_all();
  if (n < 0 || n >= 33) return 0.0;
  return values_[n];
}

double FixCvhdGlobalDistortion::compute_scalar()
{
  compute_all();
  return values_[8];
}

double FixCvhdGlobalDistortion::cvhd_timer_now() const
{
  return MPI_Wtime();
}

void FixCvhdGlobalDistortion::cvhd_timer_add(double &accum, long long &count, double t0)
{
  if (!cvhd_timer_enable_) return;
  accum += MPI_Wtime() - t0;
  ++count;
}

void FixCvhdGlobalDistortion::cvhd_timer_reset_accumulators()
{
  cvhd_t_compute_all_ = 0.0;
  cvhd_t_tag_map_ = 0.0;
  cvhd_t_ref_conn_ = 0.0;
  cvhd_t_backend_cvs_ = 0.0;
  cvhd_t_bias_ = 0.0;
  cvhd_t_apply_total_ = 0.0;
  cvhd_t_hypertime_ = 0.0;
  cvhd_t_opes_update_ = 0.0;
  cvhd_t_reset_ = 0.0;
  cvhd_t_output_ = 0.0;

  cvhd_t_pair_cache_ = 0.0;
  cvhd_t_kk_reduce_ = 0.0;
  cvhd_t_kk_force_ = 0.0;
  cvhd_t_form_cpu_ = 0.0;
  cvhd_t_fused_cache_ = 0.0;
  cvhd_t_fused_cv_ = 0.0;
  cvhd_t_fused_force_ = 0.0;
  cvhd_t_cpu_break_force_ = 0.0;
  cvhd_t_cpu_form_force_ = 0.0;

  cvhd_n_compute_all_ = 0;
  cvhd_n_tag_map_ = 0;
  cvhd_n_ref_conn_ = 0;
  cvhd_n_backend_cvs_ = 0;
  cvhd_n_bias_ = 0;
  cvhd_n_apply_total_ = 0;
  cvhd_n_hypertime_ = 0;
  cvhd_n_opes_update_ = 0;
  cvhd_n_reset_ = 0;
  cvhd_n_output_ = 0;

  cvhd_n_pair_cache_ = 0;
  cvhd_n_kk_reduce_ = 0;
  cvhd_n_kk_force_ = 0;
  cvhd_n_form_cpu_ = 0;
  cvhd_n_fused_cache_ = 0;
  cvhd_n_fused_cv_ = 0;
  cvhd_n_fused_force_ = 0;
  cvhd_n_cpu_break_force_ = 0;
  cvhd_n_cpu_form_force_ = 0;
}

void FixCvhdGlobalDistortion::cvhd_timer_maybe_print()
{
  if (!cvhd_timer_enable_) return;
  if (cvhd_timer_stride_ <= 0) return;

  const bigint step = update->ntimestep;
  if (step <= 0 || step % cvhd_timer_stride_ != 0) return;
  if (cvhd_timer_last_print_step_ == step) return;
  cvhd_timer_last_print_step_ = step;

  auto avg_us = [](double t, long long n) -> double {
    return (n > 0) ? (1.0e6 * t / static_cast<double>(n)) : 0.0;
  };

  if (comm->me == 0) {
    utils::logmesg(lmp,
                   "CVHD_TIMER step {} avg_us: compute_all={} tag_map={} ref_conn={} backend_cvs={} bias={} apply_force={} hypertime={} opes_update={} reset={} output={}\n",
                   step,
                   avg_us(cvhd_t_compute_all_, cvhd_n_compute_all_),
                   avg_us(cvhd_t_tag_map_, cvhd_n_tag_map_),
                   avg_us(cvhd_t_ref_conn_, cvhd_n_ref_conn_),
                   avg_us(cvhd_t_backend_cvs_, cvhd_n_backend_cvs_),
                   avg_us(cvhd_t_bias_, cvhd_n_bias_),
                   avg_us(cvhd_t_apply_total_, cvhd_n_apply_total_),
                   avg_us(cvhd_t_hypertime_, cvhd_n_hypertime_),
                   avg_us(cvhd_t_opes_update_, cvhd_n_opes_update_),
                   avg_us(cvhd_t_reset_, cvhd_n_reset_),
                   avg_us(cvhd_t_output_, cvhd_n_output_));

    utils::logmesg(lmp,
                   "CVHD_TIMER step {} backend avg_us: pair_cache={} kk_reduce={} kk_force={} form_cpu={} fused_cache={} fused_cv={} fused_force={} cpu_break_force={} cpu_form_force={} counts(cache/reduce/force/form/fcache/fcv/fforce)={}/{}/{}/{}/{}/{}/{}\n",
                   step,
                   avg_us(cvhd_t_pair_cache_, cvhd_n_pair_cache_),
                   avg_us(cvhd_t_kk_reduce_, cvhd_n_kk_reduce_),
                   avg_us(cvhd_t_kk_force_, cvhd_n_kk_force_),
                   avg_us(cvhd_t_form_cpu_, cvhd_n_form_cpu_),
                   avg_us(cvhd_t_fused_cache_, cvhd_n_fused_cache_),
                   avg_us(cvhd_t_fused_cv_, cvhd_n_fused_cv_),
                   avg_us(cvhd_t_fused_force_, cvhd_n_fused_force_),
                   avg_us(cvhd_t_cpu_break_force_, cvhd_n_cpu_break_force_),
                   avg_us(cvhd_t_cpu_form_force_, cvhd_n_cpu_form_force_),
                   cvhd_n_pair_cache_, cvhd_n_kk_reduce_, cvhd_n_kk_force_, cvhd_n_form_cpu_,
                   cvhd_n_fused_cache_, cvhd_n_fused_cv_, cvhd_n_fused_force_);
  }

  if (cvhd_timer_reset_each_print_) {
    cvhd_timer_reset_accumulators();
  }
}

void FixCvhdGlobalDistortion::compute_all()
{
  const bigint step = update->ntimestep;
  if (last_compute_step_ == step) return;

  const double t_all = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;

  double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
  const bool need_tag_map =
    !backend_can_skip_tag_map_each_compute() ||
    tag_map_dirty_ ||
    (cc_break_.enabled && cc_break_.rebuild_ref) ||
    (ch_break_.enabled && ch_break_.rebuild_ref) ||
    cc_connectivity_update_;

  if (need_tag_map) {
    ensure_tag_map_current();
    cvhd_timer_add(cvhd_t_tag_map_, cvhd_n_tag_map_, t0);
  }

  t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;

  bool reference_changed = false;
  bool connectivity_changed = false;

  if (cc_break_.enabled && cc_break_.rebuild_ref) {
    build_reference_pairs(cc_break_);
    reference_changed = true;
    rebuild_cc_connectivity_from_cc_break_pairs();
    connectivity_changed = true;
  }
  if (ch_break_.enabled && ch_break_.rebuild_ref) {
    build_reference_pairs(ch_break_);
    reference_changed = true;
  }

  if (reference_changed) backend_on_reference_pairs_changed();

  if (cc_connectivity_update_) {
    update_cc_connectivity();
    connectivity_changed = true;
  }

  if (connectivity_changed) backend_on_connectivity_changed();
  cvhd_timer_add(cvhd_t_ref_conn_, cvhd_n_ref_conn_, t0);

  double ccbb = 0.0;
  double chbb = 0.0;
  double ccbf = 0.0;
  t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
  backend_compute_raw_cvs(ccbb, chbb, ccbf);
  cvhd_timer_add(cvhd_t_backend_cvs_, cvhd_n_backend_cvs_, t0);

  const double cv1 = cvhdm_two(ccbb, chbb, cv1_cutoff_cc_, cv1_cutoff_ch_, cvhd_power_);
  const double cv2 = cvhdm_one(ccbf, cv2_cutoff_ccform_, cvhd_power_);

  values_[0] = ccbb;
  values_[1] = chbb;
  values_[2] = ccbf;
  values_[3] = cv1;
  values_[4] = cv2;
  values_[5] = static_cast<double>(cc_break_.pairs.size());
  values_[6] = static_cast<double>(ch_break_.pairs.size());
  values_[7] = static_cast<double>(last_cc_form_pair_count_);
  values_[12] = static_cast<double>(cc_break_.pairs.size());
  values_[13] = static_cast<double>(cc_event_break_count_);
  values_[14] = static_cast<double>(cc_event_form_count_);
  values_[15] = static_cast<double>(carbon_tags_.size());

  t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
  compute_bias_derivatives();
  cvhd_timer_add(cvhd_t_bias_, cvhd_n_bias_, t0);

  values_[18] = static_cast<double>(opes_kernels_.size());
  values_[19] = opes_sigma0_[0];
  values_[20] = opes_use_cv2_ ? opes_sigma0_[1] : 0.0;
  values_[21] = opes_rct_;
  values_[22] = opes_neff_;
  values_[23] = opes_prob_;
  values_[24] = opes_raw_bias_;
  values_[25] = opes_hypertime_;
  values_[26] = static_cast<double>(cvhd_event_count_);
  values_[27] = static_cast<double>(cvhd_wait_counter_);
  values_[28] = opes_barrier_;
  values_[29] = opes_ct_slope_;
  values_[30] = static_cast<double>(opes_ct_stable_windows_);
  values_[31] = opes_last_dE_;
  values_[32] = opes_last_damp_bias_;

  last_compute_step_ = step;
  cvhd_timer_add(cvhd_t_compute_all_, cvhd_n_compute_all_, t_all);
}

void FixCvhdGlobalDistortion::backend_sync_host_atoms_for_reference()
{
  // CPU backend already reads host atom arrays.
}

bool FixCvhdGlobalDistortion::backend_can_skip_tag_map_each_compute() const
{
  return false;
}

void FixCvhdGlobalDistortion::backend_sync_host_atoms_for_tag_map()
{
  // CPU backend already reads host atom arrays.
}

void FixCvhdGlobalDistortion::mark_tag_map_dirty()
{
  tag_map_dirty_ = true;
}

void FixCvhdGlobalDistortion::ensure_tag_map_current()
{
  if (!tag_map_dirty_) return;
  backend_sync_host_atoms_for_tag_map();
  build_tag_map();
  tag_map_dirty_ = false;
}

bool FixCvhdGlobalDistortion::backend_collect_reference_pairs(const TermConfig &, std::vector<PairRef> &)
{
  return false;
}

int FixCvhdGlobalDistortion::backend_neighbor_request_flags() const
{
  return NeighConst::REQ_DEFAULT;
}

void FixCvhdGlobalDistortion::backend_on_reference_pairs_changed()
{
  // CPU backend has no mirrored device arrays to update.
  // A future /kk subclass will rebuild/sync device reference-pair views here.
}

void FixCvhdGlobalDistortion::backend_on_connectivity_changed()
{
  // CPU backend has no mirrored device connectivity table to update.
  // A future /kk subclass will rebuild/sync device C-C partner-tag views here.
}

void FixCvhdGlobalDistortion::backend_compute_raw_cvs(double &ccbb, double &chbb, double &ccbf)
{
  ccbb = cc_break_.enabled ? compute_term_value(cc_break_) : 0.0;
  chbb = ch_break_.enabled ? compute_term_value(ch_break_) : 0.0;

  if (cc_form_.enabled) {
    const double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
    ccbf = compute_formation_value_from_neighbor_list();
    cvhd_timer_add(cvhd_t_form_cpu_, cvhd_n_form_cpu_, t0);
  } else {
    ccbf = 0.0;
  }
}

void FixCvhdGlobalDistortion::backend_apply_bias_forces()
{
  double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
  apply_breaking_forces(cc_break_, values_[0], values_[10]);
  apply_breaking_forces(ch_break_, values_[1], values_[11]);
  cvhd_timer_add(cvhd_t_cpu_break_force_, cvhd_n_cpu_break_force_, t0);

  if (opes_use_cv2_ && values_[17] != 0.0 && values_[2] > 0.0) {
    t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
    apply_formation_forces(values_[2], values_[17]);
    cvhd_timer_add(cvhd_t_cpu_form_force_, cvhd_n_cpu_form_force_, t0);
  }
}

void FixCvhdGlobalDistortion::compute_bias_derivatives()
{
  values_[8] = 0.0;   // total boost energy
  values_[9] = 0.0;   // dV/dcv1
  values_[10] = 0.0;  // dV/dccbb
  values_[11] = 0.0;  // dV/dchbb
  values_[16] = 0.0;  // dV/dcv2
  values_[17] = 0.0;  // dV/dccbf

  opes_prob_ = 0.0;
  opes_raw_bias_ = 0.0;
  opes_boost_bias_ = 0.0;

  if (!bias_enable_) return;

  compute_opes_bias_and_derivatives(values_[3], values_[4]);

  double dcv1_dccbb = 0.0;
  double dcv1_dchbb = 0.0;
  cvhdm_two_derivatives(values_[0], values_[1],
                        cv1_cutoff_cc_, cv1_cutoff_ch_, cvhd_power_,
                        dcv1_dccbb, dcv1_dchbb);

  values_[10] = values_[9] * dcv1_dccbb;
  values_[11] = values_[9] * dcv1_dchbb;

  if (opes_use_cv2_) {
    const double dcv2_dccbf = cvhdm_one_derivative(values_[2], cv2_cutoff_ccform_, cvhd_power_);
    values_[17] = values_[16] * dcv2_dccbf;
  } else {
    values_[16] = 0.0;
    values_[17] = 0.0;
  }
}

void FixCvhdGlobalDistortion::initialize_opes_state()
{
  opes_ncv_ = opes_use_cv2_ ? 2 : 1;

  if (opes_adaptive_sigma_stride_ == 0) {
    opes_adaptive_sigma_stride_ = 10 * opes_pace_;
  }

  opes_kbt_ = force->boltz * opes_temperature_;
  opes_update_barrier_dependent_parameters();

  opes_kernel_cutoff2_ = opes_kernel_cutoff_ * opes_kernel_cutoff_;
  opes_val_at_cutoff_ = std::exp(-0.5 * opes_kernel_cutoff2_);
  opes_compression_threshold2_ = opes_compression_threshold_ * opes_compression_threshold_;

  const double w0 = std::pow(opes_epsilon_, opes_bias_prefactor_);
  opes_sum_weights_ = opes_no_zed_ ? 1.0 : w0;
  opes_sum_weights2_ = opes_sum_weights_ * opes_sum_weights_;
  opes_reweight_sum_ = 1.0;
  opes_KDEnorm_ = opes_sum_weights_;
  opes_Zed_ = 1.0;
  opes_neff_ = 1.0;
  opes_rct_ = 0.0;
  opes_counter_ = 1;
  opes_reweight_count_ = 1;
  opes_adaptive_counter_ = 0;
  opes_last_update_step_ = -1;

  // Important for reset:
  // after a CVHD event the OPES adaptive-sigma estimator must restart from
  // the new basin.  Otherwise the old accumulated M2 is divided by a small
  // post-reset counter, producing a huge sigma and sometimes NaN on the first
  // new kernel.
  opes_sigma0_ = {0.0, 0.0};
  opes_av_cv_ = {0.0, 0.0};
  opes_av_M2_ = {0.0, 0.0};

  opes_reset_ct_accumulator();

  opes_hypertime_ = 0.0;
  opes_log_hypertime_ = -std::numeric_limits<double>::infinity();
  opes_hypertime_overflowed_ = false;
  opes_last_hypertime_step_ = 0;

  opes_kernels_.clear();
}

double FixCvhdGlobalDistortion::evaluate_opes_kernel(const OpesKernel &k,
                                                     const std::array<double,2> &cv,
                                                     std::array<double,2> *der) const
{
  double r2 = 0.0;
  double dist[2];

  for (int i = 0; i < opes_ncv_; ++i) {
    dist[i] = (cv[i] - k.center[i]) / k.sigma[i];
    r2 += dist[i] * dist[i];
    if (r2 >= opes_kernel_cutoff2_) return 0.0;
  }

  const double g = std::exp(-0.5 * r2);
  const double val = k.height * (g - opes_val_at_cutoff_);

  if (der) {
    for (int i = 0; i < opes_ncv_; ++i) {
      (*der)[i] += k.height * g * (-(cv[i] - k.center[i]) / (k.sigma[i] * k.sigma[i]));
    }
  }

  return val;
}

void FixCvhdGlobalDistortion::opes_update_barrier_dependent_parameters()
{
  if (!opes_user_biasfactor_ || opes_biasfactor_ <= 0.0) {
    opes_biasfactor_ = opes_barrier_ / opes_kbt_;
  }

  if (opes_biasfactor_ <= 1.0) {
    error->all(FLERR, "OPES biasfactor must be greater than 1. Increase opes_barrier or temperature settings.");
  }

  opes_bias_prefactor_ = 1.0 - 1.0 / opes_biasfactor_;
  opes_epsilon_ = std::exp(-opes_barrier_ / (opes_bias_prefactor_ * opes_kbt_));
}

void FixCvhdGlobalDistortion::opes_reset_ct_accumulator()
{
  opes_reweight_sum_ = 1.0;
  opes_reweight_count_ = 1;
  opes_rct_ = 0.0;
  opes_ct_sample_count_ = 0;
  opes_ct_stable_windows_ = 0;
  opes_ct_slope_ = 0.0;
  opes_ct_values_.clear();
}

double FixCvhdGlobalDistortion::opes_compute_ct_slope() const
{
  const int n = static_cast<int>(opes_ct_values_.size());
  if (n < 2) return 0.0;

  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;

  for (int i = 0; i < n; ++i) {
    // Use sample index instead of physical simulation time:
    // x = 1, 2, 3, ...
    // Each x corresponds to one deposited-kernel c(t) sample.  Therefore the
    // fitted slope is "energy per rct sample" and is independent of timestep
    // and output/deposition time units.
    const double x = static_cast<double>(i + 1);
    const double y = opes_ct_values_[i];
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }

  const double denom = static_cast<double>(n) * sxx - sx * sx;
  if (std::abs(denom) < 1.0e-300) return 0.0;

  return (static_cast<double>(n) * sxy - sx * sy) / denom;
}

void FixCvhdGlobalDistortion::opes_update_barrier_gate()
{
  if (opes_barrier_increment_ <= 0.0) return;
  if (opes_ct_window_ <= 1) return;

  opes_ct_values_.push_back(opes_rct_);
  ++opes_ct_sample_count_;

  if (static_cast<int>(opes_ct_values_.size()) < opes_ct_window_) return;

  opes_ct_slope_ = opes_compute_ct_slope();

  if (std::abs(opes_ct_slope_) <= opes_ct_slope_threshold_) {
    ++opes_ct_stable_windows_;
  } else {
    opes_ct_stable_windows_ = 0;
  }

  opes_ct_values_.clear();

  if (opes_ct_stable_windows_ < opes_ct_min_stable_windows_) return;

  double dE_eff = opes_barrier_increment_;
  opes_last_damp_bias_ = 0.0;

  if (opes_dE_damp_enable_ && opes_dE_damp_slowf_ > 0.0) {
    const double vbias_pos = opes_evaluate_damping_bias();
    opes_last_damp_bias_ = vbias_pos;

    const double exponent = -vbias_pos / (opes_kbt_ * opes_dE_damp_slowf_);
    dE_eff *= std::exp(std::max(-700.0, std::min(700.0, exponent)));
  }

  opes_last_dE_ = dE_eff;

  double new_barrier = opes_barrier_ + dE_eff;
  if (opes_barrier_max_ > 0.0) new_barrier = std::min(new_barrier, opes_barrier_max_);

  if (new_barrier <= opes_barrier_ + 1.0e-14) return;

  opes_barrier_ = new_barrier;
  opes_update_barrier_dependent_parameters();

  // Every time E is increased, c(t) and the slope-gate state are cleared.
  // The OPES kernels are kept, and hypertime is preserved.
  opes_reset_ct_accumulator();

  if (comm->me == 0) {
    utils::logmesg(lmp,
                   "fix cvhd/global/distortion: OPES barrier increased at step {}: E={}, dE_eff={}, damp_bias={}, epsilon={}, gamma={}\n",
                   update->ntimestep, opes_barrier_, opes_last_dE_, opes_last_damp_bias_,
                   opes_epsilon_, opes_biasfactor_);
  }
}

double FixCvhdGlobalDistortion::opes_evaluate_boost_at_cv(const std::array<double,2> &cv) const
{
  double prob_local = 0.0;
  for (const OpesKernel &k : opes_kernels_) {
    prob_local += evaluate_opes_kernel(k, cv, nullptr);
  }

  // Kernel history is replicated on every MPI rank; evaluate locally.
  double prob = prob_local;

  if (opes_KDEnorm_ > 0.0) prob /= opes_KDEnorm_;

  const double denom = prob / opes_Zed_ + opes_epsilon_;
  if (!std::isfinite(denom) || denom <= std::numeric_limits<double>::min()) {
    return 0.0;
  }

  const double raw_bias = opes_kbt_ * opes_bias_prefactor_ * std::log(denom);
  double boost = raw_bias + opes_barrier_;
  if (!std::isfinite(boost)) boost = 0.0;
  return clamp_value(boost, 0.0, opes_barrier_);
}

double FixCvhdGlobalDistortion::opes_evaluate_damping_bias() const
{
  if (!opes_use_cv2_) {
    const std::array<double,2> cv = {opes_dE_damp_pos_[0], 0.0};
    return opes_evaluate_boost_at_cv(cv);
  }

  const int ngrid = std::max(2, opes_dE_damp_grid_);
  double vmax = -std::numeric_limits<double>::infinity();

  // In 2D, follow the original OPES_CVHD damping idea:
  //   line 1: cv1 = eta_s, cv2 in [0,1]
  //   line 2: cv2 = eta_s, cv1 in [0,1]
  // and use the maximum boost bias along the two lines.
  for (int i = 0; i < ngrid; ++i) {
    const double s = (ngrid == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(ngrid - 1);

    const std::array<double,2> cv_line1 = {opes_dE_damp_pos_[0], s};
    const std::array<double,2> cv_line2 = {s, opes_dE_damp_pos_[1]};

    vmax = std::max(vmax, opes_evaluate_boost_at_cv(cv_line1));
    vmax = std::max(vmax, opes_evaluate_boost_at_cv(cv_line2));
  }

  return vmax;
}

void FixCvhdGlobalDistortion::compute_opes_bias_and_derivatives(double cv1, double cv2)
{
  const std::array<double,2> cv = {cv1, cv2};

  std::array<double,2> der_prob = {0.0, 0.0};
  double prob_local = 0.0;

  for (const OpesKernel &k : opes_kernels_) {
    prob_local += evaluate_opes_kernel(k, cv, &der_prob);
  }

  // Kernel history is replicated on every MPI rank in this plugin.
  // Therefore each rank evaluates the complete OPES probability locally.
  // Do not MPI_Allreduce here, otherwise prob and derivatives would be
  // multiplied by the MPI size.
  double prob = prob_local;
  double der0 = der_prob[0];
  double der1 = opes_use_cv2_ ? der_prob[1] : 0.0;

  if (opes_KDEnorm_ > 0.0) {
    prob /= opes_KDEnorm_;
    der0 /= opes_KDEnorm_;
    if (opes_use_cv2_) der1 /= opes_KDEnorm_;
  }

  opes_prob_ = prob;

  const double denom_raw = prob / opes_Zed_ + opes_epsilon_;
  if (!std::isfinite(denom_raw) || denom_raw <= std::numeric_limits<double>::min()) {
    opes_raw_bias_ = -opes_barrier_;
    opes_boost_bias_ = 0.0;
    values_[8] = 0.0;
    values_[9] = 0.0;
    values_[16] = 0.0;
    return;
  }

  const double denom = denom_raw;
  opes_raw_bias_ = opes_kbt_ * opes_bias_prefactor_ * std::log(denom);

  // The shifted boost should be in [0,E] apart from roundoff.  Clamping here
  // prevents underflow of epsilon or empty-support evaluations from producing
  // -inf/NaN while preserving the physically relevant range.
  opes_boost_bias_ = opes_raw_bias_ + opes_barrier_;
  if (!std::isfinite(opes_boost_bias_)) opes_boost_bias_ = 0.0;
  opes_boost_bias_ = clamp_value(opes_boost_bias_, 0.0, opes_barrier_);
  values_[8] = opes_boost_bias_;

  if (prob > 0.0) {
    // Internal convention: values_[9] and values_[16] store dV/dCV.
    // PLUMED's setOutputForce stores -dV/dCV, so the sign here is opposite
    // to the OPES_METAD setOutputForce line.
    const double pref = opes_kbt_ * opes_bias_prefactor_ / denom / opes_Zed_;
    values_[9] = pref * der0;    // dV/dcv1
    values_[16] = opes_use_cv2_ ? pref * der1 : 0.0;   // dV/dcv2

    if (!std::isfinite(values_[9])) values_[9] = 0.0;
    if (!std::isfinite(values_[16])) values_[16] = 0.0;
  } else {
    values_[9] = 0.0;
    values_[16] = 0.0;
  }
}

double FixCvhdGlobalDistortion::opes_clamp_sigma(int, double sigma) const
{
  if (!std::isfinite(sigma)) return OPES_SIGMA_MIN_HARDCODED;
  return std::max(sigma, OPES_SIGMA_MIN_HARDCODED);
}

void FixCvhdGlobalDistortion::opes_update_after_step()
{
  const bigint step = update->ntimestep;
  if (opes_last_update_step_ == step) return;
  opes_last_update_step_ = step;

  const std::array<double,2> cv = {values_[3], values_[4]};

  // Adaptive sigma follows the same Welford-style running estimate used in
  // OPES_METADCV: default ADAPTIVE_SIGMA_STRIDE = 10 * PACE.
  opes_adaptive_counter_++;
  int tau = opes_adaptive_sigma_stride_;
  if (opes_adaptive_counter_ < opes_adaptive_sigma_stride_) tau = opes_adaptive_counter_;

  for (int i = 0; i < opes_ncv_; ++i) {
    const double diff = cv[i] - opes_av_cv_[i];
    opes_av_cv_[i] += diff / static_cast<double>(tau);
    opes_av_M2_[i] += diff * (cv[i] - opes_av_cv_[i]);
  }

  // Do not deposit kernels before the adaptive sigma estimate has been measured.
  if (opes_adaptive_counter_ < opes_adaptive_sigma_stride_ && opes_counter_ == 1) {
    opes_rct_ = opes_kbt_ * std::log(opes_reweight_sum_ / static_cast<double>(opes_reweight_count_));
    return;
  }

  if (opes_pace_ <= 0 || step % opes_pace_ != 0) {
    opes_rct_ = opes_kbt_ * std::log(opes_reweight_sum_ / static_cast<double>(opes_reweight_count_));
    return;
  }

  opes_deposit_kernel(cv);
}

void FixCvhdGlobalDistortion::opes_accumulate_hypertime()
{
  const bigint step = update->ntimestep;
  bigint previous = opes_last_hypertime_step_;
  if (previous < 0) previous = 0;

  if (step <= previous) return;
  const bigint nsteps = step - previous;
  opes_last_hypertime_step_ = step;

  if (opes_kbt_ <= 0.0) return;
  if (update->dt <= 0.0) return;

  // Accumulate hypertime in log space:
  //   t_h <- t_h + nsteps*dt * exp(beta * boost)
  // Using nsteps fixes the historical 0.1 fs offset where the first MD
  // interval was not counted in output.
  const double safe_boost = std::isfinite(values_[8]) ? values_[8] : 0.0;
  const double log_increment = std::log(update->dt * static_cast<double>(nsteps)) +
                               safe_boost / opes_kbt_;

  if (!std::isfinite(opes_log_hypertime_)) {
    opes_log_hypertime_ = log_increment;
  } else {
    const double a = std::max(opes_log_hypertime_, log_increment);
    const double b = std::min(opes_log_hypertime_, log_increment);
    opes_log_hypertime_ = a + std::log1p(std::exp(b - a));
  }

  const double log_max = std::log(std::numeric_limits<double>::max());
  if (opes_log_hypertime_ < log_max - 1.0) {
    opes_hypertime_ = std::exp(opes_log_hypertime_);
    opes_hypertime_overflowed_ = false;
  } else {
    opes_hypertime_ = std::numeric_limits<double>::max();
    opes_hypertime_overflowed_ = true;
  }
}

void FixCvhdGlobalDistortion::cvhd_check_event_and_reset()
{
  if (!cvhd_reset_enable_) return;

  bool is_event_region = (values_[3] >= cvhd_event_threshold_);
  if (opes_use_cv2_) {
    is_event_region = is_event_region || (values_[4] >= cvhd_event_threshold_);
  }

  if (is_event_region) {
    ++cvhd_wait_counter_;
  } else {
    cvhd_wait_counter_ = 0;
  }

  if (cvhd_wait_counter_ >= cvhd_waittime_) {
    cvhd_perform_reset();
  }
}

void FixCvhdGlobalDistortion::cvhd_perform_reset()
{
  ++cvhd_event_count_;
  cvhd_wait_counter_ = 0;

  // A CVHD event defines a new basin/reference state.  Restart the OPES
  // barrier schedule from the configured E0 rather than carrying over the
  // previous basin's increased E.
  opes_barrier_ = opes_barrier_initial_;
  opes_last_dE_ = 0.0;
  opes_last_damp_bias_ = 0.0;
  opes_update_barrier_dependent_parameters();
  opes_reset_history();

  // Rebuild reference states and C-C connectivity from the current structure.
  bool reference_changed = false;
  bool connectivity_changed = false;

  build_tag_map();
  if (cc_break_.enabled) {
    build_reference_pairs(cc_break_);
    reference_changed = true;
    rebuild_cc_connectivity_from_cc_break_pairs();
    connectivity_changed = true;
  }
  if (ch_break_.enabled) {
    build_reference_pairs(ch_break_);
    reference_changed = true;
  }

  if (reference_changed) backend_on_reference_pairs_changed();
  if (connectivity_changed) backend_on_connectivity_changed();

  last_compute_step_ = -1;

  if (comm->me == 0) {
    utils::logmesg(lmp,
                   "fix cvhd/global/distortion: CVHD reset at step {}: events={}, E reset to {}, hypertime(fs)={}\n",
                   update->ntimestep, cvhd_event_count_, opes_barrier_, opes_hypertime_);
  }
}

void FixCvhdGlobalDistortion::opes_deposit_kernel(const std::array<double,2> &cv)
{
  // Follow the OPES_CVHD convention used in the PLUMED prototype:
  // when adaptive E is active, the positive shifted boost enters the kernel
  // weight.  With opes_barrier_increment=0, this reduces to the previous
  // fixed-E behavior.
  const double weight_bias = (opes_barrier_increment_ > 0.0) ? opes_boost_bias_ : opes_raw_bias_;
  const double log_weight_raw = weight_bias / opes_kbt_;
  const double log_weight = std::max(-OPES_LOG_WEIGHT_MAX,
                                     std::min(OPES_LOG_WEIGHT_MAX, log_weight_raw));
  double height = std::exp(log_weight);
  if (!std::isfinite(height) || height <= 0.0) return;

  const double new_sum_weights = opes_sum_weights_ + height;
  const double new_sum_weights2 = opes_sum_weights2_ + height * height;
  const double new_neff =
    std::pow(1.0 + new_sum_weights, 2.0) / (1.0 + new_sum_weights2);

  std::array<double,2> sigma = opes_sigma0_;
  const double factor = opes_biasfactor_;

  if (factor <= 0.0) return;

  if (opes_kernels_.empty()) {
    for (int i = 0; i < opes_ncv_; ++i) {
      double s = std::sqrt(std::max(opes_av_M2_[i], 0.0) /
                           std::max(1, opes_adaptive_counter_) / factor);
      opes_sigma0_[i] = opes_clamp_sigma(i, s);
      sigma[i] = opes_sigma0_[i];
    }
  } else {
    for (int i = 0; i < opes_ncv_; ++i) {
      double s = std::sqrt(std::max(opes_av_M2_[i], 0.0) /
                           std::max(1, opes_adaptive_counter_) / factor);
      sigma[i] = opes_clamp_sigma(i, s);
    }
  }

  if (!opes_fixed_sigma_) {
    const double ndim = static_cast<double>(opes_ncv_);
    const double base = std::max(new_neff * (ndim + 2.0) / 4.0, 1.0e-300);
    const double s_rescaling = std::pow(base, -1.0 / (4.0 + ndim));
    for (int i = 0; i < opes_ncv_; ++i) {
      sigma[i] = opes_clamp_sigma(i, sigma[i] * s_rescaling);
    }
  }

  for (int i = 0; i < opes_ncv_; ++i) {
    if (!std::isfinite(sigma[i]) || sigma[i] <= 0.0) return;
    height *= (opes_sigma0_[i] / sigma[i]);
  }

  if (!std::isfinite(height) || height <= 0.0) return;

  // CVHD-style TS protection: do not deposit kernels whose truncated support
  // reaches cv=1 along any dimension.
  bool deposit = true;
  for (int i = 0; i < opes_ncv_; ++i) {
    if ((1.0 - cv[i]) / sigma[i] <= opes_kernel_cutoff_) {
      deposit = false;
      break;
    }
  }

  if (deposit) {
    opes_counter_ += 1;
    opes_reweight_count_ += 1;
    opes_sum_weights_ += height;
    opes_reweight_sum_ += height;
    opes_sum_weights2_ += height * height;
    opes_KDEnorm_ = opes_sum_weights_;
    opes_neff_ = std::pow(1.0 + opes_sum_weights_, 2.0) / (1.0 + opes_sum_weights2_);
    opes_add_kernel(height, cv, sigma);
  }

  opes_rct_ = opes_kbt_ * std::log(opes_reweight_sum_ / static_cast<double>(opes_reweight_count_));
  if (!std::isfinite(opes_rct_)) opes_rct_ = 0.0;
  opes_update_barrier_gate();
}

void FixCvhdGlobalDistortion::opes_merge_kernel_into(OpesKernel &taker, const OpesKernel &giver) const
{
  const double h1 = taker.height;
  const double h2 = giver.height;
  const double h = h1 + h2;

  if (h <= 0.0) return;

  std::array<double,2> new_center = taker.center;
  std::array<double,2> new_sigma = taker.sigma;

  for (int i = 0; i < opes_ncv_; ++i) {
    new_center[i] = (h1 * taker.center[i] + h2 * giver.center[i]) / h;
  }

  for (int i = 0; i < opes_ncv_; ++i) {
    const double dc1 = taker.center[i] - new_center[i];
    const double dc2 = giver.center[i] - new_center[i];

    const double var =
      (h1 * (taker.sigma[i] * taker.sigma[i] + dc1 * dc1) +
       h2 * (giver.sigma[i] * giver.sigma[i] + dc2 * dc2)) / h;

    new_sigma[i] = opes_clamp_sigma(i, std::sqrt(std::max(var, 0.0)));
  }

  taker.height = h;
  taker.center = new_center;
  taker.sigma = new_sigma;
}

int FixCvhdGlobalDistortion::opes_get_mergeable_kernel(const std::array<double,2> &center,
                                                       int exclude) const
{
  if (opes_compression_threshold_ <= 0.0) return -1;

  int min_k = -1;
  double min_norm2 = opes_compression_threshold2_;

  for (int k = 0; k < static_cast<int>(opes_kernels_.size()); ++k) {
    if (k == exclude) continue;

    double norm2 = 0.0;
    for (int i = 0; i < opes_ncv_; ++i) {
      const double dist = (center[i] - opes_kernels_[k].center[i]) / opes_kernels_[k].sigma[i];
      norm2 += dist * dist;
      if (norm2 >= min_norm2) break;
    }

    if (norm2 < min_norm2) {
      min_norm2 = norm2;
      min_k = k;
    }
  }

  return min_k;
}

void FixCvhdGlobalDistortion::opes_add_kernel(double height,
                                              const std::array<double,2> &center,
                                              const std::array<double,2> &sigma)
{
  OpesKernel incoming;
  incoming.height = height;
  incoming.center = center;
  incoming.sigma = sigma;

  if (opes_compression_threshold_ <= 0.0 || opes_kernels_.empty()) {
    opes_kernels_.push_back(incoming);
    return;
  }

  int taker = opes_get_mergeable_kernel(incoming.center, -1);
  if (taker < 0) {
    opes_kernels_.push_back(incoming);
    return;
  }

  opes_merge_kernel_into(opes_kernels_[taker], incoming);

  if (!opes_recursive_merge_) return;

  // Recursive compression: after a merge, the enlarged kernel may become
  // mergeable with another nearby kernel.  Keep merging until no match remains.
  while (true) {
    const int giver = opes_get_mergeable_kernel(opes_kernels_[taker].center, taker);
    if (giver < 0) break;

    opes_merge_kernel_into(opes_kernels_[taker], opes_kernels_[giver]);
    opes_kernels_.erase(opes_kernels_.begin() + giver);

    if (giver < taker) --taker;
  }
}

void FixCvhdGlobalDistortion::opes_reset_history()
{
  const double saved_hypertime = opes_hypertime_;
  const double saved_log_hypertime = opes_log_hypertime_;
  const bool saved_overflow = opes_hypertime_overflowed_;
  const bigint saved_last_hypertime_step = opes_last_hypertime_step_;

  initialize_opes_state();

  opes_hypertime_ = saved_hypertime;
  opes_log_hypertime_ = saved_log_hypertime;
  opes_hypertime_overflowed_ = saved_overflow;
  opes_last_hypertime_step_ = saved_last_hypertime_step;
}

void FixCvhdGlobalDistortion::apply_breaking_forces(const TermConfig &term, double raw_value, double dV_draw)
{
  if (!term.enabled) return;
  if (term.do_form) return;
  if (raw_value <= 0.0) return;
  if (dV_draw == 0.0) return;

  // raw_value = pairsum^(1/p), so pairsum^(1/p - 1) = raw_value^(1-p).
  const double term_prefactor = std::pow(raw_value, 1.0 - static_cast<double>(term.power));

  for (const PairRef &p : term.pairs) {
    const auto ii = tag_to_local_.find(p.itag);
    const auto jj = tag_to_local_.find(p.jtag);
    if (ii == tag_to_local_.end() || jj == tag_to_local_.end()) continue;

    // Owned-centered force application:
    // - if both atoms are owned here, this rank applies both forces;
    // - if one atom is owned and the other is ghost, this rank applies only
    //   the owned atom's force;
    // - no ghost force is modified, so no reverse communication is needed.
    if (!is_owned_local_index(ii->second) && !is_owned_local_index(jj->second)) continue;

    apply_pair_breaking_force(term, ii->second, jj->second, p.ref, term_prefactor, dV_draw);
  }
}

void FixCvhdGlobalDistortion::apply_pair_breaking_force(const TermConfig &term, int i, int j, double ref,
                                                        double term_prefactor, double dV_draw)
{
  double **x = atom->x;
  double **f = atom->f;

  double dx = x[j][0] - x[i][0];
  double dy = x[j][1] - x[i][1];
  double dz = x[j][2] - x[i][2];

  if (domain->periodicity[0] || domain->periodicity[1] || domain->periodicity[2]) {
    domain->minimum_image(FLERR,dx,dy,dz);
  }

  const double rsq = dx*dx + dy*dy + dz*dz;
  if (rsq <= 0.0) return;

  const double r = std::sqrt(rsq);

  // Breaking term has no contribution below reference distance.
  if (r < ref) return;

  const double stretch = (r - ref) / term.ref;
  if (stretch == 0.0) return;

  const double pair_deriv_pref = term_prefactor * int_pow(stretch, term.power - 1) / term.ref / r;

  // d(raw)/dx_i = -g, d(raw)/dx_j = +g, where g = pair_deriv_pref * rij.
  // F = -dV/draw * d(raw)/dx.
  const double fx = dV_draw * pair_deriv_pref * dx;
  const double fy = dV_draw * pair_deriv_pref * dy;
  const double fz = dV_draw * pair_deriv_pref * dz;

  if (is_owned_local_index(i)) {
    f[i][0] += fx;
    f[i][1] += fy;
    f[i][2] += fz;
  }

  if (is_owned_local_index(j)) {
    f[j][0] -= fx;
    f[j][1] -= fy;
    f[j][2] -= fz;
  }
}

void FixCvhdGlobalDistortion::apply_formation_forces(double raw_value, double dV_draw)
{
  if (!cc_form_.enabled) return;
  if (raw_value <= 0.0) return;
  if (dV_draw == 0.0) return;

  std::vector<PairRef> formation_pairs;
  build_current_formation_force_pairs(formation_pairs);
  if (formation_pairs.empty()) return;

  const double term_prefactor = std::pow(raw_value, 1.0 - static_cast<double>(cc_form_.power));

  for (const PairRef &p : formation_pairs) {
    const auto ii = tag_to_local_.find(p.itag);
    const auto jj = tag_to_local_.find(p.jtag);
    if (ii == tag_to_local_.end() || jj == tag_to_local_.end()) continue;

    if (!is_owned_local_index(ii->second) && !is_owned_local_index(jj->second)) continue;

    apply_pair_formation_force(ii->second, jj->second, p.ref, term_prefactor, dV_draw);
  }
}

void FixCvhdGlobalDistortion::apply_pair_formation_force(int i, int j, double ref,
                                                         double term_prefactor, double dV_draw)
{
  double **x = atom->x;
  double **f = atom->f;

  double dx = x[j][0] - x[i][0];
  double dy = x[j][1] - x[i][1];
  double dz = x[j][2] - x[i][2];

  if (domain->periodicity[0] || domain->periodicity[1] || domain->periodicity[2]) {
    domain->minimum_image(FLERR,dx,dy,dz);
  }

  const double rsq = dx*dx + dy*dy + dz*dz;
  if (rsq <= 0.0) return;

  const double r = std::sqrt(rsq);

  // Formation term has no contribution above the formation reference distance.
  if (r > ref) return;

  const double stretch = (r - ref) / cc_form_.ref;
  if (stretch == 0.0) return;

  const double pair_deriv_pref = term_prefactor * int_pow(stretch, cc_form_.power - 1) / cc_form_.ref / r;

  // Same chain-rule expression as the breaking branch.  For formation,
  // stretch is negative and power-1 is odd, so the force sign is naturally
  // determined by dV/dccbf and stretch^(p-1).
  const double fx = dV_draw * pair_deriv_pref * dx;
  const double fy = dV_draw * pair_deriv_pref * dy;
  const double fz = dV_draw * pair_deriv_pref * dz;

  if (is_owned_local_index(i)) {
    f[i][0] += fx;
    f[i][1] += fy;
    f[i][2] += fz;
  }

  if (is_owned_local_index(j)) {
    f[j][0] -= fx;
    f[j][1] -= fy;
    f[j][2] -= fz;
  }
}

void FixCvhdGlobalDistortion::maybe_handle_reset(TermConfig &term, double value, bigint step)
{
  if (!term.reset_ref) return;

  if (value < term.reset_maxdist) term.reset_wait = step;

  if (step - term.reset_wait >= term.reset_time) {
    term.reset_wait = step;
    term.rebuild_ref = true;
  }
}

void FixCvhdGlobalDistortion::build_tag_map()
{
  tag_to_local_.clear();

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  tag_to_local_.reserve(static_cast<std::size_t>(nall) * 2u);

  tagint *tag = atom->tag;

  // Important: prefer owned atoms over ghost atoms.
  //
  // In periodic systems, especially even for a single MPI rank, the ghost list
  // can contain periodic images of atoms that are also owned locally.  If the
  // ghost entry overwrites the owned entry in this tag->local map, subsequent
  // owned-centered force application may miss one side of an internal pair,
  // leading to a non-zero net bias force.  Therefore owned atoms are inserted
  // first and ghost atoms are inserted only when that tag is not already present.
  for (int i = 0; i < nlocal; ++i) {
    tag_to_local_[tag[i]] = i;
  }

  for (int i = nlocal; i < nall; ++i) {
    if (tag_to_local_.find(tag[i]) == tag_to_local_.end()) {
      tag_to_local_[tag[i]] = i;
    }
  }
}

void FixCvhdGlobalDistortion::build_carbon_index()
{
  tagint *tag = atom->tag;
  int *type = atom->type;

  std::vector<long long> local_tags;
  local_tags.reserve(static_cast<std::size_t>(atom->nlocal));

  for (int i = 0; i < atom->nlocal; ++i) {
    if (!in_fix_group(i)) continue;
    if (type[i] != carbon_type_) continue;
    local_tags.push_back(static_cast<long long>(tag[i]));
  }

  const int nprocs = comm->nprocs;
  const int local_n = static_cast<int>(local_tags.size());

  std::vector<int> counts(nprocs,0), displs(nprocs,0);
  MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

  int total_n = 0;
  for (int p = 0; p < nprocs; ++p) {
    displs[p] = total_n;
    total_n += counts[p];
  }

  std::vector<long long> all_tags(total_n);

  MPI_Allgatherv(local_tags.data(), local_n, MPI_LONG_LONG,
                 all_tags.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                 world);

  std::sort(all_tags.begin(), all_tags.end());
  all_tags.erase(std::unique(all_tags.begin(), all_tags.end()), all_tags.end());

  carbon_tags_.clear();
  carbon_tags_.reserve(all_tags.size());
  carbon_tag_to_index_.clear();
  carbon_tag_to_index_.reserve(all_tags.size() * 2u);

  for (std::size_t i = 0; i < all_tags.size(); ++i) {
    const tagint t = static_cast<tagint>(all_tags[i]);
    carbon_tags_.push_back(t);
    carbon_tag_to_index_[t] = static_cast<int>(i);
  }

  cc_adj_.assign(carbon_tags_.size(), std::vector<int>());
}

void FixCvhdGlobalDistortion::rebuild_cc_connectivity_from_cc_break_pairs()
{
  if (carbon_tags_.empty()) build_carbon_index();

  for (auto &v : cc_adj_) v.clear();

  for (const PairRef &p : cc_break_.pairs) {
    set_cc_connected(p.itag, p.jtag, true);
  }
}

void FixCvhdGlobalDistortion::rebuild_cc_break_pairs_from_connectivity()
{
  cc_break_.pairs.clear();

  for (std::size_t ci = 0; ci < cc_adj_.size(); ++ci) {
    for (int cj : cc_adj_[ci]) {
      if (static_cast<std::size_t>(cj) <= ci) continue;

      PairRef p;
      p.itag = carbon_tags_[ci];
      p.jtag = carbon_tags_[cj];
      p.ref = cc_break_.ref;
      cc_break_.pairs.push_back(p);
    }
  }
}

bool FixCvhdGlobalDistortion::are_cc_connected(tagint itag, tagint jtag) const
{
  const auto ii = carbon_tag_to_index_.find(itag);
  const auto jj = carbon_tag_to_index_.find(jtag);
  if (ii == carbon_tag_to_index_.end() || jj == carbon_tag_to_index_.end()) return false;

  const int ci = ii->second;
  const int cj = jj->second;
  const std::vector<int> &adj = cc_adj_[ci];

  // C valence is small in hydrocarbons, so a linear scan is faster and more
  // cache-friendly than hashing in the inner neighbor-list loop.
  for (int x : adj) {
    if (x == cj) return true;
  }
  return false;
}

void FixCvhdGlobalDistortion::set_cc_connected(tagint itag, tagint jtag, bool connected)
{
  if (itag == jtag) return;

  const auto ii = carbon_tag_to_index_.find(itag);
  const auto jj = carbon_tag_to_index_.find(jtag);
  if (ii == carbon_tag_to_index_.end() || jj == carbon_tag_to_index_.end()) return;

  const int ci = ii->second;
  const int cj = jj->second;

  auto add_one = [](std::vector<int> &v, int x) {
    for (int y : v) {
      if (y == x) return;
    }
    v.push_back(x);
  };

  auto remove_one = [](std::vector<int> &v, int x) {
    v.erase(std::remove(v.begin(), v.end(), x), v.end());
  };

  if (connected) {
    add_one(cc_adj_[ci], cj);
    add_one(cc_adj_[cj], ci);
  } else {
    remove_one(cc_adj_[ci], cj);
    remove_one(cc_adj_[cj], ci);
  }
}

void FixCvhdGlobalDistortion::gather_unique_event_pairs(const std::vector<PairRef> &local_pairs,
                                                        std::vector<PairRef> &global_pairs) const
{
  const int nprocs = comm->nprocs;
  const int local_longs = static_cast<int>(2 * local_pairs.size());

  std::vector<long long> sendbuf(local_longs);
  for (std::size_t i = 0; i < local_pairs.size(); ++i) {
    tagint a = local_pairs[i].itag;
    tagint b = local_pairs[i].jtag;
    if (b < a) std::swap(a,b);
    sendbuf[2*i  ] = static_cast<long long>(a);
    sendbuf[2*i+1] = static_cast<long long>(b);
  }

  std::vector<int> counts(nprocs,0), displs(nprocs,0);
  MPI_Allgather(&local_longs, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

  int total_longs = 0;
  for (int p = 0; p < nprocs; ++p) {
    displs[p] = total_longs;
    total_longs += counts[p];
  }

  std::vector<long long> recvbuf(total_longs);

  MPI_Allgatherv(sendbuf.data(), local_longs, MPI_LONG_LONG,
                 recvbuf.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                 world);

  std::vector<std::pair<long long,long long>> keys;
  keys.reserve(static_cast<std::size_t>(total_longs/2));

  for (int k = 0; k + 1 < total_longs; k += 2) {
    long long a = recvbuf[k];
    long long b = recvbuf[k+1];
    if (b < a) std::swap(a,b);
    keys.emplace_back(a,b);
  }

  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

  global_pairs.clear();
  global_pairs.reserve(keys.size());

  for (const auto &kv : keys) {
    PairRef p;
    p.itag = static_cast<tagint>(kv.first);
    p.jtag = static_cast<tagint>(kv.second);
    p.ref = cc_break_.ref;
    global_pairs.push_back(p);
  }
}

void FixCvhdGlobalDistortion::gather_unique_pairs_with_ref(const std::vector<PairRef> &local_pairs,
                                                            std::vector<PairRef> &global_pairs,
                                                            double ref) const
{
  const int nprocs = comm->nprocs;
  const int local_longs = static_cast<int>(2 * local_pairs.size());

  std::vector<long long> sendbuf(local_longs);
  for (std::size_t i = 0; i < local_pairs.size(); ++i) {
    tagint a = local_pairs[i].itag;
    tagint b = local_pairs[i].jtag;
    if (b < a) std::swap(a,b);
    sendbuf[2*i  ] = static_cast<long long>(a);
    sendbuf[2*i+1] = static_cast<long long>(b);
  }

  std::vector<int> counts(nprocs,0), displs(nprocs,0);
  MPI_Allgather(&local_longs, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

  int total_longs = 0;
  for (int p = 0; p < nprocs; ++p) {
    displs[p] = total_longs;
    total_longs += counts[p];
  }

  std::vector<long long> recvbuf(total_longs);

  MPI_Allgatherv(sendbuf.data(), local_longs, MPI_LONG_LONG,
                 recvbuf.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                 world);

  std::vector<std::pair<long long,long long>> keys;
  keys.reserve(static_cast<std::size_t>(total_longs/2));

  for (int k = 0; k + 1 < total_longs; k += 2) {
    long long a = recvbuf[k];
    long long b = recvbuf[k+1];
    if (b < a) std::swap(a,b);
    keys.emplace_back(a,b);
  }

  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

  global_pairs.clear();
  global_pairs.reserve(keys.size());

  for (const auto &kv : keys) {
    PairRef p;
    p.itag = static_cast<tagint>(kv.first);
    p.jtag = static_cast<tagint>(kv.second);
    p.ref = ref;
    global_pairs.push_back(p);
  }
}

void FixCvhdGlobalDistortion::build_current_formation_force_pairs(std::vector<PairRef> &global_pairs)
{
  global_pairs.clear();

  if (!list_) {
    error->all(FLERR,
               "fix cvhd/global/distortion needs a LAMMPS neighbor list for C-C formation forces, "
               "but init_list() has not been called");
  }

  double **x = atom->x;
  int *type = atom->type;
  tagint *tag = atom->tag;

  const int inum = list_->inum;
  int *ilist = list_->ilist;
  int *numneigh = list_->numneigh;
  int **firstneigh = list_->firstneigh;

  std::vector<PairRef> local_pairs;
  local_pairs.reserve(atom->nlocal);
  local_pairs.reserve(1024);

  const double cand_rmax_sq = cc_form_candidate_rmax_ * cc_form_candidate_rmax_;

  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    if (!in_fix_group(i) || type[i] != carbon_type_) continue;

    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!in_fix_group(j) || type[j] != carbon_type_) continue;

      const tagint itag = tag[i];
      const tagint jtag = tag[j];
      if (are_cc_connected(itag,jtag)) continue;

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];
      if (domain->periodicity[0] || domain->periodicity[1] || domain->periodicity[2]) {
        domain->minimum_image(FLERR,dx,dy,dz);
      }

      const double rsq = dx*dx + dy*dy + dz*dz;
      if (cc_form_candidate_rmax_ > 0.0 && rsq > cand_rmax_sq) continue;

      // Only pairs with r <= cc_form.ref have non-zero ccbf derivative.
      if (rsq > cc_form_.ref * cc_form_.ref) continue;

      tagint a = itag;
      tagint b = jtag;
      if (b < a) std::swap(a,b);

      PairRef p;
      p.itag = a;
      p.jtag = b;
      p.ref = cc_form_.ref;
      local_pairs.push_back(p);
    }
  }

  gather_unique_pairs_with_ref(local_pairs, global_pairs, cc_form_.ref);
}

void FixCvhdGlobalDistortion::update_cc_connectivity()
{
  if (!cc_connectivity_update_) return;
  if (!cc_break_.enabled && !cc_form_.enabled) return;

  const double break_threshold =
      (cc_break_bond_threshold_ > 0.0)
      ? cc_break_bond_threshold_
      : cc_break_.ref * (1.0 + cc_break_.reset_maxdist);

  const double form_threshold =
      (cc_form_bond_threshold_ > 0.0)
      ? cc_form_bond_threshold_
      : cc_form_.ref * (1.0 - cv2_cutoff_ccform_);

  std::vector<PairRef> local_breaks;
  std::vector<PairRef> local_forms;

  // Break existing C-C bonds.  Each connected pair is tested by the rank that
  // currently owns itag.  If jtag is no longer present as a ghost on that rank,
  // the pair is safely considered broken because it exceeded the communication
  // range relevant for bonded connectivity tracking.
  for (const PairRef &p : cc_break_.pairs) {
    const auto ii = tag_to_local_.find(p.itag);
    if (ii == tag_to_local_.end()) continue;
    if (!is_owned_local_index(ii->second)) continue;

    bool broken = false;
    const auto jj = tag_to_local_.find(p.jtag);

    if (jj == tag_to_local_.end()) {
      broken = true;
    } else {
      const double r = pair_distance(ii->second, jj->second);
      if (r > break_threshold) broken = true;
    }

    if (broken) local_breaks.push_back(p);
  }

  // Form new C-C bonds.  The upper candidate bound is the LAMMPS neighbor list.
  // The reaction/event threshold is form_threshold.  Existing connected pairs
  // are excluded through the current adjacency table, not through a runtime
  // lower distance cutoff.
  if (cc_form_.enabled && list_) {
    double **x = atom->x;
    int *type = atom->type;
    tagint *tag = atom->tag;

    const int inum = list_->inum;
    int *ilist = list_->ilist;
    int *numneigh = list_->numneigh;
    int **firstneigh = list_->firstneigh;

    for (int ii = 0; ii < inum; ++ii) {
      const int i = ilist[ii];
      if (!in_fix_group(i) || type[i] != carbon_type_) continue;

      int *jlist = firstneigh[i];
      const int jnum = numneigh[i];

      for (int jj = 0; jj < jnum; ++jj) {
        const int j = jlist[jj] & NEIGHMASK;
        if (!in_fix_group(j) || type[j] != carbon_type_) continue;

        const tagint itag = tag[i];
        const tagint jtag = tag[j];

        if (are_cc_connected(itag,jtag)) continue;

        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];
        double dz = x[j][2] - x[i][2];
        if (domain->periodicity[0] || domain->periodicity[1] || domain->periodicity[2]) {
          domain->minimum_image(FLERR,dx,dy,dz);
        }

        const double rsq = dx*dx + dy*dy + dz*dz;
        if (cc_form_candidate_rmax_ > 0.0 &&
            rsq > cc_form_candidate_rmax_ * cc_form_candidate_rmax_) continue;

        const double r = std::sqrt(rsq);
        if (r > form_threshold) continue;

        tagint a = itag;
        tagint b = jtag;
        if (b < a) std::swap(a,b);

        PairRef p;
        p.itag = a;
        p.jtag = b;
        p.ref = cc_break_.ref;
        local_forms.push_back(p);
      }
    }
  }

  std::vector<PairRef> global_breaks;
  std::vector<PairRef> global_forms;

  gather_unique_event_pairs(local_breaks, global_breaks);
  gather_unique_event_pairs(local_forms, global_forms);

  if (global_breaks.empty() && global_forms.empty()) return;

  // Apply breaks first, then formations.  This allows a pair that has separated
  // and later re-formed to be treated consistently.
  for (const PairRef &p : global_breaks) {
    set_cc_connected(p.itag, p.jtag, false);
  }
  for (const PairRef &p : global_forms) {
    set_cc_connected(p.itag, p.jtag, true);
  }

  cc_event_break_count_ += static_cast<long long>(global_breaks.size());
  cc_event_form_count_ += static_cast<long long>(global_forms.size());

  rebuild_cc_break_pairs_from_connectivity();

  if (comm->me == 0 && (!global_breaks.empty() || !global_forms.empty())) {
    utils::logmesg(lmp,
                   "fix cvhd/global/distortion: connectivity update at step {}: {} C-C breaks, {} C-C formations, {} connected C-C bonds\n",
                   update->ntimestep, global_breaks.size(), global_forms.size(), cc_break_.pairs.size());
  }
}

void FixCvhdGlobalDistortion::build_reference_pairs(TermConfig &term)
{
  // First allow the backend to collect candidate reference pairs.  The Kokkos
  // backend uses the device neighbor list.  CPU fallback uses the geometric host
  // scan below, independent of list_->firstneigh storage type.
  std::vector<PairRef> local_pairs;
  local_pairs.reserve(1024);

  const bool collected_by_backend = backend_collect_reference_pairs(term, local_pairs);

  if (!collected_by_backend) {
    backend_sync_host_atoms_for_reference();

    double **x = atom->x;
    int *type = atom->type;
    tagint *tag = atom->tag;

    const int nlocal = atom->nlocal;
    const int nall = atom->nlocal + atom->nghost;

    for (int i = 0; i < nlocal; ++i) {
      if (!in_fix_group(i)) continue;

      for (int j = 0; j < nall; ++j) {
        if (j == i) continue;
        if (!in_fix_group(j)) continue;

        bool type_ok = false;
        if (term.kind == CC_PAIR) {
          type_ok = (type[i] == carbon_type_) && (type[j] == carbon_type_);
        } else {
          type_ok = ((type[i] == carbon_type_) && (type[j] == hydrogen_type_)) ||
                    ((type[i] == hydrogen_type_) && (type[j] == carbon_type_));
        }
        if (!type_ok) continue;

        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];
        double dz = x[j][2] - x[i][2];

        if (domain->periodicity[0] || domain->periodicity[1] || domain->periodicity[2]) {
          domain->minimum_image(FLERR,dx,dy,dz);
        }

        const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < term.rmin || r > term.rmax) continue;

        tagint itag = tag[i];
        tagint jtag = tag[j];
        if (itag == jtag) continue;
        if (jtag < itag) std::swap(itag,jtag);

        PairRef p;
        p.itag = itag;
        p.jtag = jtag;
        p.ref = term.ref;
        local_pairs.push_back(p);
      }
    }
  }

  gather_unique_reference_pairs(term, local_pairs);

  term.rebuild_ref = false;
  term.reset_wait = update->ntimestep;

  if (comm->me == 0) {
    utils::logmesg(lmp, "fix cvhd/global/distortion: rebuilt global reference list with {} pairs\n",
                   term.pairs.size());
  }
}

void FixCvhdGlobalDistortion::gather_unique_reference_pairs(TermConfig &term, const std::vector<PairRef> &local_pairs)
{
  const int nprocs = comm->nprocs;

  const int local_longs = static_cast<int>(2 * local_pairs.size());
  std::vector<long long> sendbuf(local_longs);

  for (std::size_t i = 0; i < local_pairs.size(); ++i) {
    sendbuf[2*i  ] = static_cast<long long>(local_pairs[i].itag);
    sendbuf[2*i+1] = static_cast<long long>(local_pairs[i].jtag);
  }

  std::vector<int> counts(nprocs,0), displs(nprocs,0);
  MPI_Allgather(&local_longs, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

  int total_longs = 0;
  for (int p = 0; p < nprocs; ++p) {
    displs[p] = total_longs;
    total_longs += counts[p];
  }

  std::vector<long long> recvbuf(total_longs);

  MPI_Allgatherv(sendbuf.data(), local_longs, MPI_LONG_LONG,
                 recvbuf.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                 world);

  std::vector<std::pair<long long,long long>> keys;
  keys.reserve(static_cast<std::size_t>(total_longs/2));

  for (int k = 0; k + 1 < total_longs; k += 2) {
    long long a = recvbuf[k];
    long long b = recvbuf[k+1];
    if (b < a) std::swap(a,b);
    keys.emplace_back(a,b);
  }

  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

  term.pairs.clear();
  term.pairs.reserve(keys.size());

  for (const auto &kv : keys) {
    PairRef p;
    p.itag = static_cast<tagint>(kv.first);
    p.jtag = static_cast<tagint>(kv.second);
    p.ref = term.ref;
    term.pairs.push_back(p);
  }
}

double FixCvhdGlobalDistortion::compute_term_value(const TermConfig &term)
{
  double local_pairsum = 0.0;

  for (const PairRef &p : term.pairs) {
    const auto ii = tag_to_local_.find(p.itag);
    if (ii == tag_to_local_.end()) continue;

    // Each reference pair is evaluated by the rank currently owning itag.
    // jtag may be either owned or ghost on that rank.
    if (!is_owned_local_index(ii->second)) continue;

    const auto jj = tag_to_local_.find(p.jtag);
    if (jj == tag_to_local_.end()) continue;

    local_pairsum += compute_pair_contribution(term, ii->second, jj->second, p.ref);
  }

  double global_pairsum = 0.0;
  MPI_Allreduce(&local_pairsum, &global_pairsum, 1, MPI_DOUBLE, MPI_SUM, world);

  if (global_pairsum <= 0.0) return 0.0;
  return std::pow(global_pairsum, 1.0/static_cast<double>(term.power));
}

double FixCvhdGlobalDistortion::compute_pair_contribution(const TermConfig &term, int i, int j, double ref) const
{
  const double r = pair_distance(i,j);
  return compute_pair_contribution_from_r(term, r, ref);
}

double FixCvhdGlobalDistortion::compute_pair_contribution_from_r(const TermConfig &term, double r, double ref) const
{
  if (term.do_form) {
    if (r > ref) return 0.0;
  } else {
    if (r < ref) return 0.0;
  }

  const double stretch = (r - ref) / term.ref;
  return int_pow(stretch, term.power);
}

double FixCvhdGlobalDistortion::compute_formation_value_from_neighbor_list()
{
  if (!list_) {
    error->all(FLERR,
               "fix cvhd/global/distortion needs a LAMMPS neighbor list for C-C formation, "
               "but init_list() has not been called");
  }

  double **x = atom->x;
  int *type = atom->type;

  const int inum = list_->inum;
  int *ilist = list_->ilist;
  int *numneigh = list_->numneigh;
  int **firstneigh = list_->firstneigh;

  double local_pairsum = 0.0;
  long long local_pair_count = 0;

  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    if (!in_fix_group(i) || type[i] != carbon_type_) continue;

    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!in_fix_group(j) || type[j] != carbon_type_) continue;

      const tagint itag = atom->tag[i];
      const tagint jtag = atom->tag[j];
      if (are_cc_connected(itag,jtag)) continue;

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];
      if (domain->periodicity[0] || domain->periodicity[1] || domain->periodicity[2]) {
        domain->minimum_image(FLERR,dx,dy,dz);
      }
      const double rsq = dx*dx + dy*dy + dz*dz;
      if (cc_form_candidate_rmax_ > 0.0 &&
          rsq > cc_form_candidate_rmax_ * cc_form_candidate_rmax_) continue;

      const double r = std::sqrt(rsq);

      ++local_pair_count;
      local_pairsum += compute_pair_contribution_from_r(cc_form_, r, cc_form_.ref);
    }
  }

  double global_pairsum = 0.0;
  long long global_pair_count = 0;

  MPI_Allreduce(&local_pairsum, &global_pairsum, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&local_pair_count, &global_pair_count, 1, MPI_LONG_LONG, MPI_SUM, world);

  last_cc_form_pair_count_ = static_cast<std::size_t>(global_pair_count);

  if (global_pairsum <= 0.0) return 0.0;
  return std::pow(global_pairsum, 1.0/static_cast<double>(cc_form_.power));
}

double FixCvhdGlobalDistortion::pair_distance(int i, int j) const
{
  double **x = atom->x;
  double dx = x[j][0] - x[i][0];
  double dy = x[j][1] - x[i][1];
  double dz = x[j][2] - x[i][2];

  if (domain->periodicity[0] || domain->periodicity[1] || domain->periodicity[2]) {
    domain->minimum_image(FLERR,dx,dy,dz);
  }

  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

double FixCvhdGlobalDistortion::int_pow(double x, int p) const
{
  double y = 1.0;
  while (p > 0) {
    if (p & 1) y *= x;
    x *= x;
    p >>= 1;
  }
  return y;
}

double FixCvhdGlobalDistortion::cvhdm_two(double a, double b, double ca, double cb, double p) const
{
  const double pi = 3.141592653589793238462643383279502884;
  if (ca <= 0.0 || cb <= 0.0 || p <= 0.0) return 0.0;

  const double na = a / ca;
  const double nb = b / cb;
  const double combined = std::pow(na,p) + std::pow(nb,p);

  if (combined <= 0.0) return 0.0;

  const double projected = std::pow(combined, 2.0/p);
  if (projected >= 1.0) return 1.0;
  if (projected <= 0.0) return 0.0;

  return 0.5 * (1.0 - std::cos(pi * projected));
}

double FixCvhdGlobalDistortion::cvhdm_one(double a, double ca, double p) const
{
  const double pi = 3.141592653589793238462643383279502884;
  if (ca <= 0.0 || p <= 0.0) return 0.0;

  const double n = a / ca;
  const double combined = std::pow(n,p);

  if (combined <= 0.0) return 0.0;

  const double projected = std::pow(combined, 2.0/p);
  if (projected >= 1.0) return 1.0;
  if (projected <= 0.0) return 0.0;

  return 0.5 * (1.0 - std::cos(pi * projected));
}

double FixCvhdGlobalDistortion::cvhdm_one_derivative(double a, double ca, double p) const
{
  if (ca <= 0.0 || p <= 0.0) return 0.0;

  const double q = std::pow(a / ca, p);
  if (q <= 0.0) return 0.0;

  const double u = std::pow(q, 2.0 / p);
  if (u >= 1.0 || u <= 0.0) return 0.0;

  const double pi = 3.141592653589793238462643383279502884;
  const double pref = std::pow(q, 2.0/p - 1.0) * pi * std::sin(pi * u);

  return pref * std::pow(a / ca, p - 1.0) / ca;
}

void FixCvhdGlobalDistortion::cvhdm_two_derivatives(double a, double b, double ca, double cb, double p,
                                                    double &da, double &db) const
{
  da = 0.0;
  db = 0.0;

  const double pi = 3.141592653589793238462643383279502884;
  if (ca <= 0.0 || cb <= 0.0 || p <= 0.0) return;

  const double na = a / ca;
  const double nb = b / cb;
  const double combined = std::pow(na,p) + std::pow(nb,p);

  if (combined <= 0.0) return;

  const double projected = std::pow(combined, 2.0/p);
  if (projected >= 1.0 || projected <= 0.0) return;

  const double prefactor = std::pow(combined, 2.0/p - 1.0) * pi * std::sin(pi * projected);

  da = prefactor * std::pow(na, p - 1.0) / ca;
  db = prefactor * std::pow(nb, p - 1.0) / cb;
}

bool FixCvhdGlobalDistortion::is_carbon(int i) const
{
  return atom->type[i] == carbon_type_;
}

bool FixCvhdGlobalDistortion::is_hydrogen(int i) const
{
  return atom->type[i] == hydrogen_type_;
}

bool FixCvhdGlobalDistortion::in_fix_group(int i) const
{
  return (atom->mask[i] & groupbit) != 0;
}

bool FixCvhdGlobalDistortion::is_owned_local_index(int i) const
{
  return i >= 0 && i < atom->nlocal;
}

void FixCvhdGlobalDistortion::open_output()
{
  if (comm->me != 0) return;
  if (ofs_.is_open()) return;

  ofs_.open(output_file_.c_str(), std::ios::out);
  if (!ofs_) error->one(FLERR, "Cannot open cvhd/global/distortion output file");

  ofs_ << "# step time ccbb chbb ccbf cv1 cv2 bias n_cc_bonds cc_break_events cc_form_events cvhd_events cvhd_wait opes_nker opes_sigma_cv1 opes_sigma_cv2 opes_rct opes_neff opes_prob opes_raw_bias opes_barrier opes_ct_slope opes_stable_windows opes_dE opes_damp_bias opes_boost\n";
  ofs_ << std::scientific << std::setprecision(16);
}

void FixCvhdGlobalDistortion::write_output()
{
  if (!ofs_.is_open()) return;

  // Keep ordinary diagnostic quantities compact.  Only hypertime can become
  // very large, so only the final opes_boost/hypertime column is printed in
  // scientific notation.
  ofs_ << std::defaultfloat << std::setprecision(10);

  ofs_ << update->ntimestep << " "
       << update->ntimestep * update->dt << " "
       << values_[0] << " "
       << values_[1] << " "
       << values_[2] << " "
       << values_[3] << " "
       << values_[4] << " "
       << values_[8] << " "
       << static_cast<long long>(values_[12]) << " "
       << static_cast<long long>(values_[13]) << " "
       << static_cast<long long>(values_[14]) << " "
       << static_cast<long long>(values_[26]) << " "
       << static_cast<long long>(values_[27]) << " "
       << static_cast<long long>(values_[18]) << " "
       << values_[19] << " "
       << values_[20] << " "
       << values_[21] << " "
       << values_[22] << " "
       << values_[23] << " "
       << values_[24] << " "
       << values_[28] << " "
       << values_[29] << " "
       << static_cast<long long>(values_[30]) << " "
       << values_[31] << " "
       << values_[32] << " "
       << std::scientific << std::setprecision(8) << values_[25]
       << std::defaultfloat << std::setprecision(10) << "\n";
}

void FixCvhdGlobalDistortion::print_summary() const
{
  if (comm->me != 0) return;

  utils::logmesg(lmp, "fix cvhd/global/distortion V3k-hybrid-perf summary:\n");
  utils::logmesg(lmp, "  config file: {}\n", config_file_);
  utils::logmesg(lmp, "  output file: {}\n", output_file_);
  utils::logmesg(lmp, "  kk_pair_cache_policy={} (Kokkos /kk only)\n", kk_pair_cache_policy_);
  utils::logmesg(lmp, "  kk_fused_neighbor_enable={}, max_cc_partners={}, max_ch_partners={} (Kokkos /kk only)\n",
                 static_cast<int>(kk_fused_neighbor_enable_), kk_max_cc_partners_, kk_max_ch_partners_);
  utils::logmesg(lmp, "  cvhd_timer_enable={}, stride={}, reset_each_print={}\n",
                 static_cast<int>(cvhd_timer_enable_), cvhd_timer_stride_,
                 static_cast<int>(cvhd_timer_reset_each_print_));
  utils::logmesg(lmp, "  MPI ranks: {}\n", comm->nprocs);
  utils::logmesg(lmp, "  carbon_type={}, hydrogen_type={}\n", carbon_type_, hydrogen_type_);
  utils::logmesg(lmp, "  cc_break global reference pairs: {}\n", cc_break_.pairs.size());
  utils::logmesg(lmp, "  ch_break global reference pairs: {}\n", ch_break_.pairs.size());
  utils::logmesg(lmp, "  cc_form current global candidate pairs: {}\n", last_cc_form_pair_count_);
  utils::logmesg(lmp, "  bias_enable={}\n", static_cast<int>(bias_enable_));
  utils::logmesg(lmp, "  OPES V3k-hybrid-perf: use_cv2={}, ncv={}, temp={}, kBT={}, barrier={}, gamma={}, prefactor={}, epsilon={}\n",
                 static_cast<int>(opes_use_cv2_), opes_ncv_, opes_temperature_, opes_kbt_,
                 opes_barrier_, opes_biasfactor_, opes_bias_prefactor_, opes_epsilon_);
  utils::logmesg(lmp, "  OPES V3k-hybrid-perf: pace={}, adaptive_sigma_stride={}, kernel_cutoff={}, sigma_min_hardcoded={}\n",
                 opes_pace_, opes_adaptive_sigma_stride_, opes_kernel_cutoff_,
                 OPES_SIGMA_MIN_HARDCODED);
  utils::logmesg(lmp, "  OPES units: kBT = force->boltz * temperature = {}\n", opes_kbt_);
  utils::logmesg(lmp, "  OPES compression: threshold={}, recursive_merge={}\n",
                 opes_compression_threshold_, static_cast<int>(opes_recursive_merge_));
  utils::logmesg(lmp, "  OPES adaptive E: E0={}, increment={}, max={}, ct_window={}, min_stable_windows={}, slope_threshold={}\n",
                 opes_barrier_initial_, opes_barrier_increment_, opes_barrier_max_, opes_ct_window_,
                 opes_ct_min_stable_windows_, opes_ct_slope_threshold_);
  utils::logmesg(lmp, "  OPES damped dE: enable={}, slowf={}, pos=({}, {}), grid={}\n",
                 static_cast<int>(opes_dE_damp_enable_), opes_dE_damp_slowf_,
                 opes_dE_damp_pos_[0], opes_dE_damp_pos_[1], opes_dE_damp_grid_);
  utils::logmesg(lmp, "  CVHD reset: enable={}, threshold={}, waittime={} steps; hypertime accumulated in log space\n",
                 static_cast<int>(cvhd_reset_enable_), cvhd_event_threshold_, cvhd_waittime_);
  utils::logmesg(lmp, "  cc_connectivity_update={}, cc_break_bond_threshold={}, cc_form_bond_threshold={}\n",
                 static_cast<int>(cc_connectivity_update_), cc_break_bond_threshold_, cc_form_bond_threshold_);
  utils::logmesg(lmp, "  cc_form_candidate_rmax={} (0 disables runtime upper filtering)\n",
                 cc_form_candidate_rmax_);
  utils::logmesg(lmp, "  carbon table size: {}, connected C-C bonds: {}\n",
                 carbon_tags_.size(), cc_break_.pairs.size());}

void FixCvhdGlobalDistortion::read_config(const char *filename)
{
  std::ifstream ifs(filename);
  if (!ifs) error->all(FLERR, "Cannot open cvhd/global/distortion config file");

  std::string line;
  int lineno = 0;

  while (std::getline(ifs,line)) {
    ++lineno;

    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0,hash);

    std::replace(line.begin(), line.end(), '=', ' ');
    line = trim(line);
    if (line.empty()) continue;

    std::istringstream iss(line);
    std::string key;
    iss >> key;

    if (key == "carbon_type") iss >> carbon_type_;
    else if (key == "hydrogen_type") iss >> hydrogen_type_;
    else if (key == "output_file") iss >> output_file_;
    else if (key == "output_stride") iss >> output_stride_;
    else if (key == "compute_stride") iss >> compute_stride_;
    else if (key == "kk_pair_cache_policy") iss >> kk_pair_cache_policy_;
    else if (key == "kk_fused_neighbor_enable") iss >> kk_fused_neighbor_enable_;
    else if (key == "kk_max_cc_partners") iss >> kk_max_cc_partners_;
    else if (key == "kk_max_ch_partners") iss >> kk_max_ch_partners_;
    else if (key == "cvhd_timer_enable") iss >> cvhd_timer_enable_;
    else if (key == "cvhd_timer_stride") iss >> cvhd_timer_stride_;
    else if (key == "cvhd_timer_reset_each_print") iss >> cvhd_timer_reset_each_print_;

    else if (key == "cvhd_power") iss >> cvhd_power_;
    else if (key == "cv1_cutoff_cc") iss >> cv1_cutoff_cc_;
    else if (key == "cv1_cutoff_ch") iss >> cv1_cutoff_ch_;
    else if (key == "cv2_cutoff_ccform") iss >> cv2_cutoff_ccform_;
    else if (key == "cc_form_candidate_rmax") iss >> cc_form_candidate_rmax_;

    else if (key == "bias_enable") iss >> bias_enable_;
    else if (key == "opes_use_cv2") iss >> opes_use_cv2_;

    else if (key == "opes_temperature") iss >> opes_temperature_;
    else if (key == "opes_barrier") iss >> opes_barrier_;
    else if (key == "opes_barrier_increment") iss >> opes_barrier_increment_;
    else if (key == "opes_barrier_max") iss >> opes_barrier_max_;
    else if (key == "opes_biasfactor") {
      iss >> opes_biasfactor_;
      if (opes_biasfactor_ > 0.0) opes_user_biasfactor_ = true;
    }
    else if (key == "opes_pace") iss >> opes_pace_;
    else if (key == "opes_adaptive_sigma_stride") iss >> opes_adaptive_sigma_stride_;
    else if (key == "opes_kernel_cutoff") iss >> opes_kernel_cutoff_;
    else if (key == "opes_compression_threshold") iss >> opes_compression_threshold_;
    else if (key == "opes_recursive_merge") iss >> opes_recursive_merge_;
    else if (key == "opes_epsilon") iss >> opes_epsilon_;
    else if (key == "opes_fixed_sigma") iss >> opes_fixed_sigma_;
    else if (key == "opes_no_zed") iss >> opes_no_zed_;
    else if (key == "opes_ct_window") iss >> opes_ct_window_;
    else if (key == "opes_ct_min_stable_windows") iss >> opes_ct_min_stable_windows_;
    else if (key == "opes_ct_slope_threshold") iss >> opes_ct_slope_threshold_;
    else if (key == "opes_dE_damp_enable") iss >> opes_dE_damp_enable_;
    else if (key == "opes_dE_damp_slowf") iss >> opes_dE_damp_slowf_;
    else if (key == "opes_dE_damp_pos_cv1") iss >> opes_dE_damp_pos_[0];
    else if (key == "opes_dE_damp_pos_cv2") iss >> opes_dE_damp_pos_[1];
    else if (key == "opes_dE_damp_grid") iss >> opes_dE_damp_grid_;

    else if (key == "cvhd_reset_enable") iss >> cvhd_reset_enable_;
    else if (key == "cvhd_waittime") iss >> cvhd_waittime_;
    else if (key == "cvhd_event_threshold") iss >> cvhd_event_threshold_;

    else if (key == "cc_connectivity_update") iss >> cc_connectivity_update_;
    else if (key == "cc_break_bond_threshold") iss >> cc_break_bond_threshold_;
    else if (key == "cc_form_bond_threshold") iss >> cc_form_bond_threshold_;

    else if (key == "cc_break_enable") iss >> cc_break_.enabled;
    else if (key == "cc_break_ref") iss >> cc_break_.ref;
    else if (key == "cc_break_rmin") iss >> cc_break_.rmin;
    else if (key == "cc_break_rmax") iss >> cc_break_.rmax;
    else if (key == "cc_break_power") iss >> cc_break_.power;
    else if (key == "cc_break_reset_ref") iss >> cc_break_.reset_ref;
    else if (key == "cc_break_reset_maxdist") iss >> cc_break_.reset_maxdist;
    else if (key == "cc_break_reset_time") iss >> cc_break_.reset_time;

    else if (key == "ch_break_enable") iss >> ch_break_.enabled;
    else if (key == "ch_break_ref") iss >> ch_break_.ref;
    else if (key == "ch_break_rmin") iss >> ch_break_.rmin;
    else if (key == "ch_break_rmax") iss >> ch_break_.rmax;
    else if (key == "ch_break_power") iss >> ch_break_.power;
    else if (key == "ch_break_reset_ref") iss >> ch_break_.reset_ref;
    else if (key == "ch_break_reset_maxdist") iss >> ch_break_.reset_maxdist;
    else if (key == "ch_break_reset_time") iss >> ch_break_.reset_time;

    else if (key == "cc_form_enable") iss >> cc_form_.enabled;
    else if (key == "cc_form_ref") iss >> cc_form_.ref;
    else if (key == "cc_form_rmin") iss >> cc_form_.rmin;
    else if (key == "cc_form_rmax") {
      // Parsed but intentionally ignored in V0.5+.
      iss >> cc_form_.rmax;
    }
    else if (key == "cc_form_power") iss >> cc_form_.power;
    else if (key == "cc_form_reset_ref") iss >> cc_form_.reset_ref;
    else if (key == "cc_form_reset_maxdist") iss >> cc_form_.reset_maxdist;
    else if (key == "cc_form_reset_time") iss >> cc_form_.reset_time;
    else if (key == "cc_form_nl_stride") iss >> cc_form_.nl_stride;

    else {
      std::ostringstream msg;
      msg << "Unknown key in cvhd/global/distortion config file at line "
          << lineno << ": " << key;
      error->all(FLERR, msg.str());
    }

    if (iss.fail() && !iss.eof()) {
      std::ostringstream msg;
      msg << "Failed to parse cvhd/global/distortion config file at line " << lineno;
      error->all(FLERR, msg.str());
    }
  }
}

void FixCvhdGlobalDistortion::validate_config() const
{
  if (carbon_type_ <= 0 || hydrogen_type_ <= 0) {
    error->all(FLERR, "cvhd/global/distortion config must define positive carbon_type and hydrogen_type");
  }

  if (carbon_type_ == hydrogen_type_) {
    error->all(FLERR, "carbon_type and hydrogen_type must be different");
  }

  if (kk_pair_cache_policy_ != "safe" && kk_pair_cache_policy_ != "ref_only") {
    error->all(FLERR, "kk_pair_cache_policy must be either safe or ref_only");
  }
  if (cvhd_timer_stride_ < 0) {
    error->all(FLERR, "cvhd_timer_stride must be non-negative");
  }
  if (kk_max_cc_partners_ <= 0 || kk_max_ch_partners_ <= 0) {
    error->all(FLERR, "kk_max_cc_partners and kk_max_ch_partners must be positive");
  }

  auto check_term = [&](const TermConfig &t, const char *name) {
    if (!t.enabled) return;
    if (t.ref <= 0.0) {
      std::ostringstream msg;
      msg << name << " ref must be positive";
      error->all(FLERR, msg.str());
    }
    if (!t.do_form && t.rmax < t.rmin) {
      std::ostringstream msg;
      msg << name << " rmax must be >= rmin";
      error->all(FLERR, msg.str());
    }
    if (t.power <= 0) {
      std::ostringstream msg;
      msg << name << " power must be positive";
      error->all(FLERR, msg.str());
    }
  };

  check_term(cc_break_, "cc_break");
  check_term(ch_break_, "ch_break");
  check_term(cc_form_, "cc_form");

  if (cvhd_power_ <= 0.0) error->all(FLERR, "cvhd_power must be positive");
  if (cv1_cutoff_cc_ <= 0.0 || cv1_cutoff_ch_ <= 0.0 || cv2_cutoff_ccform_ <= 0.0) {
    error->all(FLERR, "CVHD cutoffs must be positive");
  }
  if (cc_form_candidate_rmax_ < 0.0) error->all(FLERR, "cc_form_candidate_rmax must be non-negative");
  if (cc_form_candidate_rmax_ > 0.0 && cc_form_candidate_rmax_ < cc_form_.ref) {
    error->all(FLERR, "cc_form_candidate_rmax must be >= cc_form_ref, or 0 to disable");
  }
  if (opes_temperature_ <= 0.0) error->all(FLERR, "opes_temperature must be positive");
  if (opes_barrier_ <= 0.0) error->all(FLERR, "opes_barrier must be positive");
  if (opes_barrier_increment_ < 0.0) error->all(FLERR, "opes_barrier_increment must be non-negative");
  if (opes_barrier_max_ > 0.0 && opes_barrier_max_ < opes_barrier_) {
    error->all(FLERR, "opes_barrier_max must be >= opes_barrier, or 0 for no explicit cap");
  }
  if (opes_pace_ <= 0) error->all(FLERR, "opes_pace must be positive");
  if (opes_ct_window_ <= 1) error->all(FLERR, "opes_ct_window must be greater than 1");
  if (opes_ct_min_stable_windows_ <= 0) error->all(FLERR, "opes_ct_min_stable_windows must be positive");
  if (opes_ct_slope_threshold_ < 0.0) error->all(FLERR, "opes_ct_slope_threshold must be non-negative");
  if (opes_dE_damp_slowf_ < 0.0) error->all(FLERR, "opes_dE_damp_slowf must be non-negative");
  if (opes_dE_damp_pos_[0] < 0.0 || opes_dE_damp_pos_[0] > 1.0 ||
      opes_dE_damp_pos_[1] < 0.0 || opes_dE_damp_pos_[1] > 1.0) {
    error->all(FLERR, "opes_dE_damp_pos_cv1/cv2 must be in [0,1]");
  }
  if (opes_dE_damp_grid_ < 2) error->all(FLERR, "opes_dE_damp_grid must be at least 2");
  if (opes_kernel_cutoff_ <= 0.0) error->all(FLERR, "opes_kernel_cutoff must be positive");
  if (opes_compression_threshold_ < 0.0) error->all(FLERR, "opes_compression_threshold must be non-negative");
  if (cvhd_waittime_ <= 0) error->all(FLERR, "cvhd_waittime must be positive");
  if (cvhd_event_threshold_ <= 0.0 || cvhd_event_threshold_ > 1.0) {
    error->all(FLERR, "cvhd_event_threshold must be in (0,1]");
  }
  if (cc_break_bond_threshold_ < 0.0) error->all(FLERR, "cc_break_bond_threshold must be non-negative");
  if (cc_form_bond_threshold_ < 0.0) error->all(FLERR, "cc_form_bond_threshold must be non-negative");
}

std::string FixCvhdGlobalDistortion::trim(const std::string &s)
{
  const std::size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const std::size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::string FixCvhdGlobalDistortion::lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}
