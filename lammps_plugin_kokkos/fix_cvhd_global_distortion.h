/* ----------------------------------------------------------------------
   USER-CVHD: V3k-hybrid-perf reference implementation of global-distortion CVs for LAMMPS

   CPU reference fix:
     - MPI-safe global CV reductions
     - C-C/C-H reference-bond construction from the LAMMPS neighbor list
     - C-C formation computed from the LAMMPS neighbor list
     - computes ccbb/chbb/ccbf and CVHDM-projected cv1/cv2
     - applies a simple harmonic bias on cv1 for C-C/C-H breaking force validation
     - keeps only fixed-E OPES_CVHD bias with adaptive sigma

   Intended as a validation step before OPES/CVHD history and Kokkos support.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(cvhd/global/distortion,FixCvhdGlobalDistortion);
// clang-format on
#else

#ifndef LMP_FIX_CVHD_GLOBAL_DISTORTION_H
#define LMP_FIX_CVHD_GLOBAL_DISTORTION_H

#include "fix.h"

#include <array>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class NeighList;

class FixCvhdGlobalDistortion : public Fix {
 public:
  FixCvhdGlobalDistortion(class LAMMPS *, int, char **);
  ~FixCvhdGlobalDistortion() override;

  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void post_neighbor() override;
  void post_force(int) override;
  void end_of_step() override;
  double compute_vector(int) override;
  double compute_scalar() override;

 public:
  // These small nested data/config types are intentionally public for CUDA/Kokkos
  // plugin builds. NVCC generates CUDA stub code outside the class for kernels
  // launched from member functions containing KOKKOS_LAMBDA. The generated
  // wrapper must legally name FixCvhdGlobalDistortion::TermConfig.
  enum PairKind { CC_PAIR = 0, CH_PAIR = 1 };
  struct OpesKernel {
    double height;
    std::array<double,2> center;
    std::array<double,2> sigma;
  };

  struct PairRef {
    tagint itag;
    tagint jtag;
    double ref;
  };

  struct TermConfig {
    bool enabled = false;
    bool do_form = false;
    PairKind kind = CC_PAIR;

    double ref = 1.0;
    double rmin = 0.0;
    // For reference-bond construction this is the initial-distance cutoff.
    // For C-C formation this is ignored; the upper bound comes from LAMMPS'
    // neighbor list cutoff/skin.
    double rmax = 1.0;
    int power = 6;

    bool reset_ref = false;
    double reset_maxdist = 0.0;
    bigint reset_time = 0;
    bigint reset_wait = 0;
    bool rebuild_ref = true;

    bigint nl_stride = 0;
    bigint nl_wait = -1;

    // Global reference-pair list, replicated on all MPI ranks. Each pair is
    // stored by atom tags with itag < jtag.
    std::vector<PairRef> pairs;
  };

  std::string config_file_;
  std::string output_file_;
  std::ofstream ofs_;

  int carbon_type_;
  int hydrogen_type_;

  bigint output_stride_;
  bigint compute_stride_;

  // Kokkos-only cache invalidation policy for local-index breaking-pair caches.
  // "safe": dirty after every LAMMPS neighbor rebuild and when nlocal/nghost changes.
  // "ref_only": dirty only when reference pairs are rebuilt/reset.  This assumes
  //              single MPI rank and stable atom local indices, e.g. atom sorting off.
  std::string kk_pair_cache_policy_;

  // Experimental Kokkos fused neighbor backend.  CPU style ignores this.
  bool kk_fused_neighbor_enable_;
  int kk_max_cc_partners_;
  int kk_max_ch_partners_;

  // Host tag->local map cache. CPU backend rebuilds it every compute; Kokkos
  // hybrid backend rebuilds it only when local-index mapping may have changed.
  bool tag_map_dirty_;

  // Lightweight internal timers. Disabled by default.
  bool cvhd_timer_enable_;
  bigint cvhd_timer_stride_;
  bool cvhd_timer_reset_each_print_;
  bigint cvhd_timer_last_print_step_;

  double cvhd_t_compute_all_;
  double cvhd_t_tag_map_;
  double cvhd_t_ref_conn_;
  double cvhd_t_backend_cvs_;
  double cvhd_t_bias_;
  double cvhd_t_apply_total_;
  double cvhd_t_hypertime_;
  double cvhd_t_opes_update_;
  double cvhd_t_reset_;
  double cvhd_t_output_;

  double cvhd_t_pair_cache_;
  double cvhd_t_kk_reduce_;
  double cvhd_t_kk_force_;
  double cvhd_t_form_cpu_;
  double cvhd_t_fused_cache_;
  double cvhd_t_fused_cv_;
  double cvhd_t_fused_force_;
  double cvhd_t_cpu_break_force_;
  double cvhd_t_cpu_form_force_;

  long long cvhd_n_compute_all_;
  long long cvhd_n_tag_map_;
  long long cvhd_n_ref_conn_;
  long long cvhd_n_backend_cvs_;
  long long cvhd_n_bias_;
  long long cvhd_n_apply_total_;
  long long cvhd_n_hypertime_;
  long long cvhd_n_opes_update_;
  long long cvhd_n_reset_;
  long long cvhd_n_output_;

  long long cvhd_n_pair_cache_;
  long long cvhd_n_kk_reduce_;
  long long cvhd_n_kk_force_;
  long long cvhd_n_form_cpu_;
  long long cvhd_n_fused_cache_;
  long long cvhd_n_fused_cv_;
  long long cvhd_n_fused_force_;
  long long cvhd_n_cpu_break_force_;
  long long cvhd_n_cpu_form_force_;

  double cvhd_power_;
  double cv1_cutoff_cc_;
  double cv1_cutoff_ch_;
  double cv2_cutoff_ccform_;

  // Optional runtime upper cutoff for C-C formation candidates. This is not
  // the formation event threshold. It is only a performance filter applied
  // before evaluating ccbf; 0 disables it.
  double cc_form_candidate_rmax_;

  bool bias_enable_;

  // V2b-clean fixed-E OPES_CVHD bias parameters/state.
  // If opes_use_cv2_ is false, OPES is built only in the cv1 dimension
  // and dV/dcv2 is forced to zero.
  bool opes_use_cv2_;
  int opes_ncv_;

  double opes_temperature_;
  double opes_kbt_;
  double opes_barrier_;
  double opes_barrier_initial_;
  double opes_barrier_increment_;
  double opes_barrier_max_;
  double opes_biasfactor_;
  double opes_bias_prefactor_;
  double opes_epsilon_;
  bool opes_user_biasfactor_;
  int opes_pace_;
  int opes_adaptive_sigma_stride_;
  int opes_adaptive_counter_;
  int opes_counter_;
  int opes_reweight_count_;
  bigint opes_last_update_step_;

  int opes_ct_window_;
  int opes_ct_min_stable_windows_;
  int opes_ct_sample_count_;
  int opes_ct_stable_windows_;
  double opes_ct_slope_threshold_;
  double opes_ct_slope_;

  bool opes_dE_damp_enable_;
  double opes_dE_damp_slowf_;
  std::array<double,2> opes_dE_damp_pos_;
  int opes_dE_damp_grid_;
  double opes_last_dE_;
  double opes_last_damp_bias_;

  std::vector<double> opes_ct_values_;
  double opes_kernel_cutoff_;
  double opes_kernel_cutoff2_;
  double opes_val_at_cutoff_;
  double opes_compression_threshold_;
  double opes_compression_threshold2_;
  bool opes_recursive_merge_;
  bool opes_fixed_sigma_;
  bool opes_no_zed_;

  std::array<double,2> opes_sigma0_;
  // Hard-coded lower bound for adaptive sigma. This replaces the old
  // configurable sigma-min knobs.
  static constexpr double OPES_SIGMA_MIN_HARDCODED = 1.0e-6;
  std::array<double,2> opes_av_cv_;
  std::array<double,2> opes_av_M2_;

  double opes_sum_weights_;
  double opes_sum_weights2_;
  double opes_reweight_sum_;
  double opes_KDEnorm_;
  double opes_Zed_;
  double opes_neff_;
  double opes_rct_;
  double opes_prob_;
  double opes_raw_bias_;
  double opes_boost_bias_;
  // Cumulative hyperdynamics time.  V2d accumulates it in log space to avoid
  // overflow, then outputs a finite capped double in the historical column
  // named "opes_boost".
  double opes_hypertime_;
  double opes_log_hypertime_;
  bool opes_hypertime_overflowed_;
  bigint opes_last_hypertime_step_;

  bool cvhd_reset_enable_;
  double cvhd_event_threshold_;
  bigint cvhd_waittime_;
  bigint cvhd_wait_counter_;
  long long cvhd_event_count_;

  std::vector<OpesKernel> opes_kernels_;

  TermConfig cc_break_;
  TermConfig ch_break_;
  TermConfig cc_form_;

  // 0 ccbb
  // 1 chbb
  // 2 ccbf
  // 3 cv1
  // 4 cv2
  // 5 n_ccbb
  // 6 n_chbb
  // 7 n_ccbf
  // 8 harmonic bias energy
  // 9 dV/dcv1
  // 10 dV/dccbb
  // 11 dV/dchbb
  // 12 current number of connected C-C bonds
  // 13 cumulative number of C-C bond-breaking connectivity events
  // 14 cumulative number of C-C bond-formation connectivity events
  // 15 number of carbon atoms in the replicated connectivity table
  // 16 dV/dcv2
  // 17 dV/dccbf
  // 18 OPES number of kernels
  // 19 OPES sigma(cv1)
  // 20 OPES sigma(cv2)
  // 21 OPES rct = kBT ln <exp(beta V)>
  // 22 OPES neff
  // 23 OPES probability estimate
  // 24 OPES raw bias without constant shift
  // 25 OPES cumulative hypertime.  The output header keeps the historical
  //    column name "opes_boost" for compatibility with earlier test scripts.
  // 26 CVHD event count
  // 27 CVHD wait counter
  // 28 current OPES barrier E
  // 29 c(t) slope used for E gate
  // 30 consecutive stable c(t) windows
  // 31 last effective E increment after damping
  // 32 boost bias at damping position
  double values_[33];

  bigint last_compute_step_;
  bool initialized_;
  bool summary_printed_;

  class NeighList *list_;
  std::size_t last_cc_form_pair_count_;

  bool cc_connectivity_update_;
  double cc_break_bond_threshold_;
  double cc_form_bond_threshold_;
  long long cc_event_break_count_;
  long long cc_event_form_count_;

  // Replicated C-atom index and adjacency table.
  // carbon_tags_[ci] gives the global tag of compact C index ci.
  // carbon_tag_to_index_ maps global C tags to compact indices.
  // cc_adj_[ci] stores compact indices of C atoms currently bonded to ci.
  std::vector<tagint> carbon_tags_;
  std::unordered_map<tagint,int> carbon_tag_to_index_;
  std::vector<std::vector<int>> cc_adj_;

  // tag -> current local index, including owned and ghost atoms.
  std::unordered_map<tagint,int> tag_to_local_;

  void set_defaults();
  void read_config(const char *filename);
  void validate_config() const;
  void open_output();

  void build_tag_map();
  void build_reference_pairs(TermConfig &term);
  void gather_unique_reference_pairs(TermConfig &term, const std::vector<PairRef> &local_pairs);
  void build_carbon_index();
  void rebuild_cc_connectivity_from_cc_break_pairs();
  void rebuild_cc_break_pairs_from_connectivity();
  void update_cc_connectivity();
  void gather_unique_event_pairs(const std::vector<PairRef> &local_pairs, std::vector<PairRef> &global_pairs) const;
  void gather_unique_pairs_with_ref(const std::vector<PairRef> &local_pairs,
                                    std::vector<PairRef> &global_pairs,
                                    double ref) const;
  void build_current_formation_force_pairs(std::vector<PairRef> &global_pairs);
  bool are_cc_connected(tagint itag, tagint jtag) const;
  void set_cc_connected(tagint itag, tagint jtag, bool connected);

 protected:
  // V3a Kokkos-prep backend boundary.
  //
  // The base class keeps all host-side state-machine logic:
  //   - cfg parsing
  //   - OPES kernels/compression
  //   - adaptive E / c(t) / damping
  //   - hypertime
  //   - reset/reference rebuild/output
  //
  // CPU and future Kokkos variants should differ only in these backend
  // methods.  The Kokkos subclass will override them to read Kokkos atom
  // views and Kokkos neighbor lists directly, avoiding host-device x/f syncs.
  virtual int backend_neighbor_request_flags() const;
  virtual bool backend_can_skip_tag_map_each_compute() const;
  virtual void backend_sync_host_atoms_for_tag_map();
  virtual void backend_sync_host_atoms_for_reference();
  virtual bool backend_collect_reference_pairs(const TermConfig &term, std::vector<PairRef> &local_pairs);
  virtual void backend_on_reference_pairs_changed();
  virtual void backend_on_connectivity_changed();
  virtual void backend_compute_raw_cvs(double &ccbb, double &chbb, double &ccbf);
  virtual void backend_apply_bias_forces();

  double compute_term_value(const TermConfig &term);
  double compute_pair_contribution(const TermConfig &term, int i, int j, double ref) const;
  double compute_pair_contribution_from_r(const TermConfig &term, double r, double ref) const;
  double compute_formation_value_from_neighbor_list();
  double pair_distance(int i, int j) const;

  double int_pow(double x, int p) const;
  double cvhdm_two(double a, double b, double ca, double cb, double p) const;
  double cvhdm_one(double a, double ca, double p) const;
  double cvhdm_one_derivative(double a, double ca, double p) const;
  void cvhdm_two_derivatives(double a, double b, double ca, double cb, double p,
                             double &da, double &db) const;

  void compute_bias_derivatives();
  void initialize_opes_state();
  void opes_update_barrier_dependent_parameters();
  void opes_reset_ct_accumulator();
  void opes_update_barrier_gate();
  double opes_compute_ct_slope() const;
  double opes_evaluate_boost_at_cv(const std::array<double,2> &cv) const;
  double opes_evaluate_damping_bias() const;
  void compute_opes_bias_and_derivatives(double cv1, double cv2);
  double evaluate_opes_kernel(const OpesKernel &k,
                              const std::array<double,2> &cv,
                              std::array<double,2> *der) const;
  void opes_update_after_step();
  void opes_accumulate_hypertime();
  void cvhd_check_event_and_reset();
  void cvhd_perform_reset();
  void opes_deposit_kernel(const std::array<double,2> &cv);
  void opes_add_kernel(double height,
                       const std::array<double,2> &center,
                       const std::array<double,2> &sigma);
  int opes_get_mergeable_kernel(const std::array<double,2> &center, int exclude) const;
  void opes_merge_kernel_into(OpesKernel &taker, const OpesKernel &giver) const;
  void opes_reset_history();
  double opes_clamp_sigma(int i, double sigma) const;

  void apply_breaking_forces(const TermConfig &term, double raw_value, double dV_draw);
  void apply_pair_breaking_force(const TermConfig &term, int i, int j, double ref,
                                 double term_prefactor, double dV_draw);
  void apply_formation_forces(double raw_value, double dV_draw);
  void apply_pair_formation_force(int i, int j, double ref,
                                  double term_prefactor, double dV_draw);

  bool is_carbon(int i) const;
  bool is_hydrogen(int i) const;
  bool in_fix_group(int i) const;
  bool is_owned_local_index(int i) const;

  void mark_tag_map_dirty();
  void ensure_tag_map_current();

  double cvhd_timer_now() const;
  void cvhd_timer_add(double &accum, long long &count, double t0);
  void cvhd_timer_reset_accumulators();
  void cvhd_timer_maybe_print();

  void compute_all();
  void maybe_handle_reset(TermConfig &term, double value, bigint step);
  void write_output();
  void print_summary() const;

  static std::string trim(const std::string &s);
  static std::string lower(std::string s);
};

}    // namespace LAMMPS_NS

#endif
#endif
