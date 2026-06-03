/* ----------------------------------------------------------------------
   USER-CVHD: V3k experimental /kk subclass

   Implemented in V3k:
     - Kokkos device reductions for ccbb/chbb reference-pair raw CVs
     - Kokkos device force application for ccbb/chbb bias forces

   Still CPU fallback in V3k:
     - ccbf formation CV
     - ccbf formation force
     - OPES state machine
     - reset/reference rebuild/connectivity logic
------------------------------------------------------------------------- */

#include "fix_cvhd_global_distortion_kokkos.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <vector>

using namespace LAMMPS_NS;

FixCvhdGlobalDistortionKokkos::FixCvhdGlobalDistortionKokkos(LAMMPS *lmp,
                                                             int narg,
                                                             char **arg)
  : FixCvhdGlobalDistortion(lmp, narg, arg)
{
#ifdef LMP_KOKKOS
  kokkosable = 1;
  atomKK = static_cast<AtomKokkos *>(atom);
  execution_space = ExecutionSpaceFromDevice<LMPDeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK | MASK_MASK;
  datamask_modify = F_MASK;
  k_list_ = nullptr;

  pair_cache_dirty_ = true;
  fused_cache_dirty_ = true;
  pair_cache_last_build_step_ = -1;
  pair_cache_last_nlocal_ = -1;
  pair_cache_last_nghost_ = -1;
#endif

  if (comm && comm->me == 0) {
#ifdef LMP_KOKKOS
    utils::logmesg(lmp,
                   "fix cvhd/global/distortion/kk V3k: LMP_KOKKOS enabled; "
                   "fused_neighbor={}, breaking backend=Kokkos, formation backend={}, "
                   "pair_cache_policy={}\n",
                   static_cast<int>(kk_fused_neighbor_enable_),
                   kk_fused_neighbor_enable_ ? "Kokkos fused" : "CPU fallback",
                   kk_pair_cache_policy_);
#else
    error->all(FLERR,
               "cvhd/global/distortion/kk was compiled without -DLMP_KOKKOS. "
               "Rebuild the plugin with LMP_KOKKOS and the correct CUDA architecture, "
               "or use cvhd/global/distortion for the CPU backend.");
#endif
  }
}

void FixCvhdGlobalDistortionKokkos::init()
{
  FixCvhdGlobalDistortion::init();

#ifdef LMP_KOKKOS
  auto *request = neighbor->find_request(this);
  if (!request) {
    error->all(FLERR,
               "fix cvhd/global/distortion/kk: neighbor request not found after init");
  }

  // LAMMPS 22Jul2025 has no NeighConst::REQ_KOKKOS_DEVICE.  Kokkos lists are
  // requested by setting these NeighRequest fields after add_request().
  request->set_kokkos_host(0);
  request->set_kokkos_device(1);
#endif
}

void FixCvhdGlobalDistortionKokkos::init_list(int id, NeighList *ptr)
{
  FixCvhdGlobalDistortion::init_list(id, ptr);

#ifdef LMP_KOKKOS
  k_list_ = static_cast<NeighListKokkos<DeviceType> *>(ptr);
  if (!k_list_) {
    error->all(FLERR,
               "fix cvhd/global/distortion/kk: init_list did not receive NeighListKokkos; "
               "check that the neighbor log shows kokkos_device and bin: kk/device");
  }
#endif
}

bool FixCvhdGlobalDistortionKokkos::backend_can_skip_tag_map_each_compute() const
{
  // Kokkos backend uses device pair/fused caches during normal steps.  Host
  // tag->local map is only required when a local-index cache is rebuilt.
  return true;
}

void FixCvhdGlobalDistortionKokkos::backend_sync_host_atoms_for_tag_map()
{
#ifdef LMP_KOKKOS
  if (!atomKK) atomKK = static_cast<AtomKokkos *>(atom);
  atomKK->sync(Host, TAG_MASK);
#endif
}

void FixCvhdGlobalDistortionKokkos::backend_sync_host_atoms_for_reference()
{
#ifdef LMP_KOKKOS
  if (!atomKK) atomKK = static_cast<AtomKokkos *>(atom);
  atomKK->sync(Host, X_MASK | TYPE_MASK | TAG_MASK | MASK_MASK);
#endif
}

bool FixCvhdGlobalDistortionKokkos::backend_collect_reference_pairs(const TermConfig &term,
                                                                    std::vector<PairRef> &local_pairs)
{
#ifndef LMP_KOKKOS
  return false;
#else
  if (!k_list_) return false;
  if (!atomKK) atomKK = static_cast<AtomKokkos *>(atom);

  atomKK->sync(Device, X_MASK | TYPE_MASK | TAG_MASK | MASK_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();

  auto d_ilist = k_list_->d_ilist;
  auto d_numneigh = k_list_->d_numneigh;
  auto d_neighbors = k_list_->d_neighbors;
  const int inum = list_->inum;

  const int carbon_type = carbon_type_;
  const int hydrogen_type = hydrogen_type_;
  const int groupbit_local = groupbit;
  const bool want_cc = (term.kind == CC_PAIR);
  const double rmin2 = term.rmin * term.rmin;
  const double rmax2 = term.rmax * term.rmax;

  const int periodic0 = domain->periodicity[0];
  const int periodic1 = domain->periodicity[1];
  const int periodic2 = domain->periodicity[2];
  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double hx = 0.5 * xprd;
  const double hy = 0.5 * yprd;
  const double hz = 0.5 * zprd;

  int local_count = 0;
  Kokkos::parallel_reduce("cvhd_refpair_count",
                          Kokkos::RangePolicy<DeviceType>(0, inum),
                          KOKKOS_LAMBDA(const int ii, int &sum) {
    const int i = d_ilist(ii);
    if (!(mask(i) & groupbit_local)) return;

    const int ti = type(i);
    const bool i_c = (ti == carbon_type);
    const bool i_h = (ti == hydrogen_type);

    const int jnum = d_numneigh(i);
    for (int jj = 0; jj < jnum; ++jj) {
      const int j = d_neighbors(i,jj) & NEIGHMASK;
      if (j == i) continue;
      if (!(mask(j) & groupbit_local)) continue;

      const int tj = type(j);
      const bool j_c = (tj == carbon_type);
      const bool j_h = (tj == hydrogen_type);

      bool type_ok = false;
      if (want_cc) type_ok = i_c && j_c;
      else type_ok = (i_c && j_h) || (i_h && j_c);
      if (!type_ok) continue;

      double dx = x(j,0) - x(i,0);
      double dy = x(j,1) - x(i,1);
      double dz = x(j,2) - x(i,2);

      if (periodic0) {
        if (dx > hx) dx -= xprd;
        else if (dx < -hx) dx += xprd;
      }
      if (periodic1) {
        if (dy > hy) dy -= yprd;
        else if (dy < -hy) dy += yprd;
      }
      if (periodic2) {
        if (dz > hz) dz -= zprd;
        else if (dz < -hz) dz += zprd;
      }

      const double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq < rmin2 || rsq > rmax2) continue;
      ++sum;
    }
  }, local_count);

  local_pairs.clear();
  if (local_count <= 0) return true;

  Kokkos::View<tagint*, DeviceType> d_a("cvhd_refpair_a", local_count);
  Kokkos::View<tagint*, DeviceType> d_b("cvhd_refpair_b", local_count);
  Kokkos::View<int*, DeviceType> d_counter("cvhd_refpair_counter", 1);
  Kokkos::deep_copy(d_counter, 0);

  Kokkos::parallel_for("cvhd_refpair_fill",
                       Kokkos::RangePolicy<DeviceType>(0, inum),
                       KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    if (!(mask(i) & groupbit_local)) return;

    const int ti = type(i);
    const bool i_c = (ti == carbon_type);
    const bool i_h = (ti == hydrogen_type);

    const int jnum = d_numneigh(i);
    for (int jj = 0; jj < jnum; ++jj) {
      const int j = d_neighbors(i,jj) & NEIGHMASK;
      if (j == i) continue;
      if (!(mask(j) & groupbit_local)) continue;

      const int tj = type(j);
      const bool j_c = (tj == carbon_type);
      const bool j_h = (tj == hydrogen_type);

      bool type_ok = false;
      if (want_cc) type_ok = i_c && j_c;
      else type_ok = (i_c && j_h) || (i_h && j_c);
      if (!type_ok) continue;

      double dx = x(j,0) - x(i,0);
      double dy = x(j,1) - x(i,1);
      double dz = x(j,2) - x(i,2);

      if (periodic0) {
        if (dx > hx) dx -= xprd;
        else if (dx < -hx) dx += xprd;
      }
      if (periodic1) {
        if (dy > hy) dy -= yprd;
        else if (dy < -hy) dy += yprd;
      }
      if (periodic2) {
        if (dz > hz) dz -= zprd;
        else if (dz < -hz) dz += zprd;
      }

      const double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq < rmin2 || rsq > rmax2) continue;

      tagint a = tag(i);
      tagint b = tag(j);
      if (a == b) continue;
      if (b < a) {
        const tagint tmp = a;
        a = b;
        b = tmp;
      }

      const int idx = Kokkos::atomic_fetch_add(&d_counter(0), 1);
      if (idx < local_count) {
        d_a(idx) = a;
        d_b(idx) = b;
      }
    }
  });

  auto h_a = Kokkos::create_mirror_view(d_a);
  auto h_b = Kokkos::create_mirror_view(d_b);
  Kokkos::deep_copy(h_a, d_a);
  Kokkos::deep_copy(h_b, d_b);

  auto h_counter = Kokkos::create_mirror_view(d_counter);
  Kokkos::deep_copy(h_counter, d_counter);
  const int actual_count = std::min(local_count, h_counter(0));

  local_pairs.reserve(static_cast<std::size_t>(actual_count));
  for (int p = 0; p < actual_count; ++p) {
    PairRef pr;
    pr.itag = h_a(p);
    pr.jtag = h_b(p);
    pr.ref = term.ref;
    local_pairs.push_back(pr);
  }

  return true;
#endif
}

void FixCvhdGlobalDistortionKokkos::post_neighbor()
{
  FixCvhdGlobalDistortion::post_neighbor();
#ifdef LMP_KOKKOS
  // Neighbor rebuild/migration/sorting can change tag -> local-index mapping.
  // In safe mode, rebuild local-index pair caches after each such notification.
  // In ref_only mode, assume stable local indices and rebuild only after
  // setup/reset/rebuild_ref reference-pair changes.
  if (kk_pair_cache_policy_ == "safe") {
    mark_pair_cache_dirty();
  }

  // The fused cache stores local-index -> carbon-index mapping.  It is more
  // fragile than the reference-pair cache and must be rebuilt after every
  // neighbor rebuild in fused mode, even if kk_pair_cache_policy=ref_only.
  if (kk_fused_neighbor_enable_) {
    mark_fused_cache_dirty();
  }
#endif
}

#ifdef LMP_KOKKOS
void FixCvhdGlobalDistortionKokkos::mark_pair_cache_dirty()
{
  pair_cache_dirty_ = true;
}

void FixCvhdGlobalDistortionKokkos::mark_fused_cache_dirty()
{
  fused_cache_dirty_ = true;
}
#endif

void FixCvhdGlobalDistortionKokkos::backend_on_reference_pairs_changed()
{
#ifdef LMP_KOKKOS
  // Reference pairs changed after setup/reset/rebuild_ref.  Their local-index
  // device caches are no longer valid.
  mark_pair_cache_dirty();
  mark_fused_cache_dirty();
#endif
  FixCvhdGlobalDistortion::backend_on_reference_pairs_changed();
}

void FixCvhdGlobalDistortionKokkos::backend_on_connectivity_changed()
{
#ifdef LMP_KOKKOS
  // C-C connectivity changed; the fused C-index partner table must be rebuilt.
  mark_fused_cache_dirty();
#endif
  FixCvhdGlobalDistortion::backend_on_connectivity_changed();
}

#ifdef LMP_KOKKOS
void FixCvhdGlobalDistortionKokkos::update_pair_cache_from_tags(const TermConfig &term,
                                                                DevicePairCache &cache,
                                                                const char *label)
{
  const double t_timer = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;

  ensure_tag_map_current();

  std::vector<int> h_i;
  std::vector<int> h_j;
  std::vector<double> h_ref;
  std::vector<int> h_count;

  h_i.reserve(term.pairs.size());
  h_j.reserve(term.pairs.size());
  h_ref.reserve(term.pairs.size());
  h_count.reserve(term.pairs.size());

  for (const PairRef &p : term.pairs) {
    const auto ii = tag_to_local_.find(p.itag);
    const auto jj = tag_to_local_.find(p.jtag);
    if (ii == tag_to_local_.end() || jj == tag_to_local_.end()) continue;

    const int ilocal = ii->second;
    const int jlocal = jj->second;
    const bool i_owned = is_owned_local_index(ilocal);
    const bool j_owned = is_owned_local_index(jlocal);

    // Include pairs relevant to this rank's force update.  The reduction uses
    // h_count to preserve the CPU convention: only the rank owning itag
    // contributes the scalar pair value.
    if (!i_owned && !j_owned) continue;

    h_i.push_back(ilocal);
    h_j.push_back(jlocal);
    h_ref.push_back(p.ref);
    h_count.push_back(i_owned ? 1 : 0);
  }

  const int n = static_cast<int>(h_i.size());
  cache.n = n;

  cache.i = Kokkos::View<int*, DeviceType>(std::string("cvhd_") + label + "_i", n);
  cache.j = Kokkos::View<int*, DeviceType>(std::string("cvhd_") + label + "_j", n);
  cache.ref = Kokkos::View<double*, DeviceType>(std::string("cvhd_") + label + "_ref", n);
  cache.count = Kokkos::View<int*, DeviceType>(std::string("cvhd_") + label + "_count", n);

  auto hi = Kokkos::create_mirror_view(cache.i);
  auto hj = Kokkos::create_mirror_view(cache.j);
  auto href = Kokkos::create_mirror_view(cache.ref);
  auto hcnt = Kokkos::create_mirror_view(cache.count);

  for (int p = 0; p < n; ++p) {
    hi(p) = h_i[p];
    hj(p) = h_j[p];
    href(p) = h_ref[p];
    hcnt(p) = h_count[p];
  }

  Kokkos::deep_copy(cache.i, hi);
  Kokkos::deep_copy(cache.j, hj);
  Kokkos::deep_copy(cache.ref, href);
  Kokkos::deep_copy(cache.count, hcnt);

  cvhd_timer_add(cvhd_t_pair_cache_, cvhd_n_pair_cache_, t_timer);
}

void FixCvhdGlobalDistortionKokkos::ensure_pair_caches_current()
{
  const bool size_changed =
    (kk_pair_cache_policy_ == "safe") &&
    ((pair_cache_last_nlocal_ != atom->nlocal) ||
     (pair_cache_last_nghost_ != atom->nghost));

  if (!pair_cache_dirty_ && !size_changed) return;

  if (cc_break_.enabled) {
    update_pair_cache_from_tags(cc_break_, cc_pairs_dev_, "cc");
  } else {
    cc_pairs_dev_.n = 0;
  }

  if (ch_break_.enabled) {
    update_pair_cache_from_tags(ch_break_, ch_pairs_dev_, "ch");
  } else {
    ch_pairs_dev_.n = 0;
  }

  pair_cache_dirty_ = false;
  pair_cache_last_build_step_ = update->ntimestep;
  pair_cache_last_nlocal_ = atom->nlocal;
  pair_cache_last_nghost_ = atom->nghost;
}


void FixCvhdGlobalDistortionKokkos::update_fused_cache_from_host()
{
  const double t_timer = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;

  if (!atomKK) atomKK = static_cast<AtomKokkos *>(atom);
  // Critical for fused backend: this cache is built from host tag/type/mask,
  // while kernels use device tag/type/mask.  Synchronize the host mirror before
  // constructing local_to_cidx.
  atomKK->sync(Host, TAG_MASK | TYPE_MASK | MASK_MASK);

  const int nall = atom->nlocal + atom->nghost;
  const int ncarbon = static_cast<int>(carbon_tags_.size());
  const int max_cc = kk_max_cc_partners_;
  const int max_ch = kk_max_ch_partners_;

  fused_dev_.nall = nall;
  fused_dev_.ncarbon = ncarbon;
  fused_dev_.max_cc = max_cc;
  fused_dev_.max_ch = max_ch;

  fused_dev_.local_to_cidx = Kokkos::View<int*, DeviceType>("cvhd_fused_local_to_cidx", nall);
  fused_dev_.cc_partner_count = Kokkos::View<int*, DeviceType>("cvhd_fused_cc_count", ncarbon);
  fused_dev_.cc_partner_cidx = Kokkos::View<int**, DeviceType>("cvhd_fused_cc_partner", ncarbon, max_cc);
  fused_dev_.ch_partner_count = Kokkos::View<int*, DeviceType>("cvhd_fused_ch_count", ncarbon);
  fused_dev_.ch_partner_tag = Kokkos::View<tagint**, DeviceType>("cvhd_fused_ch_tag", ncarbon, max_ch);

  auto h_l2c = Kokkos::create_mirror_view(fused_dev_.local_to_cidx);
  auto h_cc_count = Kokkos::create_mirror_view(fused_dev_.cc_partner_count);
  auto h_cc = Kokkos::create_mirror_view(fused_dev_.cc_partner_cidx);
  auto h_ch_count = Kokkos::create_mirror_view(fused_dev_.ch_partner_count);
  auto h_ch = Kokkos::create_mirror_view(fused_dev_.ch_partner_tag);

  for (int i = 0; i < nall; ++i) h_l2c(i) = -1;
  for (int ci = 0; ci < ncarbon; ++ci) {
    h_cc_count(ci) = 0;
    h_ch_count(ci) = 0;
    for (int k = 0; k < max_cc; ++k) h_cc(ci,k) = -1;
    for (int k = 0; k < max_ch; ++k) h_ch(ci,k) = 0;
  }

  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;

  for (int i = 0; i < nall; ++i) {
    if (!(mask[i] & groupbit)) continue;
    if (type[i] != carbon_type_) continue;
    const auto it = carbon_tag_to_index_.find(tag[i]);
    if (it != carbon_tag_to_index_.end()) h_l2c(i) = it->second;
  }

  for (int ci = 0; ci < ncarbon; ++ci) {
    const int ncc = static_cast<int>(cc_adj_[ci].size());
    if (ncc > max_cc) {
      error->all(FLERR, "cvhd/global/distortion/kk fused cache exceeded kk_max_cc_partners");
    }
    h_cc_count(ci) = ncc;
    for (int k = 0; k < ncc; ++k) h_cc(ci,k) = cc_adj_[ci][k];
  }

  for (const PairRef &p : ch_break_.pairs) {
    int ci = -1;
    tagint htag = 0;

    const auto ii = carbon_tag_to_index_.find(p.itag);
    const auto jj = carbon_tag_to_index_.find(p.jtag);
    if (ii != carbon_tag_to_index_.end() && jj == carbon_tag_to_index_.end()) {
      ci = ii->second;
      htag = p.jtag;
    } else if (jj != carbon_tag_to_index_.end() && ii == carbon_tag_to_index_.end()) {
      ci = jj->second;
      htag = p.itag;
    } else {
      continue;
    }

    const int k = h_ch_count(ci);
    if (k >= max_ch) {
      error->all(FLERR, "cvhd/global/distortion/kk fused cache exceeded kk_max_ch_partners");
    }
    h_ch(ci,k) = htag;
    h_ch_count(ci) = k + 1;
  }

  Kokkos::deep_copy(fused_dev_.local_to_cidx, h_l2c);
  Kokkos::deep_copy(fused_dev_.cc_partner_count, h_cc_count);
  Kokkos::deep_copy(fused_dev_.cc_partner_cidx, h_cc);
  Kokkos::deep_copy(fused_dev_.ch_partner_count, h_ch_count);
  Kokkos::deep_copy(fused_dev_.ch_partner_tag, h_ch);

  fused_cache_dirty_ = false;
  cvhd_timer_add(cvhd_t_fused_cache_, cvhd_n_fused_cache_, t_timer);
}

void FixCvhdGlobalDistortionKokkos::ensure_fused_cache_current()
{
  const bool size_changed = (fused_dev_.nall != atom->nlocal + atom->nghost);

  if (!fused_cache_dirty_ && !size_changed) return;
  update_fused_cache_from_host();
}

void FixCvhdGlobalDistortionKokkos::compute_fused_raw_cvs_kokkos(double &ccbb,
                                                                 double &chbb,
                                                                 double &ccbf)
{
  const double t_timer = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;

  auto *k_list = k_list_;
  if (!k_list) {
    error->all(FLERR,
               "cvhd/global/distortion/kk fused backend did not receive NeighListKokkos; "
               "check that init() calls request->set_kokkos_device(1) and neighbor log shows kokkos_device/bin: kk/device");
  }

  if (!atomKK) atomKK = static_cast<AtomKokkos *>(atom);

  ensure_fused_cache_current();

  atomKK->sync(Device, X_MASK | TYPE_MASK | TAG_MASK | MASK_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();

  auto d_ilist = k_list->d_ilist;
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  const int inum = list_->inum;

  const auto local_to_cidx = fused_dev_.local_to_cidx;
  const auto cc_count = fused_dev_.cc_partner_count;
  const auto cc_partner = fused_dev_.cc_partner_cidx;
  const auto ch_count = fused_dev_.ch_partner_count;
  const auto ch_tag = fused_dev_.ch_partner_tag;

  Kokkos::View<double*, DeviceType> sums("cvhd_fused_sums", 4);
  Kokkos::deep_copy(sums, 0.0);

  const int carbon_type = carbon_type_;
  const int hydrogen_type = hydrogen_type_;
  const int groupbit_local = groupbit;

  // V3k hybrid CV: fused neighbor-list reduction computes formation ccbf only.
  // Breaking CVs ccbb/chbb are computed from reference-pair caches so that a
  // stretched bond cannot disappear from the CV when it leaves the neighbor list.
  const bool cf_enabled = cc_form_.enabled;

  const double cf_ref = cc_form_.ref;
  const int cf_power = cc_form_.power;
  double cf_cut = cf_ref;
  if (cc_form_candidate_rmax_ > 0.0 && cc_form_candidate_rmax_ < cf_cut) cf_cut = cc_form_candidate_rmax_;
  const double cf_cut2 = cf_cut * cf_cut;

  const int periodic0 = domain->periodicity[0];
  const int periodic1 = domain->periodicity[1];
  const int periodic2 = domain->periodicity[2];
  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double hx = 0.5 * xprd;
  const double hy = 0.5 * yprd;
  const double hz = 0.5 * zprd;

  Kokkos::parallel_for("cvhd_fused_neighbor_cvs",
                       Kokkos::RangePolicy<DeviceType>(0, inum),
                       KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    if (!(mask(i) & groupbit_local)) return;

    const int ti = type(i);
    const int ci = local_to_cidx(i);
    const bool i_is_c = (ti == carbon_type && ci >= 0);
    const bool i_is_h = (ti == hydrogen_type);

    const int jnum = d_numneigh(i);
    for (int jj = 0; jj < jnum; ++jj) {
      const int j = d_neighbors(i,jj) & NEIGHMASK;
      if (!(mask(j) & groupbit_local)) continue;

      const int tj = type(j);
      const int cj = local_to_cidx(j);
      const bool j_is_c = (tj == carbon_type && cj >= 0);
      const bool j_is_h = (tj == hydrogen_type);

      double dx = x(j,0) - x(i,0);
      double dy = x(j,1) - x(i,1);
      double dz = x(j,2) - x(i,2);

      if (periodic0) {
        if (dx > hx) dx -= xprd;
        else if (dx < -hx) dx += xprd;
      }
      if (periodic1) {
        if (dy > hy) dy -= yprd;
        else if (dy < -hy) dy += yprd;
      }
      if (periodic2) {
        if (dz > hz) dz -= zprd;
        else if (dz < -hz) dz += zprd;
      }

      const double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq <= 0.0 || rsq >= cf_cut2) continue;

      if (i_is_c && j_is_c && cf_enabled) {
        bool connected = false;
        const int ncc = cc_count(ci);
        for (int k = 0; k < ncc; ++k) {
          if (cc_partner(ci,k) == cj) {
            connected = true;
            break;
          }
        }
        if (connected) continue;

        const double r = sqrt(rsq);
        if (r > cf_ref) continue;

        double stretch = (r - cf_ref) / cf_ref;
        double val = 1.0;
        int pp = cf_power;
        while (pp > 0) {
          if (pp & 1) val *= stretch;
          stretch *= stretch;
          pp >>= 1;
        }
        Kokkos::atomic_add(&sums(2), val);
        Kokkos::atomic_add(&sums(3), 1.0);
      }
    }
  });

  auto hsums = Kokkos::create_mirror_view(sums);
  Kokkos::deep_copy(hsums, sums);

  double local[4] = {hsums(0), hsums(1), hsums(2), hsums(3)};
  double global[4] = {0.0, 0.0, 0.0, 0.0};
  MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, world);

  ccbb = 0.0;
  chbb = 0.0;
  ccbf = (global[2] > 0.0) ? std::pow(global[2], 1.0 / static_cast<double>(cc_form_.power)) : 0.0;
  last_cc_form_pair_count_ = static_cast<std::size_t>(global[3]);

  cvhd_timer_add(cvhd_t_fused_cv_, cvhd_n_fused_cv_, t_timer);
}

void FixCvhdGlobalDistortionKokkos::apply_fused_forces_kokkos(double ccbb, double chbb, double ccbf,
                                                              double dVdccbb, double dVdchbb,
                                                              double dVdccbf)
{
  const bool do_cc = cc_break_.enabled && ccbb > 0.0 && dVdccbb != 0.0;
  const bool do_ch = ch_break_.enabled && chbb > 0.0 && dVdchbb != 0.0;
  const bool do_cf = cc_form_.enabled && ccbf > 0.0 && dVdccbf != 0.0;

  if (!do_cc && !do_ch && !do_cf) return;

  const double t_timer = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;

  auto *k_list = k_list_;
  if (!k_list) {
    error->all(FLERR,
               "cvhd/global/distortion/kk fused backend did not receive NeighListKokkos; "
               "check that init() calls request->set_kokkos_device(1) and neighbor log shows kokkos_device/bin: kk/device");
  }

  if (!atomKK) atomKK = static_cast<AtomKokkos *>(atom);

  ensure_fused_cache_current();

  atomKK->sync(Device, X_MASK | F_MASK | TYPE_MASK | TAG_MASK | MASK_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();

  auto d_ilist = k_list->d_ilist;
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  const int inum = list_->inum;
  const int nlocal = atom->nlocal;

  const auto local_to_cidx = fused_dev_.local_to_cidx;
  const auto cc_count = fused_dev_.cc_partner_count;
  const auto cc_partner = fused_dev_.cc_partner_cidx;
  const auto ch_count = fused_dev_.ch_partner_count;
  const auto ch_tag = fused_dev_.ch_partner_tag;

  const int carbon_type = carbon_type_;
  const int hydrogen_type = hydrogen_type_;
  const int groupbit_local = groupbit;

  const double cc_ref = cc_break_.ref;
  const double ch_ref = ch_break_.ref;
  const double cf_ref = cc_form_.ref;
  const int cc_power = cc_break_.power;
  const int ch_power = ch_break_.power;
  const int cf_power = cc_form_.power;
  // Reuse cc_form_candidate_rmax as the local force cutoff for all fused-force
  // components.  Beyond this region the pair is already well past the TS/local
  // kernel region and contributes zero bias force.
  const double cand2 = (cc_form_candidate_rmax_ > 0.0)
                     ? cc_form_candidate_rmax_ * cc_form_candidate_rmax_
                     : -1.0;

  const double cc_pref = do_cc ? std::pow(ccbb, 1.0 - static_cast<double>(cc_power)) : 0.0;
  const double ch_pref = do_ch ? std::pow(chbb, 1.0 - static_cast<double>(ch_power)) : 0.0;
  const double cf_pref = do_cf ? std::pow(ccbf, 1.0 - static_cast<double>(cf_power)) : 0.0;

  const int periodic0 = domain->periodicity[0];
  const int periodic1 = domain->periodicity[1];
  const int periodic2 = domain->periodicity[2];
  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double hx = 0.5 * xprd;
  const double hy = 0.5 * yprd;
  const double hz = 0.5 * zprd;

  Kokkos::parallel_for("cvhd_fused_neighbor_force",
                       Kokkos::RangePolicy<DeviceType>(0, inum),
                       KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    if (!(mask(i) & groupbit_local)) return;

    const int ti = type(i);
    const int ci = local_to_cidx(i);
    const bool i_is_c = (ti == carbon_type && ci >= 0);
    const bool i_is_h = (ti == hydrogen_type);

    const int jnum = d_numneigh(i);
    for (int jj = 0; jj < jnum; ++jj) {
      const int j = d_neighbors(i,jj) & NEIGHMASK;
      if (!(mask(j) & groupbit_local)) continue;

      const int tj = type(j);
      const int cj = local_to_cidx(j);
      const bool j_is_c = (tj == carbon_type && cj >= 0);
      const bool j_is_h = (tj == hydrogen_type);

      double dx = x(j,0) - x(i,0);
      double dy = x(j,1) - x(i,1);
      double dz = x(j,2) - x(i,2);

      if (periodic0) {
        if (dx > hx) dx -= xprd;
        else if (dx < -hx) dx += xprd;
      }
      if (periodic1) {
        if (dy > hy) dy -= yprd;
        else if (dy < -hy) dy += yprd;
      }
      if (periodic2) {
        if (dz > hz) dz -= zprd;
        else if (dz < -hz) dz += zprd;
      }

      const double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq <= 0.0 || (cand2 > 0.0 && rsq >= cand2)) continue;
      const double r = sqrt(rsq);

      double pref = 0.0;
      double dV = 0.0;
      double ref = 0.0;
      int power = 0;
      bool active = false;
      bool formation = false;

      if (i_is_c && j_is_c) {
        bool connected = false;
        const int ncc = cc_count(ci);
        for (int k = 0; k < ncc; ++k) {
          if (cc_partner(ci,k) == cj) {
            connected = true;
            break;
          }
        }

        if (connected) {
          if (do_cc && r >= cc_ref) {
            active = true;
            formation = false;
            pref = cc_pref;
            dV = dVdccbb;
            ref = cc_ref;
            power = cc_power;
          }
        } else {
          if (do_cf) {
            if (cand2 > 0.0 && rsq > cand2) continue;
            if (r > cf_ref) continue;
            active = true;
            formation = true;
            pref = cf_pref;
            dV = dVdccbf;
            ref = cf_ref;
            power = cf_power;
          }
        }
      } else if ((i_is_c && j_is_h) || (i_is_h && j_is_c)) {
        if (!do_ch) continue;
        const int cidx = i_is_c ? ci : cj;
        const tagint htag = i_is_h ? tag(i) : tag(j);
        bool connected_ch = false;
        const int nch = ch_count(cidx);
        for (int k = 0; k < nch; ++k) {
          if (ch_tag(cidx,k) == htag) {
            connected_ch = true;
            break;
          }
        }
        if (connected_ch && r >= ch_ref) {
          active = true;
          formation = false;
          pref = ch_pref;
          dV = dVdchbb;
          ref = ch_ref;
          power = ch_power;
        }
      }

      if (!active) continue;

      const double stretch = (r - ref) / ref;
      if (stretch == 0.0) continue;

      double stretch_pow = 1.0;
      for (int q = 0; q < power - 1; ++q) stretch_pow *= stretch;

      const double pair_deriv_pref = pref * stretch_pow / ref / r;
      const double fx = dV * pair_deriv_pref * dx;
      const double fy = dV * pair_deriv_pref * dy;
      const double fz = dV * pair_deriv_pref * dz;

      if (i >= 0 && i < nlocal) {
        Kokkos::atomic_add(&f(i,0), fx);
        Kokkos::atomic_add(&f(i,1), fy);
        Kokkos::atomic_add(&f(i,2), fz);
      }

      if (j >= 0 && j < nlocal) {
        Kokkos::atomic_add(&f(j,0), -fx);
        Kokkos::atomic_add(&f(j,1), -fy);
        Kokkos::atomic_add(&f(j,2), -fz);
      }
    }
  });

  if (cvhd_timer_enable_) Kokkos::fence();
  atomKK->modified(Device, F_MASK);
  cvhd_timer_add(cvhd_t_fused_force_, cvhd_n_fused_force_, t_timer);
}

double FixCvhdGlobalDistortionKokkos::compute_breaking_value_kokkos(const TermConfig &term,
                                                                    DevicePairCache &cache)
{
  const double t_timer = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
  if (!term.enabled) return 0.0;

  // Caches are updated once per neighbor/local-index change by
  // ensure_pair_caches_current(), not on every CV evaluation.
  if (cache.n == 0) return 0.0;

  AtomKokkos *atomKK = dynamic_cast<AtomKokkos *>(atom);
  if (!atomKK) {
    error->all(FLERR, "cvhd/global/distortion/kk requires AtomKokkos but atom is not AtomKokkos");
  }

  // Use device coordinates directly.  Do not access atom->x in this reduction.
  atomKK->sync(Device, X_MASK);
  auto x = atomKK->k_x.view<DeviceType>();

  const auto d_i = cache.i;
  const auto d_j = cache.j;
  const auto d_ref = cache.ref;
  const auto d_count = cache.count;
  const int n = cache.n;

  const double term_ref = term.ref;
  const int power = term.power;

  const int periodic0 = domain->periodicity[0];
  const int periodic1 = domain->periodicity[1];
  const int periodic2 = domain->periodicity[2];
  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double hx = 0.5 * xprd;
  const double hy = 0.5 * yprd;
  const double hz = 0.5 * zprd;

  double local_pairsum = 0.0;

  Kokkos::parallel_reduce("cvhd_breaking_value",
                          Kokkos::RangePolicy<DeviceType>(0, n),
                          KOKKOS_LAMBDA(const int p, double &sum) {
    if (!d_count(p)) return;

    const int i = d_i(p);
    const int j = d_j(p);
    const double ref = d_ref(p);

    double dx = x(j,0) - x(i,0);
    double dy = x(j,1) - x(i,1);
    double dz = x(j,2) - x(i,2);

    // Orthogonal minimum image.  V3k keeps the same limitation as V3c.
    if (periodic0) {
      if (dx > hx) dx -= xprd;
      else if (dx < -hx) dx += xprd;
    }
    if (periodic1) {
      if (dy > hy) dy -= yprd;
      else if (dy < -hy) dy += yprd;
    }
    if (periodic2) {
      if (dz > hz) dz -= zprd;
      else if (dz < -hz) dz += zprd;
    }

    const double r = sqrt(dx*dx + dy*dy + dz*dz);
    if (r < ref) return;

    double stretch = (r - ref) / term_ref;
    double val = 1.0;
    int pp = power;
    while (pp > 0) {
      if (pp & 1) val *= stretch;
      stretch *= stretch;
      pp >>= 1;
    }

    sum += val;
  }, local_pairsum);

  double global_pairsum = 0.0;
  MPI_Allreduce(&local_pairsum, &global_pairsum, 1, MPI_DOUBLE, MPI_SUM, world);

  cvhd_timer_add(cvhd_t_kk_reduce_, cvhd_n_kk_reduce_, t_timer);

  if (global_pairsum <= 0.0) return 0.0;
  return std::pow(global_pairsum, 1.0 / static_cast<double>(power));
}

void FixCvhdGlobalDistortionKokkos::apply_breaking_forces_kokkos(const TermConfig &term,
                                                                 double raw_value,
                                                                 double dV_draw,
                                                                 DevicePairCache &cache)
{
  const double t_timer = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
  if (!term.enabled) return;
  if (term.do_form) return;
  if (raw_value <= 0.0) return;
  if (dV_draw == 0.0) return;
  if (cache.n == 0) return;

  AtomKokkos *atomKK = dynamic_cast<AtomKokkos *>(atom);
  if (!atomKK) {
    error->all(FLERR, "cvhd/global/distortion/kk requires AtomKokkos but atom is not AtomKokkos");
  }

  atomKK->sync(Device, X_MASK | F_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();

  const auto d_i = cache.i;
  const auto d_j = cache.j;
  const auto d_ref = cache.ref;
  const int n = cache.n;

  const int nlocal = atom->nlocal;
  const double term_ref = term.ref;
  const int power = term.power;
  const double term_prefactor = std::pow(raw_value, 1.0 - static_cast<double>(power));
  const double dV = dV_draw;

  const int periodic0 = domain->periodicity[0];
  const int periodic1 = domain->periodicity[1];
  const int periodic2 = domain->periodicity[2];
  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double hx = 0.5 * xprd;
  const double hy = 0.5 * yprd;
  const double hz = 0.5 * zprd;

  Kokkos::parallel_for("cvhd_breaking_force",
                       Kokkos::RangePolicy<DeviceType>(0, n),
                       KOKKOS_LAMBDA(const int p) {
    const int i = d_i(p);
    const int j = d_j(p);
    const double ref = d_ref(p);

    const bool i_owned = (i >= 0 && i < nlocal);
    const bool j_owned = (j >= 0 && j < nlocal);
    if (!i_owned && !j_owned) return;

    double dx = x(j,0) - x(i,0);
    double dy = x(j,1) - x(i,1);
    double dz = x(j,2) - x(i,2);

    if (periodic0) {
      if (dx > hx) dx -= xprd;
      else if (dx < -hx) dx += xprd;
    }
    if (periodic1) {
      if (dy > hy) dy -= yprd;
      else if (dy < -hy) dy += yprd;
    }
    if (periodic2) {
      if (dz > hz) dz -= zprd;
      else if (dz < -hz) dz += zprd;
    }

    const double rsq = dx*dx + dy*dy + dz*dz;
    if (rsq <= 0.0) return;

    const double r = sqrt(rsq);
    if (r < ref) return;

    const double stretch0 = (r - ref) / term_ref;
    if (stretch0 == 0.0) return;

    double stretch_pow = 1.0;
    for (int q = 0; q < power - 1; ++q) stretch_pow *= stretch0;

    const double pair_deriv_pref = term_prefactor * stretch_pow / term_ref / r;

    const double fx = dV * pair_deriv_pref * dx;
    const double fy = dV * pair_deriv_pref * dy;
    const double fz = dV * pair_deriv_pref * dz;

    if (i_owned) {
      Kokkos::atomic_add(&f(i,0), fx);
      Kokkos::atomic_add(&f(i,1), fy);
      Kokkos::atomic_add(&f(i,2), fz);
    }

    if (j_owned) {
      Kokkos::atomic_add(&f(j,0), -fx);
      Kokkos::atomic_add(&f(j,1), -fy);
      Kokkos::atomic_add(&f(j,2), -fz);
    }
  });

  if (cvhd_timer_enable_) Kokkos::fence();
  atomKK->modified(Device, F_MASK);
  cvhd_timer_add(cvhd_t_kk_force_, cvhd_n_kk_force_, t_timer);
}
#endif

void FixCvhdGlobalDistortionKokkos::backend_compute_raw_cvs(double &ccbb,
                                                            double &chbb,
                                                            double &ccbf)
{
#ifdef LMP_KOKKOS
  if (kk_fused_neighbor_enable_) {
    ensure_pair_caches_current();

    ccbb = cc_break_.enabled ? compute_breaking_value_kokkos(cc_break_, cc_pairs_dev_) : 0.0;
    chbb = ch_break_.enabled ? compute_breaking_value_kokkos(ch_break_, ch_pairs_dev_) : 0.0;

    double dummy_cc = 0.0;
    double dummy_ch = 0.0;
    compute_fused_raw_cvs_kokkos(dummy_cc, dummy_ch, ccbf);
  } else {
    ensure_pair_caches_current();

    ccbb = cc_break_.enabled ? compute_breaking_value_kokkos(cc_break_, cc_pairs_dev_) : 0.0;
    chbb = ch_break_.enabled ? compute_breaking_value_kokkos(ch_break_, ch_pairs_dev_) : 0.0;

    if (cc_form_.enabled) {
      const double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
      ccbf = compute_formation_value_from_neighbor_list();
      cvhd_timer_add(cvhd_t_form_cpu_, cvhd_n_form_cpu_, t0);
    } else {
      ccbf = 0.0;
    }
  }
#else
  FixCvhdGlobalDistortion::backend_compute_raw_cvs(ccbb, chbb, ccbf);
#endif
}

void FixCvhdGlobalDistortionKokkos::backend_apply_bias_forces()
{
#ifdef LMP_KOKKOS
  if (kk_fused_neighbor_enable_) {
    apply_fused_forces_kokkos(values_[0], values_[1], values_[2],
                              values_[10], values_[11], values_[17]);
  } else {
    // Kokkos breaking force.  Formation force remains CPU fallback.
    // compute_all() should have refreshed caches already; this is a cheap guard
    // for unusual call order.
    ensure_pair_caches_current();

    apply_breaking_forces_kokkos(cc_break_, values_[0], values_[10], cc_pairs_dev_);
    apply_breaking_forces_kokkos(ch_break_, values_[1], values_[11], ch_pairs_dev_);
    if (opes_use_cv2_ && values_[17] != 0.0 && values_[2] > 0.0) {
      const double t0 = cvhd_timer_enable_ ? cvhd_timer_now() : 0.0;
      apply_formation_forces(values_[2], values_[17]);
      cvhd_timer_add(cvhd_t_cpu_form_force_, cvhd_n_cpu_form_force_, t0);
    }
  }
#else
  FixCvhdGlobalDistortion::backend_apply_bias_forces();
#endif
}
