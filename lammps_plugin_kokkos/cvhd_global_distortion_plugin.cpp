/* ----------------------------------------------------------------------
   Plugin registration for fix cvhd/global/distortion and
   fix cvhd/global/distortion/kk

   Build these files into a DSO and load it in LAMMPS via:

     plugin load /path/to/cvhdplugin.so

------------------------------------------------------------------------- */

#include "lammpsplugin.h"
#include "version.h"
#include "fix_cvhd_global_distortion.h"
#include "fix_cvhd_global_distortion_kokkos.h"

using namespace LAMMPS_NS;

static Fix *cvhd_global_distortion_creator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixCvhdGlobalDistortion(lmp, argc, argv);
}

static Fix *cvhd_global_distortion_kk_creator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixCvhdGlobalDistortionKokkos(lmp, argc, argv);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  lammpsplugin_t plugin = {};
  plugin.version = LAMMPS_VERSION;
  plugin.style   = "fix";
  plugin.name    = "cvhd/global/distortion";
  plugin.info    = "CVHD global-distortion V3k-hybrid-perf OPES_CVHD fix";
  plugin.author  = "Junhao Guo and ChatGPT";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &cvhd_global_distortion_creator;
  plugin.handle  = handle;

  (*register_plugin)(&plugin, lmp);

  lammpsplugin_t plugin_kk = {};
  plugin_kk.version = LAMMPS_VERSION;
  plugin_kk.style   = "fix";
  plugin_kk.name    = "cvhd/global/distortion/kk";
  plugin_kk.info    = "CVHD global-distortion V3k /kk hybrid CV backend with fused local force and tag-map performance fixes";
  plugin_kk.author  = "Junhao Guo and ChatGPT";
  plugin_kk.creator.v2 = (lammpsplugin_factory2 *) &cvhd_global_distortion_kk_creator;
  plugin_kk.handle  = handle;

  (*register_plugin)(&plugin_kk, lmp);
}
