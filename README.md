# OPES_CVHD-method
Source code and caculation files of OPES_CVHD \<br>

OPES_CVHD is an enhanced sampling framework that integrates On-the-fly Probability Enhanced Sampling (OPES) with Collective Variable–Driven Hyperdynamics (CVHD) to enable efficient and kinetically reliable rare-event simulations across complex free-energy landscapes with widely separated energy barriers. \<br>

This repository provides a reference implementation of OPES_CVHD developed as an extension of the PLUMED code base and used in conjunction with LAMMPS. \<br>

##1. Method Overview
OPES_CVHD combines: \<br>
1. the adaptive, probability-based bias construction of OPES, and \<br>
2. the event-based acceleration and exact time reconstruction of CVHD. \<br>

Key features include: \<br>
1. adaptive control of the OPES bias bound through a dynamically updated barrier parameter 𝐸 \<br>
2. a damping strategy to regulate bias growth near transition states, \<br>
3. strict enforcement of the no-bias-at-transition-state condition required for hyperdynamics, compatibility only for CVHD type collective variables. \<br>

The method is particularly suitable for reactive systems involving multiple competing reactions with widely separated barriers, such as hydrocarbon pyrolysis and early-stage aromatic and polycyclic aromatic hydrocarbon formation. \<br>

##2. Repository Structure
OPES_CVHD/ \<br>
├── src/ \<br>
│   ├── CVHDM.cpp \<br>
│   ├── GlobalDistortion.cpp \<br>
│   └── OPESmetadCV.cpp \<br>
├── examples/ \<br>
│   ├── styryl/ \<br>
│   ├── cu_vacancy/ \<br>
│   └── mch_pyrolysis/ \<br>
└── README.md \<br>

##3. Source Code Provenance
The OPES_CVHD implementation builds upon existing open-source PLUMED modules. The origin of each source file in src/ is detailed below for transparency and licensing clarity. \<br>

GlobalDistortion.cpp \<br>
Origin: Copied directly from [https://github.com/kbal/plumed2](https://github.com/kbal/plumed2/tree/cvhd) \<br>
Description: Implements the global distortion collective variable used to detect bond-breaking and bond-forming events, please refer https://medialibrary.uantwerpen.be/oldcontent/personalpage31526/files/manual_cvhd.pdf. \<br>
Modifications:  \<br>
None \<br>

CVHDM.cpp \<br>
Origin: Modified from the CVHD implementation in [https://github.com/kbal/plumed2](https://github.com/kbal/plumed2/tree/cvhd) \<br>
Description: Implements collective-variable-driven hyperdynamics logic. \<br>
Modifications: \<br>
Modified to support the simultaneous input of multiple cluster variables. \<br>

OPESmetadCV.cpp \<br>
Origin: Modified from the OPES metadynamics module in https://github.com/plumed/plumed2/tree/master \<br>
Description: Implements OPES bias construction based on marginal probability distributions. \<br>
Modifications: \<br>
Introduction of a dynamically updated barrier parameter 𝐸 \<br>
Damping strategy for bias growth near transition states \<br>
Compatibility with hyperdynamics time reconstruction \<br>

##4. Licensing
This repository is distributed under the GNU Lesser General Public License v3.0 (LGPL-3.0), consistent with the licenses of the original PLUMED source code from which it is derived. \<br>
Original PLUMED and CVHD code: \<br>
https://github.com/plumed/plumed2 \<br>
https://github.com/kbal/plumed2 \<br>
License: LGPL-3.0 \<br>
Users are free to use, modify, and redistribute this code under the terms of the LGPL-3.0 license. \<br>
Please ensure that any derivative works comply with the LGPL requirements. \<br>

##5. Requirements
LAMMPS (tested with version 2Aug2023) \<br>
PLUMED (patched with OPES_CVHD source files) \<br>
CMake-compatible C++ compiler \<br>

##6. Installation
1. Clone this repository: \<br>
2. Load three files mentioned above in your plumed.dat. \<br>
3. If the original CVHD is needed, refer to [https://github.com/kbal/plumed2](https://github.com/kbal/plumed2/tree/cvhd) \<br>

##7. Reproducibility
All simulation results reported in the accompanying manuscript were generated using the input files provided in the examples/ directory. Simulation parameters, collective variable definitions, and biasing protocols follow the descriptions in the manuscript. \<br>

##8. Contact
For questions or bug reports, please contact: \<br>
Junhao Guo — 3018001262@tju.edu.cn \<br>
Yutong Wang — wangyutong@syuct.edu.cn \<br>
Prof. Guozhu Liu — gliu@tju.edu.cn \<br>

##Final note
This repository is intended as a reference implementation accompanying the JCTC manuscript. \<br>
The code is provided to support method transparency and reproducibility, rather than as a fully packaged production release. \<br>
