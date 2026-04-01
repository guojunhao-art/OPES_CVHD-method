/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2016 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "function/Function.h"
#include "core/ActionRegister.h"

#include <cmath>


namespace PLMD {
namespace function {
namespace cvhdm {

//+PLUMEDOC FUNCTION CVHDM
/*
Convert a (set of) CV(s) into the CV of the CVHDM method.

In the simplest use case (one argument), the CV is calculated as follows for
values between 0 and CUTOFF:
\f[
C=\frac{1}{2} \left ( 1 - \cos left ( \pi \frac{x}{x_{cut}} \right ) \right )
\f]
Effectively, any CV is projected on an interval between 0 and 1.

If multiple arguments are supplied, these are first combined as follows, using
the power supplied through the POWER keyword.
\f[
x=\left ( \sum_{i} x_{i}^{p} \right )^{1/p}
\f]

The coefficients c, the parameters a and the powers p are provided as vectors.

Typically, this function is used with the BONDDISTORTION or GLOBALDISTORTION
CVs, as an argument for a metadynamics or VES bias in CVHDM mode.

Notice that CVHDM is not able to predict which will be periodic domain
of the computed value automatically. The user is thus forced to specify it
explicitly. Use PERIODIC=NO if the resulting variable is not periodic,
and PERIODIC=A,B where A and B are the two boundaries if the resulting variable
is periodic.

\par Examples

The following provide an example to bias two different bond types (say, C–H and C–C) using
a single clamped CVHDM CV.

\plumedfile
CVHDM ...
LABEL=cv
ARG=cc,ch
P=8
CUTOFF=0.5
PERIODIC=NO
... CVHDM
\endplumedfile

*/
//+ENDPLUMEDOC


class CVHDM :
  public Function
{
  std::vector<double> cutoff_;
  double power_;
public:
  explicit CVHDM(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};


PLUMED_REGISTER_ACTION(CVHDM,"CVHDM")

void CVHDM::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG"); keys.use("PERIODIC");
  keys.add("compulsory","CUTOFF","1.0","the cutoff distance");
  keys.add("compulsory","P","1.0","the powers to which you are raising each of the arguments in your function");
}

CVHDM::CVHDM(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{
  power_ = 1.0;
  parseVector("CUTOFF",cutoff_);
  parse("P",power_);

  addValueWithDerivatives();
  checkRead();

  log.printf("  with cutoff:");
  for(unsigned i=0; i<cutoff_.size(); ++i) log.printf(" %f",cutoff_[i]);
  log.printf("\n");
  log.printf("  with power:");
  log.printf(" %f",power_);
  log.printf("\n");

  if( cutoff_.size()!=getNumberOfArguments() ) error("number of arguments does not match number of CUTOFF parameters");
}

void CVHDM::calculate() {
  const double pi = 3.141592653589793;
  double combined_norm = 0.0;
  double value, prefactor, projected_argument, pnorm;

  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    const double cv = getArgument(i);
    const double normalized_cv = cv/cutoff_[i];
    combined_norm += std::pow(normalized_cv, power_);
  }

  pnorm = std::pow(combined_norm,2.0/power_);
  projected_argument = pnorm;
  if(projected_argument < 1.0 && projected_argument > 0.0) {
    value = 0.5*(1.0-std::cos(pi*projected_argument));
    prefactor = std::pow(combined_norm, 2.0/power_-1.0)*pi*std::sin(pi*projected_argument);
  } else if (projected_argument >= 1.0) {
    value = 1.0;
    prefactor = 0.0;
  } else {
    value = 0.0;
    prefactor = 0.0;
  }

  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    const double cv = getArgument(i);
    const double normalized_cv = cv/cutoff_[i];
    setDerivative(i,prefactor*std::pow(normalized_cv,power_-1.0)/cutoff_[i]);
  }
  setValue(value);
}

}
}
}
