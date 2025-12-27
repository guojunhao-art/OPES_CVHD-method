# source code for gif making of CVHD and OPES_CVHD method
The Free energy used for illustration is Modified Wolfe-Quapp Potential reported by Parrinello et al. (J. Chem. Theory Comput. 2022, 18, 6500−6509):  
$U(x,y) = 2(x^4+y^4-2x^2-4y^2+2xy+0.8x+0.1y+9.28)$  
The CV used for enhanced sampling is caculated as:  
$CV=x$  
In order to align with the CVHD framework, the value of CV is shifted to satisfy the value of CV under transition state is 1.
