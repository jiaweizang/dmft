# DMFT code

This repository contains code for the dynamical mean field theory (DMFT) calculation.

This code was used to obtain the results in the following paper: 

Dynamical Mean-Field Theory of Moiré Bilayer Transition Metal Dichalcogenides: Phase Diagram, Resistivity, and Quantum Criticality
([Phys. Rev. X 12, 021064](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.021064))

Code author: Jiawei Zang (jiaweizang)



## Description
We use the single-site DMFT with the continuous-time hybridization expansion solver
as implemented in the [TRIQS](https://triqs.github.io/triqs/latest/) software library to perform the calculation. The 
results are obtained first on the imaginary axis, and we do an analytical continuation 
to get the results on the real frequency axis.

Paramagnetic state:
* `paramagnetic.py` runs the DMFT calculation for the paramagnetic state.
* `para_tail_fitting.ipynb` performs a tail fitting of the self energy.
* `para_AC_1.py` does the first step of the analytical continuation for the self energy, after it has been tail fitted. 
* `para_AC_2.py` does the second step of the analytical continuation for the self energy.

Magnetic state:
* `magnetic.py` runs the DMFT calculation for the magnetic state.
* `mag_tail_fitting.ipynb` performs a tail fitting of the self energy.
* `mag_AC_1.py` does the first step of the analytical continuation for the self energy, after it has been tail fitted. 
* `mag_AC_2.py` does the second step of the analytical continuation for the self energy. 

note: except for `para_AC_1.py` and `mag_AC_1.py`, all the other files utilize MPI library, thus they can be submitted to the cluster to significantly reduce the running time. `job.sh` provides
an example for the submission script.

After we get the self energy on the real frequency, we use Eq.2 and 8 in the [paper](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.021064)
 to calculate the resistivity.



### Required Packages

* [TRIQS](https://triqs.github.io/triqs/latest/install.html)
* [Maxent](https://triqs.github.io/maxent/latest/index.html)
