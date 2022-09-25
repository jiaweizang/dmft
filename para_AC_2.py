"""Step 2 for the analytical continuation of the self energy of paramagnetic state."""

from math import *
from triqs.gf import *
from triqs.operators import *
from h5 import HDFArchive
import triqs.utility.mpi as mpi
import numpy as np
import time
from h5 import *
from triqs_maxent import *

t = 0.25

for Ut in [8]:
    for phi in [pi / 6]:
        for n_dope in [1]:
            filename = "para_data/continue_phi=%s_n=%s_Ut=%s.h5" % (np.around(phi / pi, 3), n_dope, Ut)

            for Temp in [0.05]:  # Templist:
                res = 0
                isc = 0
                if mpi.is_master_node():
                    tim = time.time()

                    A = HDFArchive(filename, 'a')
                    res = A['maxE_midres-T=%st_U=%st' % (Temp, Ut)]
                    isc = A['isc-T=%st_U=%st' % (Temp, Ut)]
                    del A
                res = mpi.bcast(res)
                isc = mpi.bcast(isc)
                Aaux_w = res.analyzer_results['LineFitAnalyzer']['A_out']
                w = res.omega

                boundary = min(20 * Temp * t, 10)
                isc.set_Gaux_w_from_Aaux_w(Aaux_w, w, np_interp_A=10000, np_omega=5000, w_min=-boundary, w_max=boundary)
                if mpi.is_master_node():
                    with HDFArchive(filename) as A:
                        A['MaxE_S_iw-T=%st_U=%st_R' % (Temp, Ut)] = isc.S_w
                        tim2 = time.time()
                        print(Ut, phi, n_dope, Temp, "use:", tim2 - tim)
