"""Step 2 for the analytical continuation of the self energy of magnetic state."""

from math import *
from triqs.gf import *
from triqs.operators import *
from h5 import HDFArchive
import triqs.utility.mpi as mpi
import numpy as np
import time
from h5 import *
from triqs_maxent import *

err = 5 * 1e-3
phi = pi / 2
t = 0.25
density_required = 1
for Ut in [8]:
    for density_required in [1]:
        filename = "mag_data/mag_continue_phi=%s_n=%s.h5" % (np.around(phi / pi, 3), density_required)

        for Temp in [0.05]:  # Tlist:
            for block in ['up', 'down']:
                res = 0
                isc = 0
                if mpi.is_master_node():
                    tim = time.time()

                    A = HDFArchive(filename, 'a')
                    res = A['maxE_%s_midres-T=%st_U=%st' % (block, Temp, Ut)]
                    isc = A['isc_%s-T=%st_U=%st' % (block, Temp, Ut)]
                    del A
                res = mpi.bcast(res)
                isc = mpi.bcast(isc)
                Aaux_w = res.analyzer_results['LineFitAnalyzer']['A_out']
                w = res.omega
                boundary = min(20 * Temp * t, 10)
                isc.set_Gaux_w_from_Aaux_w(Aaux_w, w, np_interp_A=10000, np_omega=5000, w_min=-boundary, w_max=boundary)
                if mpi.is_master_node():
                    with HDFArchive(filename) as A:
                        A['MaxE_S_iw_%s-T=%st_U=%st_R' % (block, Temp, Ut)] = isc.S_w
                    tim2 = time.time()
                    print(Temp, "use:", tim2 - tim)



