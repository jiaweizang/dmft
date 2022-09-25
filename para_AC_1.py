"""Step 1 for the analytical continuation of the self energy of paramagnetic state."""
from math import *
from triqs.gf import *
from triqs.operators import *
from h5 import HDFArchive
import numpy as np
from triqs_maxent import *

t = 0.25
phi = pi / 6

for Ut in [8]:
    for n_dope in [1]:
        filename = "para_data/fit_tri_phi=%s_n=%s.h5" % (np.around(phi / pi, 3), n_dope)
        print(f"n={n_dope},phi={phi}")
        for Temp in [0.1]:
            print("T=", Temp)
            if Temp <= 0.9:
                err = 5 * 1e-3
            else:
                err = 10 * 1e-3
            with HDFArchive(filename) as A:
                Sigma_iw = A['Sigma_iw-T=%st_U=%st' % (Temp, Ut)]
                S_fit = A['S_fit_iw-T=%st_U=%st' % (Temp, Ut)]
                mu = A['mu-T=%st_U=%st' % (Temp, Ut)]
            S_inf = S_fit['up'](1000).real[0][0]

            # run continuation

            # Initialize SigmaContinuator
            isc = InversionSigmaContinuator(S_fit['up'], S_inf)

            gaux_iw = isc.Gaux_iw
            # tm = TauMaxEnt()
            tm = TauMaxEnt(cost_function='bryan', probability='normal')
            tm.set_G_iw(gaux_iw)
            tm.set_error(err)
            tm.omega = HyperbolicOmegaMesh(omega_min=-15, omega_max=15, n_points=500)
            tm.alpha_mesh = LogAlphaMesh(alpha_min=1e-6, alpha_max=1e0, n_points=40)
            res = tm.run()
            # save results and the SigmaContinuator to h5-file
            with HDFArchive("para_data/continue_phi=%s_n=%s_Ut=%s.h5" % (np.around(phi / pi, 3), n_dope, Ut),
                            'a') as ar:
                ar['maxE_midres-T=%st_U=%st' % (Temp, Ut)] = res.data
                ar['isc-T=%st_U=%st' % (Temp, Ut)] = isc
                ar['mu-T=%st_U=%st' % (Temp, Ut)] = mu
