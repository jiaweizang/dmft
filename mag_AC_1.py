"""Step 1 for the analytical continuation of the self energy of magnetic state."""

from math import *
from triqs.gf import *
from triqs.operators import *
from h5 import HDFArchive
import numpy as np
from triqs_maxent import *

err = 5 * 1e-3
t = 0.25

Ut = 8
philist = np.arange(9, 12, 0.5) / 24 * pi
for phi in [pi / 2]:
    for density_required in [1]:
        print(phi, density_required)
        filename = "mag_data/fit_phase_phi=%s_n=%s.h5" % (np.around(phi / pi, 3), density_required)

        for Temp in [0.05]:
            print(Ut, Temp)
            with HDFArchive(filename) as A:
                Sigma_iw = A['Sigma_iw-T=%st_U=%st' % (Temp, Ut)]
                S_fit = A['S_fit_iw-T=%st_U=%st' % (Temp, Ut)]
                mu = A['mu-T=%st_U=%st' % (Temp, Ut)]
            for block in ['up', 'down']:
                S_inf = S_fit[block](1000).real[0][0]

                # run continuation

                # Initialize SigmaContinuator
                isc = InversionSigmaContinuator(S_fit[block], S_inf)
                gaux_iw = isc.Gaux_iw
                # tm = TauMaxEnt()
                tm = TauMaxEnt(cost_function='bryan', probability='normal')
                tm.set_G_iw(gaux_iw)
                tm.set_error(err)
                tm.omega = HyperbolicOmegaMesh(omega_min=-15, omega_max=15, n_points=1000)
                # tm.omega = HyperbolicOmegaMesh(omega_min=-15, omega_max=15, n_points=200)
                tm.alpha_mesh = LogAlphaMesh(alpha_min=1e-6, alpha_max=1e0, n_points=30)
                # tm.alpha_mesh = LogAlphaMesh(alpha_min=1e-5, alpha_max=1e1, n_points=30)
                res = tm.run()
                # save results and the SigmaContinuator to h5-file
                with HDFArchive("mag_data/mag_continue_phi=%s_n=%s.h5" % (np.around(phi / pi, 3), density_required),
                                'a') as ar:
                    ar['maxE_%s_midres-T=%st_U=%st' % (block, Temp, Ut)] = res.data
                    ar['isc_%s-T=%st_U=%st' % (block, Temp, Ut)] = isc
                    ar['mu-T=%st_U=%st' % (Temp, Ut)] = mu
                    ar['S_fit_iw-T=%st_U=%st' % (Temp, Ut)] = S_fit
