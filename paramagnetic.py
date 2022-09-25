"""
DMFT calculation for the paramagnetic state of the Hubbard model
It uses Triqs version 3.0
"""

from math import *
from triqs.gf import *
from triqs.operators import *
from triqs_cthyb import Solver
from h5 import HDFArchive
import triqs.utility.mpi as mpi
import numpy as np
from triqs.atom_diag import trace_rho_op

# Parameters of the Hubbard model
t = 0.25
philist = [0, pi / 12, pi / 6]
Templist = [0.5]
Utlist = np.arange(14, 0.9, -1)
n_bins = 500  # bins used for calcualtingthe density of states


# function to calculate the dispersion of the Hubbard model with hopping t, phase phi and spin +/-
def dispersion(kxlist, kylist, t=1, phi=0, spin=1):
    phi = spin * phi
    return -2 * t * (np.cos(kxlist + phi) + np.cos(-kxlist / 2 + sqrt(3) / 2 * kylist + phi) + np.cos(
        -kxlist / 2 - sqrt(3) / 2 * kylist + phi))


# build the brillouin zone
kstep = 0.01
kxlist = []
kylist = []
for kx in np.arange(-2 * np.pi, 2 * np.pi, kstep):
    for ky in np.arange(-2 * np.pi, 2 * np.pi, kstep):
        if abs(ky) <= 2 * pi / sqrt(3) and abs(ky + sqrt(3) * kx) <= 4 * pi / sqrt(3) and abs(
                ky - sqrt(3) * kx) <= 4 * pi / sqrt(3):
            kxlist.append(kx)
            kylist.append(ky)
kxlist = np.array(kxlist)
kylist = np.array(kylist)

# DMFT calculation for different interaction, density, and temperature
for Ut in Utlist:
    Ut = int(Ut)
    for density_required in [1]:
        for Temp in Templist:
            beta = 1 / Temp / t
            U = Ut * t
            for phi in philist:
                deltan0 = 1
                # Get the dispersion over the BZ
                DOSlist = []
                E_up = dispersion(kxlist, kylist, t, phi, spin=1)
                E_down = dispersion(kxlist, kylist, t, phi, spin=-1)
                energy = np.concatenate((E_up, E_down))
                h = np.histogram(energy, bins=n_bins, density=True)
                epsilon = 0.5 * (h[1][0:-1] + h[1][1:])
                rho = h[0]
                deltaE = h[1][1] - h[1][0]

                mu = U / 2 * density_required
                # Local Hamiltonian
                h_loc = Operator()
                h_loc += U * n('up', 0) * n('down', 0)

                # Construct the impurity solver
                S = Solver(beta=beta,
                           gf_struct=[('up', 1), ('down', 1)], n_l=50)
                # The local lattice Green's function
                G = S.G0_iw.copy()
                # Initial guess
                S.Sigma_iw << U / 2 * density_required


                # function to calculate the lattice Green function
                # by giving chemical potential and S.Sigma_iw
                def cal_G_lattice(mu):

                    g_up = GfImFreq(indices=[0], beta=beta)
                    g_down = GfImFreq(indices=[0], beta=beta)
                    Gtest = BlockGf(name_list=['up', 'down'], block_list=[g_up, g_down])
                    Gtest.zero()
                    for spin in ['up', 'down']:
                        # Hilbert transform
                        for i in mpi.slice_array(np.arange(n_bins)):
                            Gtest['%s' % (spin)] += rho[i] * deltaE * \
                                                    inverse(iOmega_n + mu - epsilon[i] - S.Sigma_iw['%s' % (spin)])
                        Gtest['%s' % (spin)] = mpi.all_reduce(mpi.world, Gtest['%s' % (spin)], lambda x, y: x + y)
                        mpi.barrier()

                    return Gtest


                # function to extract density for a given mu, to be used by dichotomy function to determine mu
                def Dens(mu):

                    dens = cal_G_lattice(mu).total_density()
                    if abs(dens.imag) > 1e-10:  # 1e-20:
                        mpi.report(
                            "Warning: Imaginary part of density will be ignored ({})".format(str(abs(dens.imag))))
                    return dens.real


                # DMFT loop
                n_loops = 30
                for iteration in range(n_loops):
                    it = iteration

                    S.Sigma_iw['up'] << .5 * (S.Sigma_iw['up'] + S.Sigma_iw['down'])
                    S.Sigma_iw['down'] << S.Sigma_iw['up']

                    G = cal_G_lattice(mu)

                    # DCA self-consistency - find next impurity G0
                    for block, g0 in S.G0_iw:
                        g0 << inverse(inverse(G[block]) + S.Sigma_iw[block])

                    if iteration >= n_loops-6:
                        cycles = int(30 * 1e7 / mpi.size)  # smooth the noise for the last 5 cycles
                    else:
                        cycles = int(1e7 / mpi.size)
                    # Run the solver. The results will be in S.G_tau, S.G_iw and S.G_l
                    S.solve(h_int=h_loc,  # Local Hamiltonian
                            n_cycles=cycles,  # Number of QMC cycles
                            length_cycle=100,  # Length of one cycle
                            n_warmup_cycles=int(1e4),  # Warmup cycles
                            measure_G_l=True,  # Measure G_l
                            measure_density_matrix=True,  # Measure the reduced density matrix
                            use_norm_as_weight=True)
                    # Extract accumulated density matrix
                    density = S.density_matrix
                    # Evaluate impurity occupations
                    h_loc_diag = S.h_loc_diagonalization
                    N_den = trace_rho_op(density, n('up', 0), h_loc_diag) + trace_rho_op(density, n('down', 0),
                                                                                         h_loc_diag)
                    Nim = S.G_iw.total_density().real
                    double_occp = trace_rho_op(density, n('up', 0) * n('down', 0), h_loc_diag)

                    # below we tune the chemical potential. for particle-hole symmetric case phi=pi/6 we don't need
                    # to change chemical potential. We can also call the dichotomy function to tune the density,
                    # but the method below will be more accurate, which is important for resistivity calculation.
                    if mpi.is_master_node():
                        delta_n = (np.abs(N_den) - density_required) / density_required
                        if abs(N_den - density_required) > 0.0002:
                            if delta_n > 0.05:
                                mu = mu - 0.15 * np.sign(delta_n) * t
                            if delta_n * deltan0 < 0:
                                mu = mu - delta_n / 5 * t
                            elif abs(delta_n) < 0.00025:
                                mu = mu - delta_n * 2 * t
                            else:
                                mu = mu - delta_n * t * 4
                            print("delta_n: ", delta_n)
                            print("mu: ", mu)

                        deltan0 = delta_n
                    mu = mpi.bcast(mu)

                    if mpi.is_master_node():
                        print('-----------------------------------------------')
                        print("phi= %s, Iteration = %s" % (np.around(phi / pi, 3), it))
                        print('-----------------------------------------------')
                        print('den', np.abs(N_den), Nim)
                        print('double', double_occp)
                        print('U/t', Ut)
                        print('T', Temp)
                        print('mu', mu)
                        if it >= 10:
                            with HDFArchive("data/n=%s_para_phi=%s_Ut=%s.h5" % (
                                    density_required, np.around(phi / pi, 3), Ut)) as A:
                                A['iterations'] = it
                                A['phi'] = phi
                                A['G_tau-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.G_tau
                                A['G_iw-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.G_iw
                                A['Sigma_iw-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.Sigma_iw
                                A['G-T=%st_U=%st_%i' % (Temp, Ut, it)] = G
                                A['mu-T=%st_U=%st_%i' % (Temp, Ut, it)] = mu
                                A['N_den-T=%st_U=%st_%i' % (Temp, Ut, it)] = N_den
                                A['Nim-T=%st_U=%st_%i' % (Temp, Ut, it)] = Nim
                                A['doub-T=%st_U=%st_%i' % (Temp, Ut, it)] = double_occp
