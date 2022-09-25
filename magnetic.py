"""
DMFT code for the ferromagnetic state of Hubbard model at phase phi= pi/3~ 2pi/3,
which is equivalent to the 120 antiferromagnetic state at phi =0~ pi/3, because of gauge transformation.
120 order at phi=x corresponds to ferro-x order at phi=x+pi/3.
"""

from math import *
from triqs.gf import *
from triqs.operators import *
from triqs_cthyb import Solver
from h5 import HDFArchive
import triqs.utility.mpi as mpi
import numpy as np
from triqs.atom_diag import trace_rho_op

# Parameters
t = 0.25
Templist = [0.1, 0.2, 0.3]
Utlist = [6, 8, 10]
philist = [pi / 2]


# function to calculate the dispersion of the Hubbard model with hopping t, phase phi and spin +/-
def dispersion(kxlist, kylist, t=1, phi=0, spin=1):
    phi = spin * phi
    return -2 * t * (np.cos(kxlist + phi) + np.cos(-kxlist / 2 + sqrt(3) / 2 * kylist + phi) + np.cos(
        -kxlist / 2 - sqrt(3) / 2 * kylist + phi))


# build the brillouin zone
kstep = 0.1
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
Ns = len(kxlist)  # number of sites

# DMFT calculation for different interaction, and temperature
for Temp in Templist:
    beta = 1 / Temp / t
    for Ut in Utlist:
        U = Ut * t
        for phi in philist:
            # Get the dispersion over the BZ
            E_up = dispersion(kxlist, kylist, t, phi, spin=1)
            E_down = dispersion(kxlist, kylist, t, phi, spin=-1)

            density_required = 1.
            mu = U / 2
            # Local Hamiltonian
            h_loc = Operator()
            h_loc += U * n('ferro', 0) * n('ferro', 1)
            # Construct the impurity solver
            S = Solver(beta=beta,
                       gf_struct=[('ferro', [0, 1])], n_l=50)
            # The local lattice Green's function
            G = S.G0_iw.copy()
            # Initial guess
            S.Sigma_iw['ferro'][0, 0] << U / 2
            S.Sigma_iw['ferro'][1, 1] << U / 2
            S.Sigma_iw['ferro'][0, 1] << U / 2
            S.Sigma_iw['ferro'][1, 0] << U / 2


            # function to calculate the lattice Green function by giving chemical potential and S.Sigma_iw
            def cal_G_lattice(mu):

                g = GfImFreq(indices=[0, 1], beta=beta)
                Gtest = BlockGf(name_list=['ferro'], block_list=[g])
                Gtest.zero()
                Gsum = Gtest.copy()

                for ki in mpi.slice_array(np.arange(Ns)):
                    Gtest['ferro'][0, 0] << iOmega_n + mu - E_up[ki] - S.Sigma_iw['ferro'][0, 0]
                    Gtest['ferro'][1, 1] << iOmega_n + mu - E_down[ki] - S.Sigma_iw['ferro'][0, 0]
                    Gtest['ferro'][0, 1] << - S.Sigma_iw['ferro'][0, 1]
                    Gtest['ferro'][1, 0] << Gtest['ferro'][0, 1]
                    Gtest['ferro'].invert()
                    Gsum['ferro'] += Gtest['ferro']
                Gsum['ferro'] = mpi.all_reduce(mpi.world, Gsum['ferro'], lambda x, y: x + y)
                mpi.barrier()
                Gsum['ferro'] << Gsum['ferro'] / Ns

                return Gsum


            # function to extract density for a given mu, to be used by dichotomy function to determine mu
            def Dens(mu):

                dens = cal_G_lattice(mu).total_density()
                if abs(dens.imag) > 1e-10:  # 1e-20:
                    mpi.report("Warning: Imaginary part of density will be ignored ({})".format(str(abs(dens.imag))))
                return dens.real


            # DMFT loop
            n_loops = 30

            #           load the previous result
            #            if mpi.is_master_node():
            #                A = HDFArchive("data/fer_T=%st_phi=%s.h5"%(Temp,np.around(phi/pi,3)),'a')
            #                it=30
            #
            #                S.Sigma_iw = A['Sigma_iw-T=%st_U=%st_%i'%(Temp,Ut,it)]
            #                mu = A['mu-T=%st_U=%st_%i'%(Temp,Ut,it)]
            #                del A
            #            S.Sigma_iw = mpi.bcast(S.Sigma_iw)
            #            mu = mpi.bcast(mu)
            deltan0 = 1
            for iteration in range(n_loops):
                it = iteration + 0

                S.Sigma_iw['ferro'][0, 0] << 0.5 * (S.Sigma_iw['ferro'][0, 0] + S.Sigma_iw['ferro'][1, 1])
                S.Sigma_iw['ferro'][1, 1] << S.Sigma_iw['ferro'][0, 0]
                S.Sigma_iw['ferro'][1, 0] << 0.5 * (S.Sigma_iw['ferro'][1, 0] + S.Sigma_iw['ferro'][0, 1])
                S.Sigma_iw['ferro'][0, 1] << S.Sigma_iw['ferro'][1, 0]

                # DCA self-consistency - find next impurity G0
                G = cal_G_lattice(mu)
                S.G0_iw['ferro'] << inverse(inverse(G['ferro']) + S.Sigma_iw['ferro'])

                # #                 # Run the solver. The results will be in S.G_tau, S.G_iw and S.G_l

                S.solve(h_int=h_loc,  # Local Hamiltonian
                        n_cycles=int(2e7 / mpi.size),  # Number of QMC cycles
                        length_cycle=300,  # Length of one cycle
                        n_warmup_cycles=int(5e4),  # Warmup cycles
                        measure_G_l=True,  # Measure G_l
                        measure_density_matrix=True,  # Measure the reduced density matrix
                        use_norm_as_weight=True)

                # Extract accumulated density matrix
                density = S.density_matrix
                # Evaluate impurity occupations
                h_loc_diag = S.h_loc_diagonalization
                N_den = trace_rho_op(density, n('ferro', 0), h_loc_diag) + trace_rho_op(density, n('ferro', 1),
                                                                                        h_loc_diag)
                Nim = S.G_iw.total_density().real

                # below is the adjustment of the chemical potential. I use this process because the resistivity needs
                # a very accurate density. For other purpose you can replace the part below by dichotomy function to
                # determine chemical potential
                if mpi.is_master_node():
                    delta_n = (N_den - density_required) / density_required
                    if abs(N_den - density_required) > 0.0002:
                        if abs(delta_n) > 0.2:
                            mu = mu - np.sign(delta_n) * 0.2
                        elif delta_n * deltan0 < 0:
                            mu = mu * (1 - delta_n / 5)
                        elif abs(delta_n) < 0.05:
                            mu = mu * (1 - delta_n)
                        else:
                            mu = mu * (1 - delta_n)
                        print("delta_n: ", delta_n)
                        print("mu: ", mu)

                    deltan0 = delta_n
                mu = mpi.bcast(mu)

                if mpi.is_master_node():
                    print('-----------------------------------------------')
                    print("phi= %s, Iteration = %s" % (np.around(phi / pi, 3), it))
                    print('-----------------------------------------------')
                    print('den', N_den, Nim)
                    print('U/t', Ut)
                    print('T', Temp)
                    print('mu', mu)
                    with HDFArchive("data/fer_T=%st_phi=%s.h5" % (Temp, np.around(phi / pi, 3))) as A:
                        A['iterations'] = it
                        A['phi'] = phi
                        A['Ut'] = Ut
                        A['G_tau-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.G_tau
                        A['G_iw-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.G_iw
                        A['Sigma_iw-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.Sigma_iw
                        A['G-T=%st_U=%st_%i' % (Temp, Ut, it)] = G
                        A['mu-T=%st_U=%st_%i' % (Temp, Ut, it)] = mu
                        A['N_den-T=%st_U=%st_%i' % (Temp, Ut, it)] = N_den
                        A['Nim-T=%st_U=%st_%i' % (Temp, Ut, it)] = Nim
                        A['h_loc_diag-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.h_loc_diagonalization
                        A['density_matrix-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.density_matrix
                        A['G_l-T=%st_U=%st_%i' % (Temp, Ut, it)] = S.G_l
