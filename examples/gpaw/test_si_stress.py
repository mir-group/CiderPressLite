#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

"""
This unit test suite shows how to evaluate stress with
GPAW and CIDER functionals.
Warning: It might take a while to run compared to typical
unit tests.
"""
import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW, Mixer
from gpaw.mpi import world
from ase.parallel import parprint
from ciderpress.gpaw.cider_paw import get_cider_functional


k=3


def test_pw_si_stress(xc, use_pp=False, s_numerical=None):
    si = bulk('Si')
    si.calc = GPAW(mode=PW(250),
                   #h=0.15,
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=xc,
                   kpts=(k, k, k),
                   convergence={'energy': 1e-8},
                   parallel={'domain': min(2, world.size)},
                   setups='sg15' if use_pp else 'paw',
                   txt='si.txt')

    si.set_cell(np.dot(si.cell,
                       [[1.02, 0, 0.03],
                        [0, 0.99, -0.02],
                        [0.2, -0.01, 1.03]]),
                scale_atoms=True)

    etot = si.get_potential_energy()
    print(etot)

    # Trigger nasty bug (fixed in !486):
    si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1

    s_analytical = si.get_stress()
    parprint(s_analytical)
    # TEST_CIDER_GGA Numerical
    # [-0.00261187 -0.03790705 -0.03193711 -0.0209582   0.13427714  0.00928778]
    # TEST_CIDER_MGGA Numerical
    # [-0.00681636 -0.04026119 -0.03689781 -0.02227667  0.14441494  0.00907815]
    if s_numerical is None:
        s_numerical = si.calc.calculate_numerical_stress(si, 1e-5)
    s_err = s_numerical - s_analytical

    parprint('Analytical stress:\n', s_analytical)
    parprint('Numerical stress:\n', s_numerical)
    parprint('Error in stress:\n', s_err)
    assert np.all(abs(s_err) < 1e-4)


def get_xc(fname, use_paw=True):
    return get_cider_functional(
        fname, qmax=120, lambd=1.8, xmix=0.25,
        pasdw_ovlp_fit=True, pasdw_store_funcs=True,
        use_paw=use_paw,
    )


if __name__ == '__main__':
    xc = get_xc('functionals/CIDER23_NL_GGA.yaml')
    s_numerical = np.array([-0.00261187, -0.03790705, -0.03193711,
                            -0.0209582,   0.13427714,  0.00928778])
    # test_pw_si_stress(xc, s_numerical=None)

    xc = get_xc('functionals/CIDER23_NL_MGGA_DTR.yaml')
    s_numerical = np.array([-0.00681636, -0.04026119, -0.03689781,
                            -0.02227667,  0.14441494,  0.00907815])
    test_pw_si_stress(xc, s_numerical=None)

    xc = get_xc('functionals/CIDER23_SL_GGA.yaml')
    #s_numerical = np.array([-0.00261187, -0.03790705, -0.03193711,
    #                        -0.0209582,   0.13427714,  0.00928778])
    test_pw_si_stress(xc, s_numerical=None)

    xc = get_xc('functionals/CIDER23_SL_MGGA.yaml')
    #s_numerical = np.array([-0.00681636, -0.04026119, -0.03689781,
    #                        -0.02227667,  0.14441494,  0.00907815])
    test_pw_si_stress(xc, s_numerical=None)

    xc = get_xc('functionals/CIDER23_NL_GGA.yaml', use_paw=False)
    s_numerical = np.array([0.00205983, -0.03604186, -0.02808641,
                            -0.02021089,  0.1333823,   0.00980205])
    test_pw_si_stress(xc, use_pp=True, s_numerical=s_numerical)
   
    # It is not possible to evaluate MGGA stress with pseudopotentials.
    # It is necessary to use PAW instead.
    try:
        xc = get_xc('functionals/CIDER23_NL_MGGA.yaml', use_paw=False)
        #s_numerical = np.array([-0.00681636, -0.04026119, -0.03689781,
        #                        -0.02227667,  0.14441494,  0.00907815])
        test_pw_si_stress(xc, use_pp=True, s_numerical=None)
    except NotImplementedError:
        pass

