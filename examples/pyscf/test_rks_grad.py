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
This unit test suite shows simple examples for running
force evaluations for RKS calculations.
"""
import unittest
from pyscf import gto, lib
from ciderpress.dft.ri_cider import setup_cider_calc
from ciderpress.dft.xc_models import NormGPFunctional
from numpy.testing import assert_almost_equal


CONV_TOL = 1e-12

SETTINGS = {
    'xkernel': 'GGA_X_PBE',
    'ckernel': 'GGA_C_PBE',
    'xmix': 0.25,
    'grid_level': 2,
    'debug': False,
    'amax': 1000.0,
    'cider_lmax': 8,
    'lambd': 1.8,
    'aux_beta': 1.8,
    'onsite_direct': True,
}

def build_ks_calc(mol, mlfunc, fnl):
    assert mol.spin == 0
    spinpol = False
    calc = setup_cider_calc(
        mol, mlfunc, spinpol=spinpol,
        _force_nonlocal=fnl,
        **SETTINGS,
    )
    return calc

def setUpModule():
    global mol, mlfuncs, mf1, mf2, mf3, mf4, mf5
    mlfuncs = [
        NormGPFunctional.load('functionals/{}.yaml'.format(fname)) for fname in \
        [
            'CIDER23_SL_GGA',
            'CIDER23_SL_MGGA',
            'CIDER23_NL_GGA',
            'CIDER23_NL_MGGA',
        ]
    ]
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.build()
    mf1 = build_ks_calc(mol, mlfuncs[1], False)
    mf2 = build_ks_calc(mol, mlfuncs[1], True)
    mf3 = build_ks_calc(mol, mlfuncs[3], False)
    mf1.conv_tol = CONV_TOL
    mf1.kernel()
    mf2.conv_tol = CONV_TOL
    mf2.kernel()
    mf3.conv_tol = CONV_TOL
    mf3.kernel()
    mf4 = build_ks_calc(mol, mlfuncs[3], False)
    mf4.grids.level = 3
    mf4.grids.build()
    mf4.conv_tol = CONV_TOL
    mf4.kernel()
    mf5 = build_ks_calc(mol, mlfuncs[1], False)
    mf5.grids.build()
    mf5.conv_tol = CONV_TOL
    mf5.kernel()

def tearDownModule():
    global mol, mlfuncs, mf1, mf2, mf3, mf4, mf5
    mol.stdout.close()
    del mol, mlfuncs, mf1, mf2, mf3, mf4, mf5


class KnownValues(unittest.TestCase):

    def test_semilocal_mlxc_grad(self):
        g1 = mf1.nuc_grad_method().kernel()
        g2 = mf2.nuc_grad_method().kernel()
        assert_almost_equal(g1, g2, 10)

    def test_fd_cider_grad_nogrid(self):
        g = mf4.nuc_grad_method().kernel()
        mol1 = mol.copy()
        mf_scanner = mf4.as_scanner()

        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        assert_almost_equal(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 3)

    def test_finite_difference(self):
        g = mf5.nuc_grad_method().set(grid_response=True).kernel()
        mol1 = mol.copy()
        mf_scanner = mf5.as_scanner()

        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        # TODO precision for LDA is 6
        assert_almost_equal(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_finite_difference_nl(self):
        g = mf3.nuc_grad_method().set(grid_response=True).kernel()
        mol1 = mol.copy()
        mf_scanner = mf3.as_scanner()

        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        # TODO precision for LDA is 6
        assert_almost_equal(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 6)


if __name__ == '__main__':
    unittest.main()
