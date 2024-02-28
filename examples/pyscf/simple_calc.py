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

from pyscf import gto, dft
from ciderpress.dft import ri_cider


mlfunc = 'functionals/TEST_CIDER_MGGA.yaml'

mol = gto.M(
    atom='F 0.0 0.0 0.0; F 0.0 0.0 1.42',
    basis='def2-tzvp',
)

# various CIDER settings, as explained in the ri_cider.setup_cider_calc docstring.
settings = {
    # Semi-local exchange and correlation parts
    'xkernel': 'GGA_X_PBE',
    'ckernel': 'GGA_C_PBE',
    'xmix': 0.25, # exact exchange mixing parameter
    'grid_level': 3, # level of PySCF grids
    'amax': 3000.0, # Note: This default needs to be increased for large atoms (Z>18)
    # Note: The following three settings control the precision of the nonlocal features.
    # The defaults below are fairly conservative, and increasing lambd / decreasing
    # cider_lmax can decrease computational cost. However, we are still benchmarking
    # this, so accepting these defaults for the time being is probably for the best.
    'cider_lmax': 10, 
    'lambd': 1.6, # Note: lambd=1.8 might be a bit more numerically stable with only
                  # a small decrease in accuracy
    'aux_beta': 1.6,
}
# setup CIDER with the above settings
ks = ri_cider.setup_cider_calc(
    mol, # gto.Mole object
    mlfunc, # NormGPFunctional or path to it
    spinpol=False, # True -> unrestricted KS, False -> restricted KS,
    **settings
)
ks = ks.density_fit()
ks.with_df.auxbasis = "def2-universal-jfit"
ks.kernel()

