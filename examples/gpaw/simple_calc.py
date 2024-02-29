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

from gpaw import GPAW, PW
from ase.build import bulk
from ciderpress.gpaw.cider_paw import get_cider_functional, CiderGPAW
import yaml

# NOTE: Run this script as follows:
# mpirun -np <NPROC> python simple_calc.py

atoms = bulk('Ge')

mlfunc = 'functionals/CIDER23_NL_MGGA.yaml'

# This is the initializer for CIDER functionals for GPAW
xc = get_cider_functional(
    # IMPORTANT: NormGPFunctional object or a path to a joblib or yaml file
    # containing a CIDER functional.
    #'functionals/TEST_CIDER_MGGA.yaml',
    mlfunc,
    mlfunc_format='yaml',
    # IMPORTANT: xmix is the mixing parameter for exact exchange. Default=0.25
    # gives the PBE0/CIDER surrogate hybrid.
    xmix=0.25,
    # largest q for interpolating feature expansion, default=300 is usually fine
    qmax=300,
    # lambda parameter for interpolating features. default=1.8 is usually fine.
    # Lower lambd is more precise
    lambd=1.8,
    # pasdw_store_funcs=False (default) saves memory. True reduces cost
    pasdw_store_funcs=False,
    # pasdw_ovlp_fit=True (default) uses overlap fitting to improve precision
    # of PAW correction terms of features.
    pasdw_ovlp_fit=True,
)

# Using CiderGPAW instead of the default GPAW calculator allows calculations
# to be restarted. GPAW calculations will run with CIDER functionals but
# cannot be saved and loaded properly.
atoms.calc = CiderGPAW(
    h=0.13, # use a reasonably small grid spacing
    xc=xc, # assign the CIDER functional to xc
    mode=PW(520), # plane-wave mode with 520 eV cutoff.
    txt='-', # output file, '-' for stdout
    occupations={'name': 'fermi-dirac', 'width': 0.01},
    # ^ Fermi smearing with 0.01 eV width
    kpts={'size': (12, 12, 12), 'gamma': False}, # kpt mesh parameters
    convergence={'energy': 1e-5}, # convergence energy in eV/electron
    # Set augments_grids=True for CIDER functionals to parallelize
    # XC energy and potential evaluation more effectively
    parallel={'augment_grids': True},
)
etot = atoms.get_potential_energy() # run the calculation

