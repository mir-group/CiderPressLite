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

import numpy as np
from numpy.testing import assert_almost_equal
from pyscf import gto, dft, scf
from ciderpress.dft.xc_models import NormGPFunctional, GPFunctional
# IMPORT ri_cider MODULE FOR CIDER EVALUATION
from ciderpress.dft import ri_cider as numint
import os, sys
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from ase.data import ground_state_magnetic_moments, chemical_symbols
from ase import Atoms
from collections import Counter
import ase

"""
This script demonstrates running a CIDER calculation with accelerated
nonlocal feature evaluation. Example commands:

    python examples/pyscf/fast_cider.py <molecule_formula> <charge> <spin> <functional>
    python examples/pyscf/fast_cider.py H2 0 0 PBE
    python examples/pyscf/fast_cider.py O2 0 2 CIDER_NL_MGGA

<molecule_formula> is a chemical formula string like CH4, H2, etc. It must be included
in the list of molecules supported by ase.build.molecule()

<charge> is the integer charge of the system.

<spin> is the integer spin of the system 2S.

<functional> is the functional name. It can be semilocal, or it can be
CIDER_SL_GGA, CIDER_SL_MGGA, CIDER_NL_GGA, or CIDER_NL_MGGA, in which case
the corresponding example CIDER functional is run with the PBE0/CIDER
surrogate hybrid functional form.

At the end, prints out the total energy of the molecule and its atomization energy 
in Ha and eV, then saves the atomization energy in eV to aeresult.txt.

NOTE that the ri_cider module is used to initialize the fast CIDER calculation.
"""

name, charge, spin, functional = sys.argv[1:5]
charge = int(charge)
spin = int(spin)

spinpol = True if spin > 0 else False
BAS='def2-qzvppd'
if name == 'HF_stretch':
    BAS = 'def2-svp'
    atoms = Atoms(symbols=['H', 'F'], positions=[[0, 0, 0], [0, 0, 1.1]])
elif name.startswith('el-'):
    el = name[3:]
    atoms = Atoms(el)
elif name.endswith('.xyz'):
    ismol = True
    atoms = ase.io.read(name)
    atoms.center(vacuum=4)
else:
    ismol = True
    from ase.build import molecule
    atoms = molecule(name)
    atoms.center(vacuum=4)

if functional.startswith('CIDER'):
    functional = 'functionals/{}.yaml'.format(functional)
    is_cider = True
    mlfunc = functional
else:
    is_cider = False
formula = Counter(atoms.get_atomic_numbers())

mol = gto.M(atom=atoms_from_ase(atoms), basis=BAS, ecp=BAS, spin=spin, charge=charge, verbose=4)

if is_cider:
    # various CIDER settings, as explained in the ri_cider.setup_cider_calc docstring.
    settings = {
        'xkernel': 'GGA_X_PBE',
        'ckernel': 'GGA_C_PBE',
        'xmix': 0.25,
        'grid_level': 3,
        'debug': False,
        'amax': 3000.0,
        'cider_lmax': 10,
        'lambd': 1.6,
        'aux_beta': 1.6,
        'onsite_direct': True,
    }
    def run_calc(mol, spinpol):
        ks = numint.setup_cider_calc(mol, mlfunc, spinpol=spinpol, **settings)
        ks = ks.density_fit()
        # Can typically use smaller aux basis like def2-universal-jfit since
        # no EXX is evaluated, but we set the auxbasis to the larger, more
        # conservative def2-universal-jkfit here.
        ks.with_df.auxbasis = "def2-universal-jfit"
        ks = ks.apply(scf.addons.remove_linear_dep_)
        ks.small_rho_cutoff = 0.0
        etot = ks.kernel()
        return etot
else:
    def run_calc(mol, spinpol):
        if spinpol:
            ks = dft.UKS(mol)
        else:
            ks = dft.RKS(mol)
        ks = ks.density_fit()
        ks.with_df.auxbasis = "def2-universal-jkfit"
        ks = ks.apply(scf.addons.remove_linear_dep_)
        ks.xc = functional
        ks.grids.level = 3
        etot = ks.kernel()
        return etot

if spin == 0:
    spinpol = False
else:
    spinpol = True

etot_mol = run_calc(mol, spinpol)
etot_ae = -1 * etot_mol
for Z, count in formula.items():
    atom = gto.M(atom=chemical_symbols[Z], basis=BAS, ecp=BAS,
                 spin=int(ground_state_magnetic_moments[Z]),
                 verbose=4)
    etot_atom = run_calc(atom, True)
    etot_ae += count * etot_atom

print('Total and Atomization Energies, Ha')
print(etot_mol, etot_ae)
eh2ev = 27.211399
print('Total and Atomization Energies, eV')
print(etot_mol*eh2ev, etot_ae*eh2ev)
with open('aeresult.txt', 'w') as f:
    f.write(str(etot_ae*eh2ev))

