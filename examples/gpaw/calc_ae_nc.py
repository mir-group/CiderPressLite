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

from ase import Atoms
from gpaw import GPAW, PW, scf
import ase
import numpy as np
from ase.build import molecule
import os, sys
from collections import Counter
from ase.data import chemical_symbols
from ase.parallel import paropen
from gpaw.xc.vdw import VDWFunctional

from ciderpress.gpaw.cider_paw import get_cider_functional

name, charge, spin, functional = sys.argv[1:5]
charge = int(charge)
spin = int(spin)

vacuum = 4.0

spinpol = True if spin > 0 else False
if name.endswith('.xyz'):
    ismol = True
    atoms = ase.io.read(name)
    atoms.center(vacuum=vacuum)
elif name.endswith('.cif'):
    ismol = False
    atoms = ase.io.read(name)
else:
    ismol = True
    atoms = molecule(name)
    atoms.center(vacuum=vacuum)

if functional == 'CIDER':
    functional = 'functionals/TEST_SL_GGA.yaml'
formula = Counter(atoms.get_atomic_numbers())

calc_num = 0
def perform_calc(atoms, xc, spinpol, hund=False, spin=None, charge=None):
    global calc_num
    if os.path.exists(xc):
        lambd = 1.8
        xc = get_cider_functional(
            functional, use_paw=False,
            qmax=100, lambd=lambd, xmix=0.25,
        )
    elif 'vdW' in xc:
        xc = VDWFunctional(xc)
    kwargs = {}
    if charge is not None:
        kwargs['charge'] = charge
    if spin is not None and spin > 0:
        natm = len(atoms.get_atomic_numbers())
        atoms.set_initial_magnetic_moments([spin/natm] * natm)
    atoms.calc = GPAW(
        h=0.15,
        xc='PBE',
        #xc=xc,
        mode=PW(1000),
        txt='nccalc-{}.txt'.format(calc_num),
        maxiter=200,
        verbose=True,
        spinpol=spinpol,
        kpts=(1,1,1),
        hund=hund,
        convergence={'eigenstates': 1e-4},
        setups='sg15',
        **kwargs,
    )
    atoms.get_potential_energy()
    atoms.calc.set(xc=xc)
    calc_num += 1
    etot = atoms.get_potential_energy()
    del atoms.calc
    del xc
    print('Energy', etot)
    return etot

etot_mol = perform_calc(atoms, functional, spinpol, spin=spin, charge=charge)
etot_ae = -1 * etot_mol
for el, num in formula.items():
    atom = Atoms(chemical_symbols[el])
    atom.set_cell((2*vacuum-0.25,2*vacuum,2*vacuum+0.25))
    atom.center()
    etot_atom = perform_calc(atom, functional, True, True)
    etot_ae += num * etot_atom

print('toten: {}, aen: {}'.format(etot_mol, etot_ae))
with paropen('aeresult.txt', 'w') as f:
    f.write(str(etot_ae))

