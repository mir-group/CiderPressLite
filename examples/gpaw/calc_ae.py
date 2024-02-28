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

from ciderpress.gpaw.cider_paw import get_cider_functional, CiderGPAW

name, charge, spin, functional = sys.argv[1:5]
charge = int(charge)
spin = int(spin)

vacuum = 3.0

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
    functional = 'functionals/TEST_CIDER_MGGA.yaml'
elif functional == 'TEST_ORB':
    import yaml
    with open('functionals/HOPT22.yaml', 'r') as f:
        functional = yaml.load(f, Loader=yaml.CLoader)
elif functional == 'TEST_LIN':
    import yaml
    with open('functionals/LIN_HOPT22.yaml', 'r') as f:
        functional = yaml.load(f, Loader=yaml.CLoader)
formula = Counter(atoms.get_atomic_numbers())

calc_num = 0
def perform_calc(atoms, xc, spinpol, hund=False, spin=None, charge=None):
    global calc_num
    if not isinstance(xc, str) or os.path.exists(xc):
        xc = get_cider_functional(
                functional, qmax=100,
                lambd=1.8, xmix=0.25,
                pasdw_store_funcs=True,
                pasdw_ovlp_fit=True)
    elif 'vdW' in xc:
        xc = VDWFunctional(xc)
    kwargs = {}
    if charge is not None:
        kwargs['charge'] = charge
    if spin is not None and spin > 0:
        natm = len(atoms.get_atomic_numbers())
        atoms.set_initial_magnetic_moments([spin/natm] * natm)
    atoms.calc = CiderGPAW(
        h=0.15,
        xc='PBE',
        #xc=xc,
        mode=PW(1000),
        txt='calc-{}.txt'.format(calc_num),
        maxiter=200,
        verbose=True,
        spinpol=spinpol,
        kpts=(1,1,1),
        hund=hund,
        convergence={'eigenstates': 1e-4},
        **kwargs,
    )
    etot = atoms.get_potential_energy()
    
    atoms.calc.set(xc=xc)
    calc_num += 1
    etot = atoms.get_potential_energy()
    
    #etot += atoms.calc.get_xc_difference(xc)
    #calc_num += 1

    del atoms.calc
    del xc
    print('Energy', etot)
    return etot

etot_ae = 0

#for el, num in formula.items():
#    atom = Atoms(chemical_symbols[el])
#    atom.set_cell((2*vacuum-0.25,2*vacuum,2*vacuum+0.25))
#    atom.center()
#    etot_atom = perform_calc(atom, functional, True, True)
#    etot_ae += num * etot_atom
etot_mol = perform_calc(atoms, functional, spinpol, spin=spin, charge=charge)
etot_ae -= etot_mol
for el, num in formula.items():
    atom = Atoms(chemical_symbols[el])
    atom.set_cell((2*vacuum-0.25,2*vacuum,2*vacuum+0.25))
    atom.center()
    etot_atom = perform_calc(atom, functional, True, True)
    etot_ae += num * etot_atom

print('toten: {}, aen: {}'.format(etot_mol, etot_ae))
with paropen('aeresult.txt', 'w') as f:
    f.write(str(etot_ae))

