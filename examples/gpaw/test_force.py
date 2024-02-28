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
This unit test suite shows how to evaluate forces with
GPAW and CIDER functionals.
Warning: It might take a while to run compared to typical
unit tests.
"""
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac, PoissonSolver
from gpaw.xc.tools import vxc
from ciderpress.gpaw.cider_paw import get_cider_functional


def numeric_force(atoms, a, i, d=0.001, get_xc=None):
    """Compute numeric force on atom with index a, Cartesian component i,
    with finite step of size d
    """
    p0 = atoms.get_positions()
    p = p0.copy()
    p[a, i] += d
    atoms.set_positions(p, apply_constraint=False)
    eplus = atoms.get_potential_energy()
    p[a, i] -= 2 * d
    atoms.set_positions(p, apply_constraint=False)
    eminus = atoms.get_potential_energy()
    atoms.set_positions(p0, apply_constraint=False)
    return (eminus - eplus) / (2 * d)


def test_cider_forces(functional, get_xc=None):
    a = 5.45
    bulk = Atoms(symbols='Si8',
                 positions=[(0, 0, 0.1 / a),
                            (0, 0.5, 0.5),
                            (0.5, 0, 0.5),
                            (0.5, 0.5, 0),
                            (0.25, 0.25, 0.25),
                            (0.25, 0.75, 0.75),
                            (0.75, 0.25, 0.75),
                            (0.75, 0.75, 0.25)],
                 pbc=True)
    bulk.set_cell((a, a, a), scale_atoms=True)
    if get_xc is not None:
        functional = get_xc()
    calc = GPAW(h=0.15,
                mode=PW(520),
                xc=functional,
                nbands='150%',
                occupations=FermiDirac(width=0.01),
                kpts=(4, 4, 4),
                convergence={'energy': 1e-7},
                parallel={'augment_grids': True},
                )
    bulk.calc = calc
    f1 = bulk.get_forces()[0, 2]
    e1 = bulk.get_potential_energy()

    f2 = numeric_force(bulk, 0, 2, 0.001, get_xc=get_xc)
    print((f1, f2, f1 - f2))
    assert (f1 - f2) < 0.005


def test_gga():
    def get_xc():
        return get_cider_functional(
            'functionals/CIDER23_NL_GGA.yaml', qmax=300, lambd=1.8, xmix=0.25,
            pasdw_ovlp_fit=True, pasdw_store_funcs=True,
        )
    test_cider_forces(get_xc())


def test_mgga():
    def get_xc():
        return get_cider_functional(
            'functionals/CIDER23_NL_MGGA_DTR.yaml', qmax=300, lambd=1.8, xmix=0.25,
            pasdw_ovlp_fit=False, pasdw_store_funcs=False,
        )
    test_cider_forces(get_xc())


if __name__ == '__main__':
    test_cider_forces(get_cider_functional('functionals/CIDER23_SL_GGA.yaml'))
    test_cider_forces(get_cider_functional('functionals/CIDER23_SL_MGGA.yaml'))
    test_gga()
    test_mgga()

