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

from pyscf.gto.moleintor import ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF
from pyscf.data.radii import BRAGG, COVALENT
import numpy as np

ATOM_OF = 0

def get_core_shls(atm, bas, env, thr=10, covalent=False):
    assert (bas[:,NPRIM_OF] == 1).all()
    assert (bas[:,NCTR_OF] == 1).all()
    charges = atm[bas[:,ATOM_OF],0]
    rads = COVALENT[charges] if covalent else BRAGG[charges]
    exps = env[bas[:,PTR_EXP]]
    metric = exps * rads**2
    corebas = metric > thr
    valbas = np.logical_not(corebas)
    cres = np.arange(bas.shape[0], dtype=np.int32)[corebas]
    vres = np.arange(bas.shape[0], dtype=np.int32)[valbas]
    return cres, vres

def get_aos_from_shls(ao_loc, shls):
    lsts = []
    for shl in shls:
        lsts.append(np.arange(ao_loc[shl], ao_loc[shl+1]))
    return np.concatenate(lsts)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf.df.addons import aug_etb, make_auxmol
    mol = gto.M(atom='H 0 0 0; F 0 0 0.93', basis='def2-tzvppd')
    auxbas = aug_etb(mol, beta=1.8)
    auxmol = make_auxmol(mol, auxbasis=auxbas)
    cbas, vbas = get_core_shls(auxmol._atm, auxmol._bas, auxmol._env)
    cbas, vbas = get_core_shls(auxmol._atm, auxmol._bas, auxmol._env, thr=13, covalent=True)
    print(cbas, vbas)
    ao_loc = auxmol.ao_loc_nr()
    print(get_aos_from_shls(ao_loc, cbas))
