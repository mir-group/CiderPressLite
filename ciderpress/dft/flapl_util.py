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
import ctypes
from ciderpress.lib import load_library as load_cider_library
from pyscf.gto.eval_gto import _get_intor_and_comp, make_loc, BLKSIZE, \
    make_screen_index, libcgto

from scipy.special import hyp1f1, gamma

libcider = load_cider_library('libcider')

FRAC_LAPL_POWER = 0.5

def initialize_flapl(a=0.25, d=0.002, size=4000, lmax=6, s=FRAC_LAPL_POWER):
    print('INITIALIZING FLAPL', s)
    x = a * (np.exp(d * np.arange(size)) - 1)
    f = np.zeros((lmax + 1, size))
    xbig = x[x > 300]
    xmax = x[x < 300][-1]
    for l in range(lmax + 1):
        f[l] = hyp1f1(1.5 + s + l, 1.5 + l, -x)
        f[l] *= 2**(2 * s) * gamma(1.5 + s + l) / gamma(1.5 + l)
        fsmall = f[l, x < 300][-1]
        f[l, x > 300] = fsmall * (xmax / xbig)**(1.5 + s + l)
    libcider.initialize_spline_1f1(
        f.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(a),
        ctypes.c_double(d),
        ctypes.c_int(size),
        ctypes.c_int(lmax),
        ctypes.c_double(s),
    )

def free_flapl():
    libcider.free_spline_1f1()

def eval_flapl_gto(mol, coords, shls_slice=None, non0tab=None,
                   ao_loc=None, cutoff=None, out=None,
                   debug=False):
    if not libcider.check_1f1_initialization():
        initialize_flapl()
    if non0tab is not None:
        if (non0tab == 0).any():
            # TODO implement some sort of screening later
            raise NotImplementedError
    if mol.cart:
        feval = 'GTOval_cart_deriv0'
    else:
        feval = 'GTOval_sph_deriv0'
    eval_name, comp = _get_intor_and_comp(mol, feval)
    if comp != 1:
        raise NotImplementedError

    atm = np.asarray(mol._atm, dtype=np.int32, order='C')
    bas = np.asarray(mol._bas, dtype=np.int32, order='C')
    env = np.asarray(mol._env, dtype=np.double, order='C')
    coords = np.asarray(coords, dtype=np.double, order='F')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]

    if ao_loc is None:
        ao_loc = make_loc(bas, eval_name)

    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]
    if 'spinor' in eval_name:
        ao = np.ndarray((2, comp, nao, ngrids), dtype=np.complex128,
                        buffer=out).transpose(0, 1, 3, 2)
    else:
        #ao = np.ndarray((comp, nao, ngrids), buffer=out).transpose(0, 2, 1)
        ao = np.zeros((comp, nao, ngrids)).transpose(0, 2, 1)

    if non0tab is None:
        if cutoff is None:
            non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, nbas),
                              dtype=np.uint8)
        else:
            non0tab = make_screen_index(mol, coords, shls_slice, cutoff)

    if eval_name != 'GTOval_sph_deriv0':
        raise NotImplementedError
    # drv = getattr(libcgto, eval_name)
    # normal call tree is GTOval_sph_deriv0 -> GTOval_sph -> GTOeval_sph_drv
    drv = getattr(libcgto, 'GTOeval_sph_drv')
    if debug:
        contract_fn = getattr(libcgto, 'GTOcontract_exp0')
    else:
        contract_fn = getattr(libcider, 'GTOcontract_flapl0')
    drv(
        getattr(libcgto, 'GTOshell_eval_grid_cart'),
        contract_fn,
        ctypes.c_double(1),
        ctypes.c_int(ngrids),
        (ctypes.c_int * 2)(1, 1),
        (ctypes.c_int * 2)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p)
    )

    if comp == 1:
        if 'spinor' in eval_name:
            ao = ao[:, 0]
        else:
            ao = ao[0]
    return ao

def eval_kao(mol, coords, deriv=0, shls_slice=None,
             non0tab=None, cutoff=None, out=None, verbose=None):
    assert deriv == 0
    comp = 1
    if mol.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
    return eval_flapl_gto(mol, coords, shls_slice, non0tab,
                          cutoff=cutoff, out=out)
