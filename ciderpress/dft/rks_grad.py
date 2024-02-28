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
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
import pyscf.grad.rks as rks_grad
import pyscf.df.grad.rks as rks_grad_df
from pyscf.grad.rks import _gga_grad_sum_, _tau_grad_dot_, \
    grids_response_cc
from pyscf.dft import numint, radi, gen_grid, xc_deriv
from pyscf import __config__
from ciderpress.dft.ri_cider import get_wv_rks_cider
import ctypes


def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if mf.nlc != '':
        if ks_grad.nlcgrids is not None:
            nlcgrids = ks_grad.nlcgrids
        else:
            nlcgrids = mf.nlcgrids
        if nlcgrids.coords is None:
            nlcgrids.build(with_non0tab=True)
    if grids.coords is None:
        grids.build(with_non0tab=True)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.nlc:
            assert 'VV10' in mf.nlc.upper()
            enlc, vnlc = get_vxc_full_response(ni, mol, nlcgrids,
                                               mf.xc+'__'+mf.nlc, dm,
                                               max_memory=max_memory,
                                               verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.nlc:
            assert 'VV10' in mf.nlc.upper()
            enlc, vnlc = get_vxc(ni, mol, nlcgrids, mf.xc+'__'+mf.nlc, dm,
                                 max_memory=max_memory,
                                 verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    has_df = hasattr(mf, 'with_df') and mf.with_df is not None
    if not has_df:
        # no density fitting
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            vj = ks_grad.get_j(mol, dm)
            vxc += vj
        else:
            vj, vk = ks_grad.get_jk(mol, dm)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                with mol.with_range_coulomb(omega):
                    vk += ks_grad.get_k(mol, dm) * (alpha - hyb)
            vxc += vj - vk * .5

        return lib.tag_array(vxc, exc1_grid=exc)
    else:
        # has density fitting, TODO check that this always works
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            vj = ks_grad.get_j(mol, dm)
            vxc += vj
            if ks_grad.auxbasis_response:
                e1_aux = vj.aux.sum ((0,1))
        else:
            vj, vk = ks_grad.get_jk(mol, dm)
            if ks_grad.auxbasis_response:
                vk.aux *= hyb
            vk[:] *= hyb # Don't erase the .aux tags!
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                # TODO: replaced with vk_sr which is numerically more stable for
                # inv(int2c2e)
                vk_lr = ks_grad.get_k(mol, dm, omega=omega)
                vk[:] += vk_lr * (alpha - hyb)
                if ks_grad.auxbasis_response:
                    vk.aux[:] += vk_lr.aux * (alpha - hyb)
            vxc += vj - vk * .5
            if ks_grad.auxbasis_response:
                e1_aux = (vj.aux - vk.aux * .5).sum ((0,1))

        if ks_grad.auxbasis_response:
            logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
        else:
            vxc = lib.tag_array(vxc, exc1_grid=exc)
        return vxc


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    if ni.ri_conv is None:
        ni.setup_aux(mol)
    xctype = ni._xc_type(xc_code)
    if xctype == 'NLC':
        return rks_grad.get_vxc(ni, mol, grids, xc_code, dms,
                                relativity, hermi, max_memory,
                                verbose)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    ao_deriv = 2
    Nrhofeat = 4 if xctype == 'GGA' else 5

    vmat = np.zeros((nset,3,nao,nao))

    rho_full = np.zeros((nset, Nrhofeat, grids.weights.size), dtype=np.float64, order='C')
    ip0 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rho = make_rho(idm, ao, mask, xctype)
            rho_full[idm,:,ip0:ip1] = rho
        ip0 = ip1
    ip0 = ip1 = None

    wv_full = []
    for idm in range(nset):
        wv = get_wv_rks_cider(
            ni, mol, grids, xc_code, rho_full[idm],
            relativity, hermi, verbose,
        )[2]
        wv_full.append(wv)

    ip0 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rho = np.ascontiguousarray(rho_full[idm,:,ip0:ip1])
            wv = np.ascontiguousarray(wv_full[idm][:,ip0:ip1])
            _gga_grad_sum_(vmat[idm], mol, ao, wv[:4], mask, ao_loc)
            if xctype == 'MGGA':
                wv[5] *= 2.0
                _tau_grad_dot_(vmat[idm], mol, ao, wv[5], mask, ao_loc, True)
        ip0 = ip1

    exc = None
    if nset == 1:
        vmat = vmat[0]
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0,
                          hermi=1, max_memory=2000, verbose=None):
    if ni.ri_conv is None:
        ni.setup_aux(mol)
    xctype = ni._xc_type(xc_code)
    if xctype == 'NLC':
        return rks_grad.get_vxc(ni, mol, grids, xc_code, dms,
                                relativity, hermi, max_memory,
                                verbose)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    excsum = 0
    vmat = np.zeros((3,nao,nao))

    ao_deriv = 2
    Nrhofeat = 4 if xctype == 'GGA' else 5

    rho_full = np.zeros((nset, Nrhofeat, grids.weights.size), dtype=np.float64, order='C')
    ip0 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ip1 = ip0 + weight.size
        rho = make_rho(0, ao, mask, xctype)
        rho_full[0,:,ip0:ip1] = rho
        ip0 = ip1
    ip0 = ip1 = None

    wv_full = []
    cidergg_g, excsum_tmp, wv, exc = get_wv_rks_cider(
        ni, mol, grids, xc_code, rho_full[0],
        relativity, hermi, verbose, grad_mode=True,
    )[:4]
    wv_full.append(wv)

    ip0 = 0
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                        cutoff=grids.cutoff)
        ip1 = ip0 + weight.size
        rho = np.ascontiguousarray(rho_full[0,:,ip0:ip1])
        wv = np.ascontiguousarray(wv_full[0][:,ip0:ip1])
        vtmp = np.zeros((3,nao,nao))
        _gga_grad_sum_(vtmp, mol, ao, wv[:4], mask, ao_loc)
        if xctype == 'MGGA':
            wv[5] *= 2.0
            _tau_grad_dot_(vtmp, mol, ao, wv[5], mask, ao_loc, True)
        vmat += vtmp

        # response of weights
        excsum += np.einsum('r,r,nxr->nx', exc[ip0:ip1], rho[0], weight1)
        # response of grids coordinates
        excsum[atm_id] += np.einsum('xij,ji->x', vtmp, dms) * 2
        excsum += np.dot(weight1, cidergg_g[ip0:ip1])
        ip0 = ip1

    excsum += excsum_tmp
    # - sign because nabla_X = -nabla_x
    return excsum, -vmat



class Gradients(rks_grad.Gradients):

    get_veff = get_veff


class DFGradients(rks_grad_df.Gradients):

    get_veff = get_veff
