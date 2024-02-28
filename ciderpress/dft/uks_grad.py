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
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad.rks import _gga_grad_sum_, _tau_grad_dot_, \
    grids_response_cc
from pyscf.grad import rks as rks_grad
from pyscf.grad import uks as uks_grad
from pyscf.df.grad import uks as uks_grad_df
from pyscf.dft.numint import _format_uks_dm
from pyscf.dft import gen_grid
from ciderpress.dft.ri_cider import get_wv_uks_cider
from pyscf import __config__


_gga_grad_sum_ = rks_grad._gga_grad_sum_
_tau_grad_dot_ = rks_grad._tau_grad_dot_

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
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
        if mf.nlc:
            assert 'VV10' in mf.nlc.upper()
            enlc, vnlc = rks_grad.get_vxc_full_response(
                ni, mol, nlcgrids, mf.xc+'__'+mf.nlc, dm[0]+dm[1],
                max_memory=max_memory, verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.nlc:
            assert 'VV10' in mf.nlc.upper()
            enlc, vnlc = rks_grad.get_vxc(ni, mol, nlcgrids, mf.xc+'__'+mf.nlc,
                                          dm[0]+dm[1], max_memory=max_memory,
                                          verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    has_df = hasattr(mf, 'with_df') and mf.with_df is not None
    if not has_df:
        # no density fitting
        if abs(hyb) < 1e-10:
            vj = ks_grad.get_j(mol, dm)
            vxc += vj[0] + vj[1]
        else:
            vj, vk = ks_grad.get_jk(mol, dm)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                with mol.with_range_coulomb(omega):
                    vk += ks_grad.get_k(mol, dm) * (alpha - hyb)
            vxc += vj[0] + vj[1] - vk

        return lib.tag_array(vxc, exc1_grid=exc)
    else:
        # density fitting is used, TODO check this always works more thoroughly
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            vj = ks_grad.get_j(mol, dm)
            vxc += vj[0] + vj[1]
            if ks_grad.auxbasis_response:
                e1_aux = vj.aux.sum ((0,1))
        else:
            vj, vk = ks_grad.get_jk(mol, dm)
            if ks_grad.auxbasis_response:
                vk.aux = vk.aux * hyb
            vk[:] *= hyb # inplace * for vk[:] to keep the .aux tag
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                vk_lr = ks_grad.get_k(mol, dm, omega=omega)
                vk[:] += vk_lr * (alpha - hyb)
                if ks_grad.auxbasis_response:
                    vk.aux[:] += vk_lr.aux * (alpha - hyb)
            vxc += vj[0] + vj[1] - vk
            if ks_grad.auxbasis_response:
                e1_aux = vj.aux.sum ((0,1))
                e1_aux -= numpy.trace (vk.aux, axis1=0, axis2=1)

        if ks_grad.auxbasis_response:
            logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
        else:
            vxc = lib.tag_array(vxc, exc1_grid=exc)
        return vxc


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
            max_memory=2000, verbose=None):
    if ni.ri_conv is None:
        ni.setup_aux(mol)
    xctype = ni._xc_type(xc_code)
    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    ao_loc = mol.ao_loc_nr()
    ao_deriv = 2
    Nrhofeat = 4 if xctype == 'GGA' else 5

    vmat = np.zeros((nset,2,3,nao,nao))

    rhoa_full = np.zeros((nset, Nrhofeat, grids.weights.size),
                         dtype=np.float64, order='C')
    rhob_full = np.zeros((nset, Nrhofeat, grids.weights.size),
                         dtype=np.float64, order='C')
    ip0 = 0

    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rhoa_full[idm,:,ip0:ip1] = make_rhoa(idm, ao, mask, xctype)
            rhob_full[idm,:,ip0:ip1] = make_rhob(idm, ao, mask, xctype)
        ip0 = ip1
    ip0 = ip1 = None

    wva_full = []
    wvb_full = []
    weight, coords = grids.weights, grids.coords
    for idm in range(nset):
        wva, wvb = get_wv_uks_cider(
            ni, mol, grids, xc_code, (rhoa_full[idm], rhob_full[idm]),
            relativity, hermi, verbose,
        )[2:4]
        wva_full.append(wva)
        wvb_full.append(wvb)

    ip0 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rho_a = np.ascontiguousarray(rhoa_full[idm,:,ip0:ip1])
            rho_b = np.ascontiguousarray(rhob_full[idm,:,ip0:ip1])
            wva = np.ascontiguousarray(wva_full[idm][:,ip0:ip1])
            wvb = np.ascontiguousarray(wvb_full[idm][:,ip0:ip1])
            _gga_grad_sum_(vmat[idm,0], mol, ao, wva[:4], mask, ao_loc)
            _gga_grad_sum_(vmat[idm,1], mol, ao, wvb[:4], mask, ao_loc)
            if xctype == 'MGGA':
                wva[5] *= 2.0
                wvb[5] *= 2.0
                _tau_grad_dot_(vmat[idm,0], mol, ao, wva[5], mask, ao_loc, True)
                _tau_grad_dot_(vmat[idm,1], mol, ao, wvb[5], mask, ao_loc, True)

    if nset == 1:
        vmat = vmat[0]

    exc = np.zeros((mol.natm,3))
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
                          max_memory=2000, verbose=None):
    if ni.ri_conv is None:
        ni.setup_aux(mol)
    xctype = ni._xc_type(xc_code)
    if xctype == 'NLC':
        raise NotImplementedError
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    aoslices = mol.aoslice_by_atom()
    Nrhofeat = 4 if xctype == 'GGA' else 5
    ao_deriv = 2

    excsum = 0
    vmat = np.zeros((2,3,nao,nao))

    rhoa_full = np.zeros((1, Nrhofeat, grids.weights.size),
                         dtype=np.float64, order='C')
    rhob_full = np.zeros((1, Nrhofeat, grids.weights.size),
                         dtype=np.float64, order='C')
    ip0 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ip1 = ip0 + weight.size
        rhoa_full[0,:,ip0:ip1] = make_rho(0, ao, mask, xctype)
        rhob_full[0,:,ip0:ip1] = make_rho(1, ao, mask, xctype)
        ip0 = ip1
    ip0 = ip1 = None

    wva_full = []
    wvb_full = []
    cidergg_g, excsum_tmp, wva, wvb, exc = get_wv_uks_cider(
        ni, mol, grids, xc_code, (rhoa_full[0], rhob_full[0]),
        relativity, hermi, verbose, grad_mode=True,
    )[:5]
    wva_full.append(wva)
    wvb_full.append(wvb)

    ip0 = 0
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                        cutoff=grids.cutoff)
        ip1 = ip0 + weight.size
        rho_a = np.ascontiguousarray(rhoa_full[0,:,ip0:ip1])
        rho_b = np.ascontiguousarray(rhob_full[0,:,ip0:ip1])
        wva = np.ascontiguousarray(wva_full[0][:,ip0:ip1])
        wvb = np.ascontiguousarray(wvb_full[0][:,ip0:ip1])
        vtmp = np.zeros((2,3,nao,nao))
        _gga_grad_sum_(vtmp[0], mol, ao, wva[:4], mask, ao_loc)
        _gga_grad_sum_(vtmp[1], mol, ao, wvb[:4], mask, ao_loc)
        if xctype == 'MGGA':
            wva[5] *= 2.0
            wvb[5] *= 2.0
            _tau_grad_dot_(vtmp[0], mol, ao, wva[5], mask, ao_loc, True)
            _tau_grad_dot_(vtmp[1], mol, ao, wvb[5], mask, ao_loc, True)
        excsum += np.einsum('r,r,nxr->nx', exc[ip0:ip1], rho_a[0]+rho_b[0], weight1)
        vmat += vtmp
        excsum[atm_id] += np.einsum('sxij,sji->x', vtmp, dms) * 2
        excsum += np.dot(weight1, cidergg_g[ip0:ip1])
        ip0 = ip1

    excsum += excsum_tmp
    return excsum, -vmat


class Gradients(uks_grad.Gradients):

    get_veff = get_veff


class DFGradients(uks_grad_df.Gradients):

    get_veff = get_veff
