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

from ciderpress.dft.cider_kernel import get_exponent_b, get_exponent_d
import numpy as np
import time
from pyscf.dft.numint import NumInt, _contract_rho, _rks_gga_wv0, \
    _uks_gga_wv0, _rks_mgga_wv0, _uks_mgga_wv0, _tau_dot, _format_uks_dm, \
    _dot_ao_ao_sparse, _scale_ao_sparse, _tau_dot_sparse, NBINS, \
    _scale_ao, _dot_ao_ao


CIDER_NUM_TOL = 1e-8


def get_full_flapl(ni, mol, dms, grids, xctype):
    hermi = 1
    max_memory = 2000
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_deriv = 0
    Nrhofeat = 3
    rho_full = np.zeros((nset, Nrhofeat, grids.weights.size), dtype=np.float64, order='C')
    ip0 = 0
    for ao, kao, mask, weight, coords \
            in ni.block_loop_flapl(mol, grids, nao, ao_deriv, max_memory):
        mask = None  # TODO remove once screening is fixed
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rho0 = make_rho(idm, ao, mask, 'LDA')
            rho1 = make_rho(idm, kao, mask, 'LDA')
            rho2 = make_rho(idm, ao + kao, mask, 'LDA')
            rho2 -= rho0 + rho1
            rho2 *= 0.5
            rho_full[idm, 0, ip0:ip1] = rho0
            rho_full[idm, 1, ip0:ip1] = rho1
            rho_full[idm, 2, ip0:ip1] = rho2
        ip0 = ip1
    return rho_full


def get_full_flapl_grad(ni, mol, wv_full, grids, nset, nbins, pair_mask,
                        ao_loc, vmat):
    ip0 = 0
    nao = mol.nao_nr()
    for ao, kao, mask, weight, coords \
            in ni.block_loop_flapl(mol, grids, nao, 0, 2000):
        mask = None # TODO remove once screening is fixed
        ip1 = ip0 + weight.size
        for idm in range(nset):
            wv = np.ascontiguousarray(wv_full[idm][:, ip0:ip1])
            aow = _scale_ao_sparse(kao, wv[0], mask, ao_loc)
            aow += _scale_ao_sparse(ao, wv[1], mask, ao_loc)
            _dot_ao_ao_sparse(
                kao, aow, None, nbins, mask, pair_mask, ao_loc,
                hermi=0, out=vmat[idm],
            )


def get_feat_convolutions_fwd(ni, mol, grids, p_ag, rho_full, ap_ag):
    p_ag = p_ag * grids.weights
    ap_ag = ap_ag * grids.weights
    p_ag[:, rho_full[0] < CIDER_NUM_TOL] = 0
    ap_ag[:, rho_full[0] < CIDER_NUM_TOL] = 0
    if not ni.ri_conv.is_num_ai_setup:
        ni.ri_conv.set_num_ai(grids)
    if ni.ri_conv.onsite_direct:
        f_qarlp, f_gq = ni.ri_conv.transform_orbital_feat_fwd(
            grids, np.ascontiguousarray(p_ag.T),
            np.ascontiguousarray(ap_ag.T),
        )
    else:
        f_qarlp = ni.ri_conv.transform_orbital_feat_fwd(
            grids, np.ascontiguousarray(p_ag.T),
            np.ascontiguousarray(ap_ag.T),
        )
        f_gq = np.zeros((ni.ri_conv.all_coords.shape[0], f_qarlp.shape[0]))
    f_arqlp = np.ascontiguousarray(f_qarlp.transpose(1, 2, 0, 3, 4))
    aotst = np.empty((ni.ri_conv.coords_ord.shape[0], ni.ri_conv.nlm * 4))
    for a in range(mol.natm):
        aotst, ind_g = ni.ri_conv.eval_spline_bas_single(a, out=aotst)
        ni.ri_conv.compute_mol_convs_single_(a, f_arqlp[a], f_gq, aotst, ind_g)
    return f_gq, f_arqlp


def get_feat_convolutions_fwd2(ni, mol, grids, p_ag, rho_full):
    p_ag = p_ag * grids.weights
    p_ag[:, rho_full[0] < CIDER_NUM_TOL] = 0
    if not ni.ri_conv.is_num_ai_setup:
        ni.ri_conv.set_num_ai(grids)
    if ni.ri_conv.onsite_direct:
        f_qarlp, f_gq = ni.ri_conv.transform_orbital_feat_fwd(
            grids, np.ascontiguousarray(p_ag.T),
        )
    else:
        f_qarlp = ni.ri_conv.transform_orbital_feat_fwd(
            grids, np.ascontiguousarray(p_ag.T),
        )
        f_gq = np.zeros((ni.ri_conv.all_coords.shape[0], f_qarlp.shape[0]))
    f_arqlp = np.ascontiguousarray(f_qarlp.transpose(1, 2, 0, 3, 4))
    aotst = np.empty((ni.ri_conv.coords_ord.shape[0], ni.ri_conv.nlm * 4))
    for a in range(mol.natm):
        aotst, ind_g = ni.ri_conv.eval_spline_bas_single(a, out=aotst)
        ni.ri_conv.compute_mol_convs_single_(a, f_arqlp[a], f_gq, aotst, ind_g)
    return f_gq, f_arqlp


def get_feat_convolutions_bwd(ni, mol, grids, f_gq, rho, f_arqlp_buf,
                              p_ag, dp_ag):
    atime = time.monotonic()
    f_gq = np.ascontiguousarray(f_gq)
    aotst = np.empty((ni.ri_conv.coords_ord.shape[0], ni.ri_conv.nlm * 4))
    f_arqlp_buf[:] = 0.0
    for a in range(mol.natm):
        aotst, ind_g = ni.ri_conv.eval_spline_bas_single(a, out=aotst)
        ni.ri_conv.compute_mol_convs_single_(
            a, f_gq, f_arqlp_buf[a], aotst, ind_g, pot=True
        )
    vbas_qarlp = np.ascontiguousarray(f_arqlp_buf.transpose(2, 0, 1, 3, 4))
    if ni.ri_conv.onsite_direct:
        v_gq, av_gq = ni.ri_conv.transform_orbital_feat_bwd(
            grids, vbas_qarlp, f_gq=f_gq
        )
    else:
        v_gq, av_gq = ni.ri_conv.transform_orbital_feat_bwd(grids, vbas_qarlp)
    v_gq[rho[0] < CIDER_NUM_TOL] = 0
    av_gq[rho[0] < CIDER_NUM_TOL] = 0
    dedrho = np.einsum('gq,gq->g', v_gq, p_ag.T)
    dedtau = np.einsum('gq,gq->g', av_gq, p_ag.T)
    deda = np.einsum('gq,gq->g', v_gq, dp_ag.T) * rho[0]
    deda += np.einsum('gq,gq->g', av_gq, dp_ag.T) * rho[5]
    return dedrho, dedtau, deda


def get_feat_convolutions_bwd2(ni, mol, grids, f_gq, rho, f_arqlp_buf, p_ag, dp_ag):
    atime = time.monotonic()
    f_gq = np.ascontiguousarray(f_gq)
    aotst = np.empty((ni.ri_conv.coords_ord.shape[0], ni.ri_conv.nlm * 4))
    f_arqlp_buf[:] = 0.0
    for a in range(mol.natm):
        aotst, ind_g = ni.ri_conv.eval_spline_bas_single(a, out=aotst)
        ni.ri_conv.compute_mol_convs_single_(
            a, f_gq, f_arqlp_buf[a], aotst, ind_g, pot=True
        )
    vbas_qarlp = np.ascontiguousarray(f_arqlp_buf.transpose(2, 0, 1, 3, 4))
    if ni.ri_conv.onsite_direct:
        v_gq = ni.ri_conv.transform_orbital_feat_bwd(
            grids, vbas_qarlp, f_gq=f_gq
        )
    else:
        v_gq = ni.ri_conv.transform_orbital_feat_bwd(grids, vbas_qarlp)
    v_gq[rho[0] < CIDER_NUM_TOL] = 0
    dedrho = np.einsum('gq,gq->g', v_gq, p_ag.T)
    deda = np.einsum('gq,gq->g', v_gq, dp_ag.T) * rho[0]
    return dedrho, deda


def get_wv_rks_cider_vh(ni, mol, grids, xc_code, rho,
                        relativity, hermi, verbose=None,
                        grad_mode=False, taux=None):
    if taux is not None:
        assert ni.uses_flapl
        flapl_rho = taux
    else:
        flapl_rho = None
    if grad_mode:
        raise NotImplementedError
    wtime = time.monotonic()
    is_gga = (ni._xc_type(xc_code) == 'GGA')
    if not is_gga:
        rhop = np.empty((6, rho.shape[-1]))
        rhop[:4] = rho[:4]
        rhop[4] = 0
        rhop[5] = rho[4]
        rho = rhop

    mlfunc = ni.mlfunc_x
    vv_gg_settings = {
        'get_exponent': get_exponent_b,
        'a0': mlfunc.a0,
        'fac_mul': mlfunc.fac_mul,
        'amin': mlfunc.amin,
    }
    gg_scales = [vv_gg_settings['a0'] ** 1.5]

    weight, coords = grids.weights, grids.coords
    Nalpha = len(ni.ri_conv.alphas)
    nfeat = 7
    tmp = ni.ri_conv.get_cider_coefs_fwd(rho, derivs='wv', **vv_gg_settings)
    p_ag, dp_ag, cider_exp, dadn, dadsigma, dadtau = tmp
    if mlfunc.desc_version == 'h':
        theta = p_ag * rho[0]
        atheta = p_ag * rho[5]
        f_gq, f_arqlp = get_feat_convolutions_fwd(
            ni, mol, grids, theta, rho, atheta,
        )
        feat_i, f_xg, ft_xg, fxdx, fxtdxt, fxdxt = \
            ni.ri_conv.get_feat_from_f(f_gq.T, rho, mlfunc.a0, cider_exp)
    elif mlfunc.desc_version == 'i':
        theta = p_ag * rho[0]
        f_gq, f_arqlp = get_feat_convolutions_fwd2(
            ni, mol, grids, theta, rho,
        )
        feat_i, f_xg, fxdx, fxdxt = \
            ni.ri_conv.get_feat_from_f(
                f_gq.T, rho, mlfunc.a0, cider_exp, flapl_rho
            )
    else:
        theta = p_ag * rho[0]
        f_gq, f_arqlp = get_feat_convolutions_fwd2(
            ni, mol, grids, theta, rho,
        )
        feat_i, dfeat_i, p_iqg = (
            ni.ri_conv.get_feat_from_f(
                f_gq.T, rho, vv_gg_settings, flapl_rho
            )[:3]
        )
    exc, vxc = ni.eval_xc(xc_code, mol, rho, feat_i,
                          0, relativity, 1, verbose=verbose)[:2]
    vrho, vsigma, vfeat_i, vtau = vxc
    den = rho[0] * weight
    nelec = den.sum()
    excsum = np.dot(den, exc)

    if mlfunc.desc_version == 'h':
        vf_qg, deda, dedrho, dedgrad = ni.ri_conv.get_vfeat_scf(
            vfeat_i[0], rho, mlfunc.a0, cider_exp, feat_i,
            f_gq.T, f_xg, ft_xg, fxdx, fxtdxt
        )
        vf_qg *= weight
        dedrho_tmp, dedtau, deda_tmp = get_feat_convolutions_bwd(
            ni, mol, grids, vf_qg.T, rho, f_arqlp, p_ag, dp_ag
        )
        deda += deda_tmp
        dedrho += dedrho_tmp
        vxc[3][:] += dedtau
    elif mlfunc.desc_version == 'i':
        vf_qg, deda, dedrho, dedgrad, vflapl = ni.ri_conv.get_vfeat_scf(
            vfeat_i[0], rho, mlfunc.a0, cider_exp, feat_i,
            f_gq.T, f_xg, flapl_rho
        )
        vf_qg *= weight
        dedrho_tmp, deda_tmp = get_feat_convolutions_bwd2(
            ni, mol, grids, vf_qg.T, rho, f_arqlp, p_ag, dp_ag
        )
        deda += deda_tmp
        dedrho += dedrho_tmp
    else:
        vf_qg, deda = ni.ri_conv.get_vfeat_scf(
            vfeat_i[0], dfeat_i, p_iqg, vv_gg_settings
        )
        vf_qg *= weight
        dedrho, deda_tmp = get_feat_convolutions_bwd2(
            ni, mol, grids, vf_qg.T, rho, f_arqlp, p_ag, dp_ag
        )
        deda += deda_tmp
    #if grad_mode:
    #    vxc_cider, cidergg_g = vxc_cider
    vxc[0][:] += deda * dadn
    vxc[1][:] += deda * dadsigma
    vxc[3][:] += deda * dadtau
    vxc[0][:] += dedrho
    wv = _rks_mgga_wv0(rho, vxc, weight)
    if mlfunc.desc_version != 'k':
        wv[1:4] += weight * dedgrad
    if ni.uses_flapl:
        # TODO factor of 2 might be off
        wv = np.concatenate([wv, 0.5 * vflapl * weight], axis=0)
    #print('WTIME', grad_mode, time.monotonic() - wtime)

    #if grad_mode:
    #    return cidergg_g, excsum, wv, exc
    #else:
    #    return nelec, excsum, wv, exc
    return nelec, excsum, wv, exc


def get_wv_uks_cider_vh(ni, mol, grids, xc_code, rho,
                        relativity, hermi, verbose=None,
                        grad_mode=False, taux=None):
    if taux is not None:
        assert ni.uses_flapl
        flapl_rho = taux
    else:
        flapl_rho = None
    if grad_mode:
        raise NotImplementedError
    is_gga = (ni._xc_type(xc_code) == 'GGA')
    if not is_gga:
        rhop = np.empty((2, 6, rho[0].shape[-1]))
        for s in range(2):
            rhop[s, :4] = rho[s][:4]
            rhop[s, 4] = 0
            rhop[s, 5] = rho[s][4]
        rho = rhop

    mlfunc = ni.mlfunc_x
    vv_gg_settings = {
        'get_exponent': get_exponent_b,
        'a0': mlfunc.a0,
        'fac_mul': mlfunc.fac_mul,
        'amin': mlfunc.amin,
    }
    gg_scales = [vv_gg_settings['a0'] ** 1.5]
    weight, coords = grids.weights, grids.coords
    rho_a, rho_b = rho
    nfeat = 7
    Nalpha = len(ni.ri_conv.alphas)
    vv_gg_settings['nspin'] = 2

    tmp = ni.ri_conv.get_cider_coefs_fwd(rho_a, derivs='wv', **vv_gg_settings)
    pa_ag, dpa_ag, a_a, dadn_a, dadsigma_a, dadtau_a = tmp
    tmp = ni.ri_conv.get_cider_coefs_fwd(rho_b, derivs='wv', **vv_gg_settings)
    pb_ag, dpb_ag, a_b, dadn_b, dadsigma_b, dadtau_b = tmp
    if mlfunc.desc_version == 'h':
        theta = pa_ag * rho_a[0]
        atheta = pa_ag * rho_a[5]
        fa_gq, fa_arqlp = get_feat_convolutions_fwd(
            ni, mol, grids, theta, rho_a, atheta,
        )
        theta = pb_ag * rho_b[0]
        atheta = pb_ag * rho_b[5]
        fb_gq, fb_arqlp = get_feat_convolutions_fwd(
            ni, mol, grids, theta, rho_b, atheta,
        )
        feata_i, fa_xg, fta_xg, fxdx_a, fxtdxt_a, fxdxt_a = \
            ni.ri_conv.get_feat_from_f(fa_gq.T, rho_a, mlfunc.a0, a_a, nspin=2)
        featb_i, fb_xg, ftb_xg, fxdx_b, fxtdxt_b, fxdxt_b = \
            ni.ri_conv.get_feat_from_f(fb_gq.T, rho_b, mlfunc.a0, a_b, nspin=2)
    elif mlfunc.desc_version == 'i':
        theta = pa_ag * rho_a[0]
        fa_gq, fa_arqlp = get_feat_convolutions_fwd2(
            ni, mol, grids, theta, rho_a,
        )
        theta = pb_ag * rho_b[0]
        fb_gq, fb_arqlp = get_feat_convolutions_fwd2(
            ni, mol, grids, theta, rho_b,
        )
        feata_i, fa_xg, fxdx_a, fgdx_a = \
            ni.ri_conv.get_feat_from_f(
                fa_gq.T, rho_a, mlfunc.a0, a_a, flapl_rho[0], nspin=2
            )
        featb_i, fb_xg, fxdx_b, fgdx_b = \
            ni.ri_conv.get_feat_from_f(
                fb_gq.T, rho_b, mlfunc.a0, a_b, flapl_rho[1], nspin=2
            )
    else:
        theta = pa_ag * rho_a[0]
        fa_gq, fa_arqlp = get_feat_convolutions_fwd2(
            ni, mol, grids, theta, rho_a,
        )
        theta = pb_ag * rho_b[0]
        fb_gq, fb_arqlp = get_feat_convolutions_fwd2(
            ni, mol, grids, theta, rho_b,
        )
        feata_i, dfeata_i, pa_iqg = (
            ni.ri_conv.get_feat_from_f(fa_gq.T, rho_a, vv_gg_settings, nspin=2)[:3]
        )
        featb_i, dfeatb_i, pb_iqg = (
            ni.ri_conv.get_feat_from_f(fb_gq.T, rho_b, vv_gg_settings, nspin=2)[:3]
        )

    feat_i = np.stack([feata_i, featb_i])
    exc, vxc = ni.eval_xc(xc_code, mol, np.stack([rho_a, rho_b]), feat_i,
                          1, relativity, 1, verbose=verbose)[:2]
    den = (rho_a[0] + rho_b[0]) * weight
    nelec = den.sum()
    excsum = np.dot(den, exc)

    vgrad = {}
    vrho, vsigma, vfeat_i, vtau = vxc

    if mlfunc.desc_version == 'h':
        vf_qg, deda, dedrho, dedgrad = ni.ri_conv.get_vfeat_scf(
            vfeat_i[0], rho_a, mlfunc.a0, a_a, feat_i[0],
            fa_gq.T, fa_xg, fta_xg, fxdx_a, fxtdxt_a, nspin=2
        )
        vf_qg *= weight
        dedrho_tmp, dedtau, deda_tmp = get_feat_convolutions_bwd(
            ni, mol, grids, vf_qg.T, rho_a, fa_arqlp, pa_ag, dpa_ag
        )
        deda += deda_tmp
        dedrho += dedrho_tmp
        vxc[3][:, 0] += dedtau
    elif mlfunc.desc_version == 'i':
        vf_qg, deda, dedrho, dedgrad, vflapla = ni.ri_conv.get_vfeat_scf(
            vfeat_i[0], rho_a, mlfunc.a0, a_a, feat_i[0],
            fa_gq.T, fa_xg, flapl_rho[0], nspin=2
        )
        vf_qg *= weight
        dedrho_tmp, deda_tmp = get_feat_convolutions_bwd2(
            ni, mol, grids, vf_qg.T, rho_a, fa_arqlp, pa_ag, dpa_ag
        )
        deda += deda_tmp
        dedrho += dedrho_tmp
    else:
        vf_qg, deda = ni.ri_conv.get_vfeat_scf(
            vfeat_i[0], dfeata_i, pa_iqg, vv_gg_settings, nspin=2
        )
        vf_qg *= weight
        dedrho, deda_tmp = get_feat_convolutions_bwd2(
            ni, mol, grids, vf_qg.T, rho_a, fa_arqlp, pa_ag, dpa_ag
        )
        deda += deda_tmp
    vxc[0][:, 0] += deda * dadn_a
    vxc[1][:, 0] += deda * dadsigma_a
    vxc[3][:, 0] += deda * dadtau_a
    vxc[0][:, 0] += dedrho
    if mlfunc.desc_version != 'k':
        vgrad[0] = dedgrad

    if mlfunc.desc_version == 'h':
        vf_qg, deda, dedrho, dedgrad = ni.ri_conv.get_vfeat_scf(
            vfeat_i[1], rho_b, mlfunc.a0, a_b, feat_i[1],
            fb_gq.T, fb_xg, ftb_xg, fxdx_b, fxtdxt_b, nspin=2
        )
        vf_qg *= weight
        dedrho_tmp, dedtau, deda_tmp = get_feat_convolutions_bwd(
            ni, mol, grids, vf_qg.T, rho_b, fb_arqlp, pb_ag, dpb_ag
        )
        deda += deda_tmp
        dedrho += dedrho_tmp
        vxc[3][:, 1] += dedtau
    elif mlfunc.desc_version == 'i':
        vf_qg, deda, dedrho, dedgrad, vflaplb = ni.ri_conv.get_vfeat_scf(
            vfeat_i[1], rho_b, mlfunc.a0, a_b, feat_i[1],
            fb_gq.T, fb_xg, flapl_rho[1], nspin=2
        )
        vf_qg *= weight
        dedrho_tmp, deda_tmp = get_feat_convolutions_bwd2(
            ni, mol, grids, vf_qg.T, rho_b, fb_arqlp, pb_ag, dpb_ag
        )
        deda += deda_tmp
        dedrho += dedrho_tmp
    else:
        vf_qg, deda = ni.ri_conv.get_vfeat_scf(
            vfeat_i[1], dfeatb_i, pb_iqg, vv_gg_settings, nspin=2
        )
        vf_qg *= weight
        dedrho, deda_tmp = get_feat_convolutions_bwd2(
            ni, mol, grids, vf_qg.T, rho_b, fb_arqlp, pb_ag, dpb_ag
        )
        deda += deda_tmp
    vxc[0][:, 1] += deda * dadn_b
    vxc[1][:, 2] += deda * dadsigma_b
    vxc[3][:, 1] += deda * dadtau_b
    vxc[0][:, 1] += dedrho
    if mlfunc.desc_version != 'k':
        vgrad[1] = dedgrad

    wva, wvb = _uks_mgga_wv0((rho_a, rho_b), vxc, weight)
    if mlfunc.desc_version != 'k':
        wva[1:4] += weight * vgrad[0]
        wvb[1:4] += weight * vgrad[1]
    if ni.uses_flapl:
        # TODO factor of 2 might be off
        wva = np.concatenate([wva, 0.5 * vflapla * weight], axis=0)
        wvb = np.concatenate([wvb, 0.5 * vflaplb * weight], axis=0)
    # print('WTIME', grad_mode, time.monotonic() - wtime)

    # if grad_mode:
    #    return cidergg_g, excsum, wv, exc
    # else:
    #    return nelec, excsum, wv, exc
    return nelec, excsum, wva, wvb, exc
