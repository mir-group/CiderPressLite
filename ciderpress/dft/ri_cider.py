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

from pyscf import dft, df, lib
from pyscf.scf import hf
from pyscf.dft.numint import NumInt, _contract_rho, _rks_gga_wv0, \
    _uks_gga_wv0, _rks_mgga_wv0, _uks_mgga_wv0, _tau_dot, _format_uks_dm, \
    _dot_ao_ao_sparse, _scale_ao_sparse, _tau_dot_sparse, NBINS, \
    _scale_ao, _dot_ao_ao
from pyscf.dft.libxc import eval_xc, is_nlc, is_gga

from ciderpress.dft.cider_kernel import call_xc_kernel_gga, call_xc_kernel_mgga, \
    call_xc_kernel_mmgga, get_exponent_d, get_exponent_b
from ciderpress.dft.cider_conv import CiderConvSpline, CiderConvSplineG, \
    CiderConvSplineH, CiderConvSplineHv2, CiderConvSplineI, CIDER_DEFAULT_LMAX, \
    CiderConv2Spline, CiderConvSplineK
from ciderpress.dft.xc_models import NormGPFunctional
from ciderpress.dft.xc_evaluator import MappedFunctional, \
    GGAFunctionalEvaluator, MGGAFunctionalEvaluator
from ciderpress.dft.vh_cider import get_wv_rks_cider_vh, get_wv_uks_cider_vh, \
    get_full_flapl, get_full_flapl_grad
from ciderpress.dft.flapl_util import eval_kao
from pyscf.dft.numint import BLKSIZE, _sparse_enough, _empty_aligned
from pyscf.dft.gen_grid import ALIGNMENT_UNIT


import numpy as np
import numpy
import time
import joblib


CIDER_NUM_TOL = 1e-8
CIDER_DEFAULT_VV10_COEFF = [6.0, 0.01]


def setup_cider_calc(
    mol, mlfunc, mlfunc_format=None,
    spinpol=False, xc=None,
    xkernel=None, ckernel=None,
    vv10_coeff=None, grid_level=3,
    xmix=1.0, debug=False,
    _force_nonlocal=None,
    **kwargs
):
    """
    Initiliaze an RKS or UKS calculation with a CIDER exchange
    functional. If xc, xkernel, ckernel, and xmix are not specified,
    The equivalent of HF with CIDER in place of EXX is performed.
    The XC energy is

    E_xc = xmix * E_x^CIDER + (1-xmix) * xkernel + ckernel + xc

    NOTE: Only GGA-level XC functionals can be used with GGA-level
        (orbital-independent) CIDER functionals currently.

    Args:
        mol (pyscf.gto.Mole): molecule object
        mlfunc (NormGPFunctional, str): CIDER exchange functional or file name
        mlfunc_format (str, None): 'joblib' or 'yaml', specifies the format
            of mlfunc if it is a string corresponding to a file name.
            If unspecified, infer from file extension and raise error
            if file type cannot be determined.
        spinpol (bool, False): If True, UKS is used. If False, RKS is used.
        xc (str, None): If specified, this semi-local XC code is evaluated
             and added to the total XC energy.
        xkernel (str, None): Semi-local X code in libxc. Scaled by (1-xmix).
        ckernel (str, None): Semi-local C code in libxc.
        vv10_coeff (tuple of 2 ints, None):
            VV10 coefficients. If None, VV10 term is not evaluated.
        grid_level (int, 3): PySCF grid level to use
        xmix (float): Fraction of CIDER exchange used.
        debug (bool): FOR TESTING ONLY. Uses PBE X potential instead
            of CIDER potential, but still uses CIDER energy.
        _force_nonlocal (bool, False): FOR TESTING ONLY. Default
            None is preferred. If _force_nonlocal is True, nonlocal
            feature generation is run even if the CIDER functional
            is semi-local.
        **kwargs: Optional arguments for the CIDER feature evaluation.
              Some of these are important, and the defaults might not
              be sufficient for some systems.
            amax (float, 3000): Maximum exponent for fitting CIDER
                kernel function. Needs to be increased for some larger
                atoms. amax=(Z^2)*1000/36 is a good, conservative estimate.
            amin (float, None): Minimum exponent for fitting CIDER
                kernel function. Typicially set automatically and need
                not be tuned.
            lambd (float, 1.6): ETB parameter for CIDER kernel function
                expansion. 1.6 is good for most cases.
            aux_beta (float, 1.8): ETB parameter for fitting theta
                components that are integrated by the CIDER kernel
                function. 1.8 is good for most cases; use 1.6 if high
                numerical precision is desired. Small lambd
                can cause numerical instability in some cases.

    Returns:
        An RKS or UKS object, depending on spinpol
    """

    if isinstance(mlfunc, str):
        if mlfunc_format is None:
            if mlfunc.endswith('.yaml'):
                mlfunc_format = 'yaml'
            elif mlfunc.endswith('.joblib'):
                mlfunc_format = 'joblib'
            else:
                raise ValueError('Unsupported file format')
        if mlfunc_format == 'yaml':
            mlfunc = NormGPFunctional.load(mlfunc)
        elif mlfunc_format == 'joblib':
            mlfunc = joblib.load(mlfunc)
        else:
            raise ValueError('Unsupported file format')
    print(type(mlfunc))
    if not isinstance(mlfunc, (NormGPFunctional, MappedFunctional)):
        raise ValueError('mlfunc must be NormGPFunctional')

    if (((mlfunc.desc_version == 'b' and mlfunc.feature_list.nfeat == 2) or
         (mlfunc.desc_version == 'd' and mlfunc.feature_list.nfeat == 1))
         and not _force_nonlocal):
        semilocal = True
    else:
        semilocal = False
    if spinpol:
        if semilocal:
            ks_cls = dft.uks.UKS
        else:
            ks_cls = CiderUKS
    else:
        if semilocal:
            ks_cls = dft.rks.RKS
        else:
            ks_cls = CiderRKS
    cls = CiderNumIntDebug if debug else CiderNumInt
    if semilocal:
        cls = SLCiderNumInt
    ks = ks_cls(mol)
    ks.xc = ''
    if xc is not None:
        # xc is another way to specify non-mixed part of kernel
        if ckernel is not None:
            ckernel = ckernel + ' + ' + xc
        else:
            ckernel = xc
    ks._numint = cls(mol, mlfunc, xkernel, ckernel, vv10_coeff, xmix,
                     **kwargs)
    if semilocal:
        ks.grids.level = grid_level
        ks.grids.build()
        return ks
    from ciderpress.dft.cider_grid import CiderGrids
    ks.small_rho_cutoff = 0
    ks.grids = CiderGrids(ks.mol, ks._numint._cider_lmax)
    ks.grids.level = grid_level
    ks.grids.build()
    return ks


def _block_loop_flapl(ni, mol, grids, nao=None, deriv=0, max_memory=2000,
                      non0tab=None, blksize=None, buf=None, buf2=None):
    '''Define this macro to loop over grids by blocks.
    '''
    if grids.coords is None:
        grids.build(with_non0tab=True)
    if nao is None:
        nao = mol.nao
    ngrids = grids.coords.shape[0]
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    # NOTE to index grids.non0tab, the blksize needs to be an integer
    # multiplier of BLKSIZE
    if blksize is None:
        blksize = int(max_memory*1e6/((comp+1)*nao*8*BLKSIZE))
        blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 1200)) * BLKSIZE
    assert blksize % BLKSIZE == 0

    if non0tab is None and mol is grids.mol:
        non0tab = grids.non0tab
    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                              dtype=numpy.uint8)
        non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
    screen_index = non0tab

    # the xxx_sparse() functions require ngrids 8-byte aligned
    allow_sparse = ngrids % ALIGNMENT_UNIT == 0

    if buf is None:
        buf = _empty_aligned(comp * blksize * nao)
    if buf2 is None:
        buf2 = _empty_aligned(comp * blksize * nao)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        mask = screen_index[ip0//BLKSIZE:]
        # TODO: pass grids.cutoff to eval_ao
        ao = ni.eval_ao(mol, coords, deriv=deriv, non0tab=mask,
                        cutoff=grids.cutoff, out=buf)
        kao = eval_kao(mol, coords, deriv=deriv, non0tab=mask,
                       cutoff=grids.cutoff, out=buf2)
        if not allow_sparse and not _sparse_enough(mask):
            # Unset mask for dense AO tensor. It determines which eval_rho
            # to be called in make_rho
            mask = None
        yield ao, kao, mask, weight, coords


class CiderGGAHybridKernel:

    call_xc_kernel = call_xc_kernel_gga

    def __init__(self, mlfunc, xmix):
        self.mlfunc = mlfunc
        self.xmix = xmix
        if isinstance(self.mlfunc, MappedFunctional):
            self.call_xc_kernel = GGAFunctionalEvaluator(self.mlfunc, amix=xmix)

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, feat_sg):
        # assumes all variables are initialized and uses +=
        vfeat_sg = np.zeros_like(feat_sg)
        self.call_xc_kernel(
            e_g, n_sg, sigma_xg, feat_sg,
            v_sg, dedsigma_xg, vfeat_sg,
        )
        return vfeat_sg


class CiderMGGAHybridKernel:

    call_xc_kernel = call_xc_kernel_mgga

    def __init__(self, mlfunc, xmix):
        self.mlfunc = mlfunc
        self.xmix = xmix
        if isinstance(self.mlfunc, MappedFunctional):
            self.call_xc_kernel = MGGAFunctionalEvaluator(self.mlfunc, amix=xmix)

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg, feat_sg):
        vfeat_sg = np.zeros_like(feat_sg)
        self.call_xc_kernel(
            e_g, n_sg, sigma_xg, tau_sg, feat_sg,
            v_sg, dedsigma_xg, dedtau_sg, vfeat_sg,
        )
        return vfeat_sg


def _rks_mmgga_wv0(rho, vxc, vtaux, weight):
    vrho, vgamma, vlapl, vtau = vxc[:4]
    ngrid = vrho.size
    wv = numpy.zeros((7,ngrid))
    wv[0] = weight * vrho
    wv[1:4] = (weight * vgamma * 2) * rho[1:4]
    # *0.5 is for tau = 1/2 \nabla\phi\dot\nabla\phi
    wv[5] = weight * vtau * .5
    # *0.5 because v+v.T should be applied in the caller
    wv[0] *= .5
    wv[5] *= .5
    wv[6] = weight * vtaux
    return wv


def _uks_mmgga_wv0(rho, vxc, vtaux, weight):
    rhoa, rhob = rho
    vrho, vsigma, vlapl, vtau = vxc
    ngrid = vrho.shape[0]
    wva, wvb = numpy.zeros((2,7,ngrid))
    wva[0] = vrho[:,0] * .5  # v+v.T should be applied in the caller
    wva[1:4] = rhoa[1:4] * vsigma[:,0] * 2  # sigma_uu
    wva[1:4]+= rhob[1:4] * vsigma[:,1]      # sigma_ud
    wva[5] = vtau[:,0] * .25
    wva[6] = vtaux[:,0]
    wva *= weight
    wvb[0] = vrho[:,1] * .5  # v+v.T should be applied in the caller
    wvb[1:4] = rhob[1:4] * vsigma[:,2] * 2  # sigma_dd
    wvb[1:4]+= rhoa[1:4] * vsigma[:,1]      # sigma_ud
    wvb[5] = vtau[:,1] * .25
    wvb[6] = vtaux[:,1]
    wvb *= weight
    return wva, wvb


class CiderMMGGAHybridKernel:

    def __init__(self, mlfunc, xmix):
        self.mlfunc = mlfunc
        self.xmix = xmix

    call_xc_kernel = call_xc_kernel_mmgga

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg,
                  feat_sg, taux_sg):
        vfeat_sg = np.zeros_like(feat_sg)
        vtaux_sg = np.zeros_like(taux_sg)
        self.call_xc_kernel(
            e_g, n_sg, sigma_xg, tau_sg, feat_sg, taux_sg,
            v_sg, dedsigma_xg, dedtau_sg, vfeat_sg, vtaux_sg,
        )
        return vfeat_sg, vtaux_sg


def get_gg_settings(mlfunc, is_gga):
    get_expnt = get_exponent_d if is_gga else get_exponent_b
    gg_settings = [{
        'get_exponent': get_expnt,
        'a0': 2**i * mlfunc.a0 / 2,
        'fac_mul': 2**i * mlfunc.fac_mul / 2,
        'amin': 2**i * mlfunc.amin / 2
    } for i in range(3)]
    vvmul = mlfunc.vvmul
    vv_gg_settings = {
        'get_exponent': get_expnt,
        'a0': mlfunc.a0 * vvmul,
        'fac_mul': mlfunc.fac_mul * vvmul,
        'amin': mlfunc.amin * vvmul
    }
    return gg_settings, vv_gg_settings, \
           [((gg_settings[i]['a0'] + vv_gg_settings['a0']) / 2)**1.5 \
            for i in range(3)]


def get_feat_convolutions_fwd(ni, mol, grids, rho_full,
                              vv_gg_settings, debug_feat=False):
    atime = time.monotonic()
    p_ag = ni.ri_conv.get_cider_coefs_fwd(rho_full, **vv_gg_settings)[0]
    nalpha = p_ag.shape[0]
    if debug_feat:
        p_ag_full = p_ag*rho_full[0]
    p_ag *= rho_full[0] * grids.weights
    p_ag[:,rho_full[0]<CIDER_NUM_TOL] = 0
    # TODO compute as p_ga from the outset to preserve memory
    if not ni.ri_conv.is_num_ai_setup:
        ni.ri_conv.set_num_ai(grids)
    if ni.ri_conv.onsite_direct:
        f_qarlp, f_gq = ni.ri_conv.transform_orbital_feat_fwd(
            grids, np.ascontiguousarray(p_ag.T)
        )
    else:
        f_qarlp = ni.ri_conv.transform_orbital_feat_fwd(
            grids, np.ascontiguousarray(p_ag.T)
        )
        f_gq = np.zeros((ni.ri_conv.all_coords.shape[0], nalpha))

    f_arqlp = np.ascontiguousarray(f_qarlp.transpose(1,2,0,3,4))
    aotst = np.empty((ni.ri_conv.coords_ord.shape[0], ni.ri_conv.nlm*4))
    btime, ctime = 0, 0
    for a in range(mol.natm):
        t0 = time.monotonic()
        aotst, ind_g = ni.ri_conv.eval_spline_bas_single(a, out=aotst)
        t1 = time.monotonic()
        ni.ri_conv.compute_mol_convs_single_(a, f_arqlp[a], f_gq, aotst, ind_g)
        t2 = time.monotonic()
        btime += t1 - t0
        ctime += t2 - t1

    if debug_feat:
        return f_gq, p_ag_full
    else:
        return f_gq, f_arqlp

def get_grad_convolutions(ni, mol, f_arqlp, f_gq):
    aotst_buf = np.empty((3, ni.ri_conv.coords_ord.shape[0], ni.ri_conv.nlm*4))
    excsum = np.zeros((mol.natm, 3))
    f_gq = np.ascontiguousarray(f_gq)
    ftmp_gq = np.empty_like(f_gq)
    for a in range(mol.natm):
        aotst, ind_g = ni.ri_conv.eval_spline_bas_single_grad(a, out=aotst_buf)
        for v in range(3):
            ftmp_gq[:] = 0.0
            ni.ri_conv.compute_mol_convs_single_(
                a, f_arqlp[a], ftmp_gq, aotst[v], ind_g
            )
            ftmp = _contract_rho(ftmp_gq, f_gq)
            ni.ri_conv.contract_grad_terms(excsum, ftmp, a, v)
    return excsum

def get_feat_convolutions_bwd(ni, mol, grids, rho, f_gq, vv_gg_settings,
                              f_arqlp_buf, is_gga=True, grad_mode=False):
    atime = time.monotonic()
    aotst = np.empty((ni.ri_conv.coords_ord.shape[0], ni.ri_conv.nlm*4))
    for a in range(mol.natm):
        aotst, ind_g = ni.ri_conv.eval_spline_bas_single(a, out=aotst)
        ni.ri_conv.compute_mol_convs_single_(
            a, f_gq, f_arqlp_buf[a], aotst, ind_g, pot=True
        )

    ngrids = grids.weights.size
    vbas_qarlp = np.ascontiguousarray(f_arqlp_buf.transpose(2,0,1,3,4))
    if ni.ri_conv.onsite_direct:
        vbas_ga = ni.ri_conv.transform_orbital_feat_bwd(
            grids, vbas_qarlp, f_gq=f_gq
        )
    else:
        vbas_ga = ni.ri_conv.transform_orbital_feat_bwd(grids, vbas_qarlp)

    if is_gga:
        p_ag, dp_ag, dadn, dadsigma = \
            ni.ri_conv.get_cider_coefs_bwd(rho, derivs=True, **vv_gg_settings)
    else:
        p_ag, dp_ag, dadn, dadsigma, dadtau = \
            ni.ri_conv.get_cider_coefs_bwd(rho, derivs=True, **vv_gg_settings)
    tmp_ga = vbas_ga # TODO F order
    tmp_g = _contract_rho(tmp_ga, p_ag.T)
    dtmp_g = _contract_rho(tmp_ga, dp_ag.T)
    tmp_g[rho[0]<CIDER_NUM_TOL] = 0
    dtmp_g[rho[0]<CIDER_NUM_TOL] = 0
    if is_gga:
        res = [
            tmp_g + rho[0] * dtmp_g * dadn,
            rho[0] * dtmp_g * dadsigma
        ]
    else:
        res = [
            tmp_g + rho[0] * dtmp_g * dadn,
            rho[0] * dtmp_g * dadsigma,
            None,
            rho[0] * dtmp_g * dadtau,
        ]
    if grad_mode:
        return res, tmp_g * rho[0]
    else:
        return res

def get_wv_rks_cider(ni, mol, grids, xc_code, rho,
                     relativity, hermi, verbose=None,
                     debug_feat=False, grad_mode=False,
                     taux=None):
    wtime = time.monotonic()
    is_gga = (ni._xc_type(xc_code) == 'GGA')
    if ni.uses_taux:
        assert taux is not None
    if not is_gga:
        rhop = np.empty((6, rho.shape[-1]))
        rhop[:4] = rho[:4]
        rhop[4] = 0
        rhop[5] = rho[4]
        rho = rhop

    weight, coords = grids.weights, grids.coords
    gg_settings, vv_gg_settings, gg_scales = \
        get_gg_settings(ni.mlfunc_x, is_gga=is_gga)
    Nalpha = len(ni.ri_conv.alphas)
    nfeat = 3
    dadn_i = {}
    dadsigma_i = {}
    if not is_gga:
        dadtau_i = {}
    f_ga, f_arqlp = get_feat_convolutions_fwd(
        ni, mol, grids, rho, vv_gg_settings, debug_feat=debug_feat
    )
    f_ga = np.asfortranarray(f_ga)
    feat_i = np.zeros((nfeat,rho.shape[-1]))
    dfeat_i = np.zeros((nfeat,rho.shape[-1]))
    p_ia = {}
    for i in range(nfeat):
        tmp = ni.ri_conv.get_cider_coefs_bwd(rho, derivs=True, **gg_settings[i])
        if is_gga:
            p_ag, dp_ag, dadn_i[i], dadsigma_i[i] = tmp
        else:
            p_ag, dp_ag, dadn_i[i], dadsigma_i[i], dadtau_i[i] = tmp
        feat_i[i] = gg_scales[i] * _contract_rho(f_ga, p_ag.T)
        dfeat_i[i] = gg_scales[i] * _contract_rho(f_ga, dp_ag.T)
        p_ia[i] = p_ag
    if debug_feat:
        return feat_i, f_arqlp

    # Get XC energy and potential from CIDER functional
    if ni.uses_taux:
        exc, vxc = ni.eval_xc(xc_code, mol, rho, feat_i,
                              0, relativity, 1, verbose=verbose,
                              taux=taux)[:2]
    else:
        exc, vxc = ni.eval_xc(xc_code, mol, rho, feat_i,
                              0, relativity, 1, verbose=verbose)[:2]

    if is_gga:
        vrho, vsigma, vfeat_i = vxc
        vfeat_i = vfeat_i[0]
        for i in range(nfeat):
            vxc[0][:] += vfeat_i[i] * dfeat_i[i] * dadn_i[i]
            vxc[1][:] += vfeat_i[i] * dfeat_i[i] * dadsigma_i[i]
    else:
        vrho, vsigma, vfeat_i, vtau = vxc
        if ni.uses_taux:
            vfeat_i, vtaux = vfeat_i
        vfeat_i = vfeat_i[0]
        for i in range(nfeat):
            vxc[0][:] += vfeat_i[i] * dfeat_i[i] * dadn_i[i]
            vxc[1][:] += vfeat_i[i] * dfeat_i[i] * dadsigma_i[i]
            vxc[3][:] += vfeat_i[i] * dfeat_i[i] * dadtau_i[i]

    den = rho[0] * weight
    nelec = den.sum()
    excsum = np.dot(den, exc)
    vf_ga = np.zeros((rho.shape[-1], Nalpha))
    for i in range(nfeat):
        const = gg_scales[i]
        vf_ga += p_ia[i].T * (vfeat_i[i] * const * weight)[:,None]

    if grad_mode:
        excsum = get_grad_convolutions(ni, mol, f_arqlp, vf_ga)

    #vxc_full[0].append(vxc[0])
    #vxc_full[1].append(vxc[1])
    is_gga = (ni._xc_type(xc_code) == 'GGA')
    vxc_cider = get_feat_convolutions_bwd(
        ni, mol, grids, rho, vf_ga, vv_gg_settings, f_arqlp,
        is_gga=is_gga, grad_mode=grad_mode
    )
    if grad_mode:
        vxc_cider, cidergg_g = vxc_cider
    vxc[0] += vxc_cider[0]
    vxc[1] += vxc_cider[1]
    if ni.uses_taux:
        vxc[3] += vxc_cider[3]
        wv = _rks_mmgga_wv0(rho, vxc, vtaux, weight)
    elif is_gga:
        wv = _rks_gga_wv0(rho, vxc, weight)
    else:
        vxc[3] += vxc_cider[3]
        wv = _rks_mgga_wv0(rho, vxc, weight)

    if grad_mode:
        return cidergg_g, excsum, wv, exc
    else:
        return nelec, excsum, wv, exc

def get_wv_uks_cider(ni, mol, grids, xc_code, rho,
                     relativity, hermi, verbose=None,
                     debug_feat=False, grad_mode=False,
                     taux=None):
    is_gga = (ni._xc_type(xc_code) == 'GGA')
    if ni.uses_taux:
        assert taux is not None and taux[0] is not None
    if not is_gga:
        rhop = np.empty((2, 6, rho[0].shape[-1]))
        for s in range(2):
            rhop[s,:4] = rho[s][:4]
            rhop[s,4] = 0
            rhop[s,5] = rho[s][4]
        rho = rhop

    weight, coords = grids.weights, grids.coords
    rho_a, rho_b = rho
    gg_settings, vv_gg_settings, gg_scales = \
        get_gg_settings(ni.mlfunc_x, is_gga=is_gga)
    Nalpha = len(ni.ri_conv.alphas)
    nfeat = 3
    # IMPORTANT: set nspin for get_exponent_d here
    vv_gg_settings['nspin'] = 2
    for i in range(nfeat):
        gg_settings[i]['nspin'] = 2
    Nalpha = len(ni.ri_conv.alphas)
    nfeat = 3
    dadn_a_i = {}
    dadsigma_a_i = {}
    dadn_b_i = {}
    dadsigma_b_i = {}
    if not is_gga:
        dadtau_a_i = {}
        dadtau_b_i = {}
    fa_ga, fa_arqlp = get_feat_convolutions_fwd(
        ni, mol, grids, rho_a, vv_gg_settings, debug_feat=debug_feat
    )
    fb_ga, fb_arqlp = get_feat_convolutions_fwd(
        ni, mol, grids, rho_b, vv_gg_settings, debug_feat=debug_feat
    )
    fa_ga = np.asfortranarray(fa_ga)
    fb_ga = np.asfortranarray(fb_ga)
    feat_i = np.zeros((2,nfeat,rho_a.shape[-1]))
    dfeat_i = np.zeros((2,nfeat,rho_a.shape[-1]))
    pa_ia = {}
    pb_ia = {}
    for i in range(nfeat):
        tmp = ni.ri_conv.get_cider_coefs_bwd(rho_a, derivs=True, **gg_settings[i])
        if is_gga:
            p_ag, dp_ag, dadn_a_i[i], dadsigma_a_i[i] = tmp
        else:
            p_ag, dp_ag, dadn_a_i[i], dadsigma_a_i[i], dadtau_a_i[i] = tmp
        feat_i[0,i] = gg_scales[i] * _contract_rho(fa_ga, p_ag.T)
        dfeat_i[0,i] = gg_scales[i] * _contract_rho(fa_ga, dp_ag.T)
        pa_ia[i] = p_ag

        tmp = ni.ri_conv.get_cider_coefs_bwd(rho_b, derivs=True, **gg_settings[i])
        if is_gga:
            p_ag, dp_ag, dadn_b_i[i], dadsigma_b_i[i] = tmp
        else:
            p_ag, dp_ag, dadn_b_i[i], dadsigma_b_i[i], dadtau_b_i[i] = tmp
        feat_i[1,i] = gg_scales[i] * _contract_rho(fb_ga, p_ag.T)
        dfeat_i[1,i] = gg_scales[i] * _contract_rho(fb_ga, dp_ag.T)
        pb_ia[i] = p_ag
    if debug_feat:
        return feat_i, fa_aqrlp

    if ni.uses_taux:
        exc, vxc = ni.eval_xc(xc_code, mol, np.array([rho_a, rho_b]), feat_i,
                              1, relativity, 1, verbose=verbose,
                              taux=np.asarray(taux))[:2]
    else:
        exc, vxc = ni.eval_xc(xc_code, mol, np.array([rho_a, rho_b]), feat_i,
                              1, relativity, 1, verbose=verbose)[:2]

    vrho, vsigma, vfeat_i = vxc[:3]
    if ni.uses_taux:
        vfeat_i, vtaux = vfeat_i
    for i in range(nfeat):
        vxc[0][:,0] += vfeat_i[0,i] * dfeat_i[0,i] * dadn_a_i[i]
        vxc[1][:,0] += vfeat_i[0,i] * dfeat_i[0,i] * dadsigma_a_i[i]
        vxc[0][:,1] += vfeat_i[1,i] * dfeat_i[1,i] * dadn_b_i[i]
        vxc[1][:,2] += vfeat_i[1,i] * dfeat_i[1,i] * dadsigma_b_i[i]
    if not is_gga:
        for i in range(nfeat):
            vxc[3][:,0] += vfeat_i[0,i] * dfeat_i[0,i] * dadtau_a_i[i]
            vxc[3][:,1] += vfeat_i[1,i] * dfeat_i[1,i] * dadtau_b_i[i]

    den = (rho_a[0] + rho_b[0]) * weight
    nelec = den.sum()
    excsum = np.dot(den, exc)

    vf_ga = np.zeros((rho_a.shape[-1], Nalpha))
    for i in range(nfeat):
        const = gg_scales[i]
        vf_ga += pa_ia[i].T * (vfeat_i[0,i] * const * weight)[:,None]
    if grad_mode:
        excsum = get_grad_convolutions(ni, mol, fa_arqlp, vf_ga)
    vxc_a_cider = get_feat_convolutions_bwd(
        ni, mol, grids, rho_a, vf_ga, vv_gg_settings, fa_arqlp,
        is_gga=is_gga, grad_mode=grad_mode
    )
    if grad_mode:
        vxc_a_cider, cidergg_g = vxc_a_cider

    vf_ga = np.zeros((rho_b.shape[-1], Nalpha))
    for i in range(nfeat):
        const = gg_scales[i]
        vf_ga += pb_ia[i].T * (vfeat_i[1,i] * const * weight)[:,None]
    if grad_mode:
        excsum += get_grad_convolutions(ni, mol, fb_arqlp, vf_ga)
    vxc_b_cider = get_feat_convolutions_bwd(
        ni, mol, grids, rho_b, vf_ga, vv_gg_settings, fb_arqlp,
        is_gga=is_gga, grad_mode=grad_mode
    )
    if grad_mode:
        vxc_b_cider, cb_g = vxc_b_cider
        cidergg_g += cb_g

    vxc[0][:,0] += vxc_a_cider[0]
    vxc[0][:,1] += vxc_b_cider[0]
    vxc[1][:,0] += vxc_a_cider[1]
    vxc[1][:,2] += vxc_b_cider[1]
    if ni._xc_type(xc_code) == 'GGA':
        wva, wvb = _uks_gga_wv0((rho_a, rho_b), vxc, weight)
    elif ni.uses_taux:
        vxc[3][:, 0] += vxc_a_cider[3]
        vxc[3][:, 1] += vxc_b_cider[3]
        wva, wvb = _uks_mmgga_wv0((rho_a, rho_b), vxc, vtaux.T, weight)
    else:
        vxc[3][:,0] += vxc_a_cider[3]
        vxc[3][:,1] += vxc_b_cider[3]
        wva, wvb = _uks_mgga_wv0((rho_a, rho_b), vxc, weight)
    if grad_mode:
        return cidergg_g, excsum, wva, wvb, exc
    else:
        return nelec, excsum, wva, wvb, exc


def nr_rks_ri(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
              max_memory=2000, verbose=None):
    START_TIME = time.monotonic()
    if ni.ri_conv is None:
        ni.setup_aux(mol)
    xctype = ni._xc_type(xc_code)
    if xctype == 'NLC':
        return NumInt.nr_rks(ni, mol, grids, xc_code, dms, relativity,
                             hermi, max_memory, verbose)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    if ni.uses_taux:
        Ts = 0.5 * mol.intor_symmetric('int1e_kin')
        if not isinstance(dms, np.ndarray) or dms.ndim == 3:
            kdms = []
            for idm in range(nset):
                dm = dms[idm]
                kdms.append(lib.dot(dm.T, lib.dot(Ts, dm)))
            kdms = np.ascontiguousarray(np.stack(kdms))
        else:
            kdms = lib.dot(dms.T, lib.dot(Ts, dms))
        make_taux = ni._gen_rho_evaluator(mol, kdms, hermi, False, grids)[0]
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    ao_deriv = 1
    Nrhofeat = 4 if xctype == 'GGA' else 5

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, nao, nao))
    aow = None
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)

    init_time = time.monotonic()
    rho_full = np.zeros((nset, Nrhofeat, grids.weights.size), dtype=np.float64, order='C')
    if ni.uses_taux:
        taux_full = np.zeros((nset, grids.weights.size), dtype=np.float64, order='C')
    elif ni.uses_flapl:
        taux_full = get_full_flapl(ni, mol, dms, grids, 'LDA')
    else:
        taux_full = None
    ip0 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        mask = None # TODO remove once screening is fixed
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rho = make_rho(idm, ao, mask, xctype)
            rho_full[idm, :, ip0:ip1] = rho
            if ni.uses_taux:
                taux_full[idm, ip0:ip1] = make_taux(idm, ao[0], None, 'LDA')
        ip0 = ip1
    ip0 = ip1 = None
    #print('ITIME', time.monotonic() - init_time)

    wv_full = []
    for idm in range(nset):
        if taux_full is None:
            taux = None
        else:
            taux = taux_full[idm]
        _get_wv = get_wv_rks_cider_vh if ni.mlfunc_x.desc_version in ['h', 'i', 'k'] \
                  else get_wv_rks_cider
        nelec[idm], excsum[idm], wv = _get_wv(
            ni, mol, grids, xc_code, rho_full[idm],
            relativity, hermi, verbose, taux=taux,
            #debug_feat=debug_feat
        )[:3]
        wv_full.append(wv)

    init_time = time.monotonic()
    tgg = 0
    tmgg = 0
    ip0 = 0
    if xctype == 'MGGA':
        v1 = np.zeros_like(vmat)
    if ni.uses_taux:
        v2 = np.zeros_like(vmat)
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        mask = None # TODO remove once screening is fixed
        ip1 = ip0 + weight.size
        for idm in range(nset):
            t0 = time.monotonic()
            wv = np.ascontiguousarray(wv_full[idm][:,ip0:ip1])
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(
                ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                hermi=0, out=vmat[idm],
            )
            t1 = time.monotonic()
            if xctype == 'MGGA':
                #_tau_dot_sparse(
                #    ao, ao, wv[5], nbins, mask, pair_mask, ao_loc,
                #    out=v1[idm],
                #)
                wvtmp = wv[5:6]
                aow = _scale_ao_sparse(ao[1:2], wvtmp, mask, ao_loc, out=aow)
                _dot_ao_ao_sparse(
                    ao[1], aow, None, nbins, mask, pair_mask, ao_loc,
                    hermi=0, out=v1[idm],
                )
                aow = _scale_ao_sparse(ao[2:3], wvtmp, mask, ao_loc, out=aow)
                _dot_ao_ao_sparse(
                    ao[2], aow, None, nbins, mask, pair_mask, ao_loc,
                    hermi=0, out=v1[idm],
                )
                aow = _scale_ao_sparse(ao[3:4], wvtmp, mask, ao_loc, out=aow)
                _dot_ao_ao_sparse(
                    ao[3], aow, None, nbins, mask, pair_mask, ao_loc,
                    hermi=0, out=v1[idm],
                )
            t2 = time.monotonic()
            tgg += t1 - t0
            tmgg += t2 - t1
            if ni.uses_taux:
                aow_tmp = _scale_ao_sparse(ao[0], wv[6], mask, ao_loc)
                _dot_ao_ao_sparse(
                    ao[0], aow_tmp, None, nbins, mask, pair_mask, ao_loc,
                    hermi=0, out=v2[idm],
                )
        ip0 = ip1
    #print('PTIME', time.monotonic() - init_time, tgg, tmgg)
    if ni.uses_flapl:
        # pass
        get_full_flapl_grad(ni, mol, [wv_full[idm][6:, :] for idm in range(nset)],
                            grids, nset, nbins, pair_mask, ao_loc, vmat)
    vmat = lib.hermi_sum(vmat, axes=(0,2,1))
    if xctype == 'MGGA':
        vmat += 2 * v1
    if ni.uses_taux:
        if dms.ndim == 2:
            tmp = lib.dot(lib.dot(Ts, dms), v2[0].T)
            vmat[0] += tmp + tmp.T
        else:
            for idm in range(nset):
                tmp = lib.dot(lib.dot(Ts, dms[idm]), v2[idm].T)
                vmat[idm] += tmp + tmp.T
    #print('FULL XC LOOP', time.monotonic()-START_TIME)

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]

    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat


def nr_uks_ri(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
              max_memory=2000, verbose=None):
    START_TIME = time.monotonic()
    if ni.ri_conv is None:
        ni.setup_aux(mol)
    xctype = ni._xc_type(xc_code)
    if xctype == 'NLC':
        return NumInt.nr_uks(ni, mol, grids, xc_code, dms, relativity,
                             hermi, max_memory, verbose)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    ao_deriv = 1
    Nrhofeat = 4 if xctype == 'GGA' else 5

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]
    if ni.uses_taux:
        Ts = mol.intor_symmetric('int1e_kin')
        if not isinstance(dma, np.ndarray) or dma.ndim == 3:
            kdma = []
            kdmb = []
            for idm in range(nset):
                dm = dma[idm]
                kdma.append(lib.dot(dm.T, lib.dot(Ts, dm)))
                dm = dmb[idm]
                kdmb.append(lib.dot(dm.T, lib.dot(Ts, dm)))
            kdma = np.ascontiguousarray(np.stack(kdma))
            kdmb = np.ascontiguousarray(np.stack(kdmb))
        else:
            kdma = lib.dot(dma.T, lib.dot(Ts, dma))
            kdmb = lib.dot(dmb.T, lib.dot(Ts, dmb))
        make_tauxa = ni._gen_rho_evaluator(mol, kdma, hermi, False, grids)[0]
        make_tauxb = ni._gen_rho_evaluator(mol, kdmb, hermi, False, grids)[0]

    nelec = np.zeros((2, nset))
    excsum = np.zeros(nset)
    vmat = np.zeros((2, nset, nao, nao))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    aow = None

    rhoa_full = np.zeros((nset, Nrhofeat, grids.weights.size),
                         dtype=np.float64, order='C')
    rhob_full = np.zeros((nset, Nrhofeat, grids.weights.size),
                         dtype=np.float64, order='C')
    if ni.uses_taux:
        tauxa_full = np.zeros((nset, grids.weights.size), dtype=np.float64, order='C')
        tauxb_full = np.zeros((nset, grids.weights.size), dtype=np.float64, order='C')
    elif ni.uses_flapl:
        tauxa_full = get_full_flapl(ni, mol, dma, grids, 'LDA')
        tauxb_full = get_full_flapl(ni, mol, dmb, grids, 'LDA')
    else:
        tauxa_full = None
        tauxb_full = None
    ip0 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        mask = None # TODO remove once screening is fixed
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rhoa_full[idm,:,ip0:ip1] = make_rhoa(idm, ao, mask, xctype)
            rhob_full[idm,:,ip0:ip1] = make_rhob(idm, ao, mask, xctype)
            if ni.uses_taux:
                tauxa_full[idm,ip0:ip1] = make_tauxa(idm, ao[0], mask, 'LDA')
                tauxb_full[idm,ip0:ip1] = make_tauxb(idm, ao[0], mask, 'LDA')
        ip0 = ip1
    ip0 = ip1 = None

    wva_full = []
    wvb_full = []
    for idm in range(nset):
        if tauxa_full is None:
            taux = None
        else:
            taux = (tauxa_full[idm], tauxb_full[idm])
        _get_wv = get_wv_uks_cider_vh if ni.mlfunc_x.desc_version in ['h', 'i', 'k'] \
            else get_wv_uks_cider
        nelec[idm], excsum[idm], wva, wvb = _get_wv(
            ni, mol, grids, xc_code, (rhoa_full[idm], rhob_full[idm]),
            relativity, hermi, verbose, taux=taux,
            #debug_feat=debug_feat
        )[:4]
        wva_full.append(wva)
        wvb_full.append(wvb)

    ip0 = 0
    if xctype == 'MGGA':
        v1 = np.zeros_like(vmat)
    if ni.uses_taux:
        v2 = np.zeros_like(vmat)
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        mask = None # TODO remove once screening is fixed
        ip1 = ip0 + weight.size
        for idm in range(nset):
            wva = np.ascontiguousarray(wva_full[idm][:,ip0:ip1])
            wvb = np.ascontiguousarray(wvb_full[idm][:,ip0:ip1])
            aow = _scale_ao_sparse(ao[:4], wva[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[0,idm])
            aow[:] = 0
            aow = _scale_ao_sparse(ao[:4], wvb[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[1,idm])
            if xctype == 'MGGA':
                _tau_dot_sparse(ao, ao, wva[5], nbins, mask,
                                pair_mask, ao_loc, out=v1[0,idm])
                _tau_dot_sparse(ao, ao, wvb[5], nbins, mask,
                                pair_mask, ao_loc, out=v1[1,idm])
            if ni.uses_taux:
                aow_tmp = _scale_ao_sparse(ao[0], wva[6], mask, ao_loc)
                _dot_ao_ao_sparse(
                    ao[0], aow_tmp, None, nbins, mask, pair_mask, ao_loc,
                    hermi=0, out=v2[0,idm],
                )
                aow_tmp = _scale_ao_sparse(ao[0], wvb[6], mask, ao_loc)
                _dot_ao_ao_sparse(
                    ao[0], aow_tmp, None, nbins, mask, pair_mask, ao_loc,
                    hermi=0, out=v2[1, idm],
                )
        ip0 = ip1
    if ni.uses_flapl:
        # pass
        get_full_flapl_grad(ni, mol, [wva_full[idm][6:, :] for idm in range(nset)],
                            grids, nset, nbins, pair_mask, ao_loc, vmat[0])
        get_full_flapl_grad(ni, mol, [wvb_full[idm][6:, :] for idm in range(nset)],
                            grids, nset, nbins, pair_mask, ao_loc, vmat[1])
    vmat = lib.hermi_sum(
        vmat.reshape(-1,nao,nao), axes=(0,2,1)
    ).reshape(2,nset,nao,nao)
    if xctype == 'MGGA':
        vmat += 2 * v1
    if ni.uses_taux:
        assert nset == 1 and dma.ndim == 2
        tmp = lib.dot(lib.dot(Ts, dma), v2[0,0].T)
        vmat[0,0] += tmp + tmp.T
        tmp = lib.dot(lib.dot(Ts, dmb), v2[1,0].T)
        vmat[1,0] += tmp + tmp.T
        #for idm in range(nset):
        #    tmp = lib.dot(lib.dot(Ts, dma[idm]), v2[0,idm].T)
        #    vmat[0,idm] += tmp + tmp.T
        #    tmp = lib.dot(lib.dot(Ts, dmb[idm]), v2[1,idm].T)
        #    vmat[1,idm] += tmp + tmp.T
    #print('FULL XC LOOP', time.monotonic()-START_TIME)

    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = numpy.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat


def _eval_xc_cider(self, xc_code, mol, rho_data, feat,
                   spin=0, relativity=0, deriv=1, omega=None,
                   verbose=None, taux=None):
    if deriv != 1:
        raise NotImplementedError('Only deriv=1 implemented, but deriv={}'.format(deriv))
    if self.xc_code is not None:
        xc_code = self.xc_code
    if spin == 0:
        N = rho_data.shape[-1]
        sigma = np.einsum('vg,vg->g', rho_data[1:4], rho_data[1:4])
        xctype = self.libxc.xc_type(xc_code)
        if xctype == 'LDA':
            nvar = 1
        elif xctype == 'GGA':
            nvar = 4
        elif xctype == 'MGGA':
            nvar = 6
        else:
            raise ValueError
        exc, vxc = eval_xc(
            xc_code, rho_data[:nvar],
            spin=0, relativity=relativity,
            deriv=deriv, omega=omega, verbose=verbose
        )[:2]
        exc *= rho_data[0]
        if self.is_gga:
            vnl = self.cider_kernel.calculate(
                exc, rho_data[0].reshape(1,-1), vxc[0].reshape(1,-1),
                sigma.reshape(1,-1), vxc[1].reshape(1,-1), feat.reshape(1,feat.shape[0],-1),
            )
            exc /= rho_data[0] + 1e-16
            return exc, [vxc[0], vxc[1], vnl]
        else:
            if len(vxc) == 2:
                vxc = vxc + [None, np.zeros_like(vxc[0])]
            if self.uses_taux:
                assert taux is not None
                vnl = self.cider_kernel.calculate(
                    exc, rho_data[0].reshape(1, -1), vxc[0].reshape(1, -1),
                    sigma.reshape(1, -1), vxc[1].reshape(1, -1),
                    rho_data[5].reshape(1, -1), vxc[3].reshape(1, -1),
                    feat.reshape(1, feat.shape[0], -1), taux.reshape(1, -1),
                )
            else:
                vnl = self.cider_kernel.calculate(
                    exc, rho_data[0].reshape(1,-1), vxc[0].reshape(1,-1),
                    sigma.reshape(1,-1), vxc[1].reshape(1,-1),
                    rho_data[5].reshape(1,-1), vxc[3].reshape(1,-1),
                    feat.reshape(1,feat.shape[0],-1),
                )
            exc /= rho_data[0] + 1e-16
            return exc, [vxc[0], vxc[1], vnl, vxc[3]]
    else:
        # rho_data (spin, x, grid)
        N = rho_data.shape[-1]
        sigma_xg = np.zeros((3,N))
        sigma_xg[0] = np.einsum('vg,vg->g', rho_data[0,1:4], rho_data[0,1:4])
        sigma_xg[1] = np.einsum('vg,vg->g', rho_data[0,1:4], rho_data[1,1:4])
        sigma_xg[2] = np.einsum('vg,vg->g', rho_data[1,1:4], rho_data[1,1:4])
        xctype = self.libxc.xc_type(xc_code)
        if xctype == 'LDA':
            nvar = 1
        elif xctype == 'GGA':
            nvar = 4
        elif xctype == 'MGGA':
            nvar = 6
        else:
            raise ValueError
        exc, vxc = eval_xc(
            xc_code, (rho_data[0][:nvar], rho_data[1][:nvar]),
            spin=spin, relativity=relativity,
            deriv=deriv, omega=omega, verbose=verbose
        )[:2]
        rhot = rho_data[0,0] + rho_data[1,0]
        exc *= rhot
        if self.is_gga:
            vnl = self.cider_kernel.calculate(
                exc, rho_data[:,0], vxc[0].T,
                sigma_xg, vxc[1].T,
                feat
            )
            exc /= rhot + 1e-16
            return exc, [vxc[0], vxc[1], vnl]
        else:
            if len(vxc) == 2:
                vxc = vxc + [None, np.zeros_like(vxc[0])]
            if self.uses_taux:
                assert taux is not None
                vnl = self.cider_kernel.calculate(
                    exc, rho_data[:,0], vxc[0].T,
                    sigma_xg, vxc[1].T,
                    rho_data[:,5], vxc[3].T,
                    feat, taux,
                )
            else:
                vnl = self.cider_kernel.calculate(
                    exc, rho_data[:, 0], vxc[0].T,
                    sigma_xg, vxc[1].T,
                    rho_data[:, 5], vxc[3].T,
                    feat,
                )
            exc /= rhot + 1e-16
            return exc, [vxc[0], vxc[1], vnl, vxc[3]]


class CiderNumInt(NumInt):

    nr_rks = nr_rks_ri

    nr_uks = nr_uks_ri

    eval_xc = _eval_xc_cider

    block_loop_flapl = _block_loop_flapl

    def __init__(self, mol, mlfunc_x, xkernel=None,
                 ckernel=None, vv10_coeff=None, xmix=1.0,
                 **kwargs):
        super(CiderNumInt, self).__init__()
        if vv10_coeff is None:
            vv10_coeff = CIDER_DEFAULT_VV10_COEFF
        self._vv10_coeff = vv10_coeff
        if xkernel is not None:
            self.xc_code = '{}*{} + {}'.format(1-xmix, xkernel, ckernel)
        elif ckernel is not None:
            self.xc_code = ckernel
        else:
            self.xc_code = '{}*{}'.format(0.00, 'PBE')
        self.xmix = xmix
        self.mlfunc_x = mlfunc_x
        assert mlfunc_x.desc_version in ['b', 'd', 'f', 'g', 'h', 'i', 'j', 'k'], \
               'Unsupported desc version'
        self.uses_taux = False
        self.uses_flapl = False
        if mlfunc_x.desc_version == 'd':
            self.cider_kernel = CiderGGAHybridKernel(
                mlfunc_x, xmix
            )
            if not is_gga(self.xc_code):
                raise ValueError('No MGGA baseline for CIDER GGA')
            self.is_gga = True
        elif mlfunc_x.desc_version == 'b':
            self.cider_kernel = CiderMGGAHybridKernel(
                mlfunc_x, xmix
            )
            self.is_gga = False
        elif mlfunc_x.desc_version in ['g', 'h', 'i', 'j', 'k']:
            self.cider_kernel = CiderMGGAHybridKernel(
                mlfunc_x, xmix
            )
            if mlfunc_x.desc_version in ['i']:
                self.uses_flapl = True
            self.is_gga = False
        elif mlfunc_x.desc_version == 'f':
            if not hasattr(self.mlfunc_x, 'vvmul'):
                self.mlfunc_x.vvmul = 1.0 # TODO should set elsewhere
            self.cider_kernel = CiderMMGGAHybridKernel(
                mlfunc_x, xmix
            )
            self.is_gga = False
            self.uses_taux = True
            #self.cider_kernel = CiderMGGAHybridKernel(
            #    mlfunc_x, xmix
            #)
        else:
            raise NotImplementedError
        self.ccs_kwargs = kwargs
        self.aux_non0tab = None
        self.buf = None
        self.aux_buf = None
        self.ri_conv = None
        self._cider_lmax = (self.ccs_kwargs.pop('cider_lmax', None)
                            or CIDER_DEFAULT_LMAX)
        self._cider_amin = self.ccs_kwargs.pop('amin', None)

    def reset(self, mol=None):
        self.mol = mol
        self.ri_conv = None
        self.aux_non0tab = None
        self.buf = None
        self.aux_buf = None

    def _xc_type(self, xc_code):
        if is_nlc(xc_code):
            return 'NLC'
        return 'GGA' if self.is_gga else 'MGGA'

    def nlc_coeff(self, xc_code):
        return self._vv10_coeff

    def rsh_and_hybrid_coeff(self, xc_code, spin=0):
        return 0, 0, 0

    def setup_aux(self, mol):
        vmul = self.mlfunc_x.vvmul
        amin = self._cider_amin
        if amin is None:
            amin = self.mlfunc_x.amin / (2*np.e)
            if vmul is not None:
                amin = min(amin/2, amin*vmul)
        cc_cls = CiderConvSpline
        if self.mlfunc_x.desc_version == 'g':
            cc_cls = CiderConvSplineG
        elif self.mlfunc_x.desc_version == 'h':
            #cc_cls = CiderConvSplineH
            cc_cls = CiderConvSplineHv2
        elif self.mlfunc_x.desc_version == 'i':
            cc_cls = CiderConvSplineI
        elif self.mlfunc_x.desc_version == 'k':
            cc_cls = CiderConvSplineK
        self.ri_conv = cc_cls(
            mol, amin=amin, cider_lmax=self._cider_lmax,
            **(self.ccs_kwargs)
        )
        self.big_buf = None
        self.big_non0tab = None


class CiderNumIntDebug(CiderNumInt):

    def eval_xc(self, xc_code, mol, rho_data, feat,
                spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        if self.xc_code is not None:
            xc_code = self.xc_code
        if spin == 0:
            N = rho_data.shape[-1]
            sigma = np.einsum('vg,vg->g', rho_data[1:4], rho_data[1:4])
            exc, vxc = eval_xc(
                xc_code, rho_data,
                spin=0, relativity=relativity,
                deriv=deriv, omega=omega, verbose=verbose
            )[:2]
            exc *= rho_data[0]
            if self.is_gga:
                vnl = self.cider_kernel.calculate(
                    exc, rho_data[0].reshape(1,-1), vxc[0].reshape(1,-1),
                    sigma.reshape(1,-1), vxc[1].reshape(1,-1), feat.reshape(1,feat.shape[0],-1),
                )
            else:
                vnl = self.cider_kernel.calculate(
                    exc, rho_data[0].reshape(1,-1), vxc[0].reshape(1,-1),
                    sigma.reshape(1,-1), vxc[1].reshape(1,-1),
                    rho_data[5].reshape(1,-1), np.zeros_like(vxc[0]).reshape(1,-1), # no need for tau potential
                    feat.reshape(1,feat.shape[0],-1),
                )
            exc /= rho_data[0] + 1e-16
            _, vxc = eval_xc(
                'PBE', rho_data,
                spin=0, relativity=relativity,
                deriv=deriv, omega=omega, verbose=verbose
            )[:2]
            if self.is_gga:
                return exc, [vxc[0], vxc[1], 0*vnl]
            else:
                return exc, [vxc[0], vxc[1], 0*vnl, 0*vxc[0]]
        else:
            # rho_data (spin, x, grid)
            N = rho_data.shape[-1]
            sigma_xg = np.zeros((3,N))
            sigma_xg[0] = np.einsum('vg,vg->g', rho_data[0,1:4], rho_data[0,1:4])
            sigma_xg[1] = np.einsum('vg,vg->g', rho_data[0,1:4], rho_data[1,1:4])
            sigma_xg[2] = np.einsum('vg,vg->g', rho_data[1,1:4], rho_data[1,1:4])
            exc, vxc = eval_xc(
                xc_code, rho_data,
                spin=1, relativity=relativity,
                deriv=deriv, omega=omega, verbose=verbose
            )[:2]
            rhot = rho_data[0,0] + rho_data[1,0]
            exc *= rhot
            if self.is_gga:
                vnl = self.cider_kernel.calculate(
                    exc, rho_data[:,0], vxc[0].T,
                    sigma_xg, vxc[1].T, feat
                )
            else:
                vnl = self.cider_kernel.calculate(
                    exc, rho_data[:,0], vxc[0].T,
                    sigma_xg, vxc[1].T,
                    rho_data[:,5], np.zeros_like(vxc[0]).T, # no need for tau potential
                    feat
                )
            exc /= rhot + 1e-16
            _, vxc = eval_xc(
                'PBE', rho_data,
                spin=1, relativity=relativity,
                deriv=deriv, omega=omega, verbose=verbose
            )[:2]
            if self.is_gga:
                return exc, [vxc[0], vxc[1], 0*vnl]
            else:
                return exc, [vxc[0], vxc[1], 0*vnl, 0*vxc[0]]


class SLCiderNumInt(NumInt):

    _eval_xc_helper = _eval_xc_cider

    def __init__(self, mol, mlfunc_x, xkernel=None,
                 ckernel=None, vv10_coeff=None, xmix=1.0,
                 **kwargs):
        super(SLCiderNumInt, self).__init__()
        if vv10_coeff is None:
            vv10_coeff = CIDER_DEFAULT_VV10_COEFF
        self._vv10_coeff = vv10_coeff
        if xkernel is not None:
            self.xc_code = '{}*{} + {}'.format(1-xmix, xkernel, ckernel)
        elif ckernel is not None:
            self.xc_code = ckernel
        else:
            self.xc_code = '{}*{}'.format(0.00, 'PBE')
        self.xmix = xmix
        self.mlfunc_x = mlfunc_x
        assert mlfunc_x.desc_version in ['b', 'd'], 'Unsupported desc version'
        # TODO implement version f later to allow s^2, alpha, tau/taux features
        self.uses_taux = (mlfunc_x.desc_version == 'f')
        if mlfunc_x.desc_version == 'd':
            self.cider_kernel = CiderGGAHybridKernel(
                mlfunc_x, xmix
            )
            if not is_gga(self.xc_code):
                raise ValueError('No MGGA baseline for CIDER GGA')
            self.is_gga = True
        else:
            self.cider_kernel = CiderMGGAHybridKernel(
                mlfunc_x, xmix
            )
            self.is_gga = False

    def nlc_coeff(self, xc_code):
        return self._vv10_coeff

    def _xc_type(self, xc_code):
        if is_nlc(xc_code):
            return 'NLC'
        return 'GGA' if self.is_gga else 'MGGA'

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        ngrids = rho[0].shape[-1]
        feat_shape = (3,ngrids) if spin == 0 else (2,3,ngrids)
        feat = np.zeros(feat_shape)
        exc, vxc = self._eval_xc_helper(
            xc_code, None, rho, feat,
            spin=spin, relativity=relativity, deriv=deriv, omega=omega,
            verbose=verbose,
        )
        vxc[2] = None
        return exc, vxc, None, None


class _CiderKS(object):
    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, mol=None):
        hf.SCF.reset(self, mol)
        self.grids.reset(mol)
        self.nlcgrids.reset(mol)
        self._numint.reset(mol)
        return self

    Hessian = method_not_implemented
    NMR = method_not_implemented
    NSR = method_not_implemented
    Polarizability = method_not_implemented
    RotationalGTensor = method_not_implemented
    MP2 = method_not_implemented
    CISD = method_not_implemented
    CCSD = method_not_implemented
    CASCI = method_not_implemented
    CASSCF = method_not_implemented


class _DFCider(df.df_jk._DFHF):

    def nuc_grad_method(self):
        if isinstance(self, CiderRKS):
            from ciderpress.dft.rks_grad import DFGradients
            return DFGradients(self)
        elif isinstance(self, CiderUKS):
            from ciderpress.dft.uks_grad import DFGradients
            return DFGradients(self)
        else:
            raise NotImplementedError

    Gradients = nuc_grad_method

    Hessian = df.df_jk._DFHF.method_not_implemented


class CiderRKS(_CiderKS, dft.rks.RKS):

    def nuc_grad_method(self):
        from ciderpress.dft import rks_grad
        return rks_grad.Gradients(self)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        import pyscf.df.df_jk as df_jk
        _TMP = df_jk._DFHF
        df_jk._DFHF = _DFCider
        new_mf = df_jk.density_fit(self, auxbasis, with_df, only_dfj)
        df_jk._DFHF = _TMP
        return new_mf

    Gradients = nuc_grad_method


class CiderUKS(_CiderKS, dft.uks.UKS):

    def nuc_grad_method(self):
        from ciderpress.dft import uks_grad
        return uks_grad.Gradients(self)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        import pyscf.df.df_jk as df_jk
        _TMP = df_jk._DFHF
        df_jk._DFHF = _DFCider
        new_mf = df_jk.density_fit(self, auxbasis, with_df, only_dfj)
        df_jk._DFHF = _TMP
        return new_mf

    Gradients = nuc_grad_method


class DFCiderRKS(df.df_jk._DFHF, CiderRKS):

    def __init__(self, new_mf):
        self.__dict__.update(new_mf.__dict__)

    def nuc_grad_method(self):
        from ciderpress.dft.rks_grad import DFGradients
        return DFGradients(self)

    Gradients = nuc_grad_method


class DFCiderUKS(df.df_jk._DFHF, CiderUKS):

    def __init__(self, new_mf):
        self.__dict__.update(new_mf.__dict__)

    def nuc_grad_method(self):
        from ciderpress.dft.uks_grad import DFGradients
        return DFGradients(self)

    Gradients = nuc_grad_method
