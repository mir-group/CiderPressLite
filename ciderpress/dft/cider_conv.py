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

from pyscf.gto.mole import ATOM_OF, ANG_OF, PTR_EXP
from pyscf import gto, lib, scf
from pyscf.gto import moleintor
from pyscf.data import elements

import numpy
import numpy as np
import ctypes
import time

import scipy.linalg.lapack
from pyscf.data.radii import COVALENT as COVALENT_RADII

from ciderpress.dft.sph_harm_coeff import get_deriv_ylm_coeff
from ciderpress.lib import load_library as load_cider_library
from ciderpress.dft.flapl_util import FRAC_LAPL_POWER
libcider = load_cider_library('libcider')

CIDER_DEFAULT_LMAX = 10
DEFAULT_ALPHA_BETA = 1.6
DEFAULT_AUX_BETA = 1.8
DEFAULT_AMAX = 3000.0
DEFAULT_AMIN = 0.01
DEFAULT_ONSITE_DIRECT = True


def aug_etb_for_cider(mol, beta=DEFAULT_AUX_BETA, upper_fac=1.0, lower_fac=1.0,
                      start_at=0, lmax=CIDER_DEFAULT_LMAX,
                      def_afac_max=4.0, def_afac_min=0.5):
    '''augment weigend basis with even-tempered gaussian basis
    exps = alpha*beta^i for i = 1..N
    For this implementation, exponent values should be on the
    same 'grid' ..., 1/beta, 1, beta, beta^2, ...
    '''
    uniq_atoms = set([a[0] for a in mol._atom])

    newbasis = {}
    for symb in uniq_atoms:
        nuc_charge = gto.charge(symb)
        if False:
            pass
        #?elif symb in mol._ecp:
        else:
            conf = elements.CONFIGURATION[nuc_charge]
            max_shells = lmax
            emin_by_l = [1e99] * 8
            emax_by_l = [0] * 8
            l_max = 7
            for b in mol._basis[symb]:
                l = b[0]
                if l >= max_shells+1:
                    continue

                if isinstance(b[1], int):
                    e_c = numpy.array(b[2:])
                else:
                    e_c = numpy.array(b[1:])
                es = e_c[:,0]
                cs = e_c[:,1:]
                es = es[abs(cs).max(axis=1) > 1e-3]
                emax_by_l[l] = max(es.max(), emax_by_l[l])
                emin_by_l[l] = min(es.min(), emin_by_l[l])

            l_max1 = l_max + 1
            emin_by_l = numpy.array(emin_by_l)
            emax_by_l = numpy.array(emax_by_l)

# Estimate the exponents ranges by geometric average
            emax = numpy.sqrt(numpy.einsum('i,j->ij', emax_by_l, emax_by_l))
            emin = numpy.sqrt(numpy.einsum('i,j->ij', emin_by_l, emin_by_l))
            liljsum = numpy.arange(l_max1)[:,None] + numpy.arange(l_max1)
            emax_by_l = [emax[liljsum==ll].max() for ll in range(l_max1*2-1)]
            emin_by_l = [emin[liljsum==ll].min() for ll in range(l_max1*2-1)]
            # Tune emin and emax
            emin_by_l = numpy.array(emin_by_l) * lower_fac # *2 for alpha+alpha on same center
            emax_by_l = numpy.array(emax_by_l) * upper_fac  #/ (numpy.arange(l_max1*2-1)*.5+1)

            def_amax = def_afac_max / COVALENT_RADII[nuc_charge]**2
            def_amin = def_afac_min / COVALENT_RADII[nuc_charge]**2

            cond = (emax_by_l == 0)
            emax_by_l[cond] = def_amax
            emin_by_l[cond] = def_amin
            emax_by_l = np.maximum(def_amax, emax_by_l[:lmax+1])
            emin_by_l = np.minimum(def_amin, emin_by_l[:lmax+1])

            ns = numpy.log((emax_by_l+emin_by_l)/emin_by_l) / numpy.log(beta)
            nmaxs = numpy.ceil(np.log(emax_by_l) / numpy.log(beta))
            nmins = numpy.floor(np.log(emin_by_l) / numpy.log(beta))
            emin_by_l = beta**nmins
            ns = nmaxs-nmins+1
            etb = []
            for l, n in enumerate(numpy.ceil(ns).astype(int)):
                #print(l, n, emin_by_l[l], emax_by_l[l], beta)
                if n > 0 and l <= lmax:
                    etb.append((l, n, emin_by_l[l], beta))
            newbasis[symb] = gto.expand_etbs(etb)

    return newbasis, ns, np.max(nmaxs) - nmaxs


def aug_etb_for_cider_v2(mol, beta=1.8, start_at=0, upper_buf=2.0, lower_buf=0.5, lmax=CIDER_DEFAULT_LMAX):
    nuc_start = gto.charge(start_at)
    uniq_atoms = set([a[0] for a in mol._atom])

    newbasis = {}
    for symb in uniq_atoms:
        nuc_charge = gto.charge(symb)
        conf = elements.CONFIGURATION[nuc_charge]
        max_shells = lmax if isinstance(lmax, int) else lmax[symb]
        emin_by_l = [1e99] * 8
        emax_by_l = [0] * 8
        l_max = 0
        for b in mol._basis[symb]:
            l = b[0]
            l_max = max(l_max, l)
            if l >= max_shells+1:
                continue

            if isinstance(b[1], int):
                e_c = numpy.array(b[2:])
            else:
                e_c = numpy.array(b[1:])
            es = e_c[:,0]
            cs = e_c[:,1:]
            es = es[abs(cs).max(axis=1) > 1e-3]
            emax_by_l[l] = max(es.max(), emax_by_l[l])
            emin_by_l[l] = min(es.min(), emin_by_l[l])

            l_max1 = l_max + 1
            emin_by_l = numpy.array(emin_by_l[:l_max1])
            emax_by_l = numpy.array(emax_by_l[:l_max1])

            emax = numpy.sqrt(numpy.einsum('i,j->ij', emax_by_l, emax_by_l))
            emin = numpy.sqrt(numpy.einsum('i,j->ij', emin_by_l, emin_by_l))
            liljsum = numpy.arange(l_max1)[:,None] + numpy.arange(l_max1)
            emax_by_l = [emax[liljsum==ll].max() for ll in range(l_max1*2-1)]
            emin_by_l = [emin[liljsum==ll].min() for ll in range(l_max1*2-1)]
            # Tune emin and emax
            emin_by_l = numpy.array(emin_by_l) * 2 * lower_buf # *2 for alpha+alpha on same center
            emax_by_l = numpy.array(emax_by_l) * 2 * upper_buf  #/ (numpy.arange(l_max1*2-1)*.5+1)

            ns = numpy.log((emax_by_l+emin_by_l)/emin_by_l) / numpy.log(beta)
            nmaxs = numpy.ceil(np.log(emax_by_l) / numpy.log(beta))
            nmins = numpy.floor(np.log(emin_by_l) / numpy.log(beta))
            emin_by_l = beta**nmins
            ns = nmaxs-nmins+1
            etb = []
            for l, n in enumerate(numpy.ceil(ns).astype(int)):
                print(l, n, emin_by_l[l], beta)
                if n > 0 and l < lmax+1:
                    etb.append((l, n, emin_by_l[l], beta))
            newbasis[symb] = gto.expand_etbs(etb)

    return newbasis, ns, np.max(nmaxs) - nmaxs


def _conv_bas_from_aux_bas(basis, alphas, betas=None):
    if betas is None:
        betas = alphas
    alpha_max = np.max(alphas) + np.max(betas)
    alpha_min = np.min(alphas) + np.min(betas)

    newbasis = {}
    for symb in basis.keys():
        lmax = np.max([bas[0] for bas in basis[symb]])
        etb = []
        for l in range(lmax+1):
            gammas = [bas[1][0] for bas in basis[symb] if bas[0] == l]
            gmax = np.max(gammas)
            gmin = np.min(gammas)
            gmax = gmax * alpha_max / (gmax + alpha_max)
            gmin = gmin * alpha_min / (gmin + alpha_min)
            n_l = np.ceil(np.log(gmax / gmin) / np.log(beta)) + 1
            beta_l = np.exp( np.log(gmax / gmin) / (n_l - 1) )
            etb.append((l, n_l, gmin, beta_l))
        newbasis[symb] = gto.expand_etbs(etb)
    return newbasis

def convert_basis_to_cinp(basis):
    atoms = [a[0] for a in mol._atom]
    newbasis = {}
    for symb in basis.keys():
        lmax = np.max([bas[0] for bas in basis[symb]])
        #                 l ,               expnt
        newbasis[symb] = (np.zeros(lmax+1, dtype=np.int32), [])
        maxl_so_far = 0
        for b in basis[symb]:
            l = b[0]
            if l < maxl_so_far:
                raise ValueError
            maxl_so_far = l
            newbasis[symb][0][l] += 1
            newbasis[symb][1].append(b[1][0])
        lloc = np.append([0], np.cumsum(newbasis[symb][0]))
        lloc = np.ascontiguousarray(lloc.astype(np.int32))
        expnts = np.asarray(newbasis[symb][1], dtype=np.float64, order='C')
        newbasis[symb] = (lloc, expnts)
    inp_l = []
    inp_expnt = []
    for symb in atoms:
        inp_l.append(newbasis[symb][0])
        inp_expnt.append(newbasis[symb][1])
    inp_l = (ctypes.c_void_p*mol.natm)(*inp_l)
    inp_expnt = (ctypes.c_void_p*mol.natm)(*inp_expnt)
    return inp_l, inp_expnt


def get_atco_v2(mol, alphas, **settings):
    auxbasis, _, _ = aug_etb_for_cider_v2(mol, **settings)
    convbasis = _conv_bas_from_aux_bas(auxbasis, alphas)
    aux_inp_loc, aux_inp_expnt = convert_basis_to_cinp(mol, auxbasis)
    conv_inp_loc, conv_inp_expnt = convert_basis_to_cinp(mol, convbasis)
    atco = lib.c_null_ptr()
    rad_arr = grids.rad_arr
    rad_loc = grids.ra_loc
    setup_auxl_list(
        ctypes.by_ref(atco),
        ctypes.c_int(mol.natm),
        ctypes.c_int(0),
        lmaxs.ctypes.data_as(ctypes.c_void_p),
        rad_arr.ctypes.data_as(ctypes.c_void_p),
        rad_loc.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_void_p*mol.natm)(*rad_kgrids),
        (ctypes.c_void_p*mol.natm)(*conv_rad_grids),
        (ctypes.c_void_p*mol.natm)(*rad_kgrids),
        aux_inp_expnt, conv_inp_expnt,
        aux_inp_loc, conv_inp_loc,
    )
    return atco


def _gamma_loc_from_gammas(auxmol, alphas, gammas, gcoefs, lmax, lower=True, debug=False):
    atom_loc = gto.mole.aoslice_by_atom(auxmol)
    alpha_max = alphas[0]
    lambd = alphas[0] / alphas[1]
    ngamma = gammas.size
    ag_loc = [0]
    all_gammas = []
    all_gcoefs = []
    gamma_ids = []
    lmaxs = []
    ngamma_loc = [0]
    ia_lst = auxmol._bas[:,ATOM_OF]
    l_lst = auxmol._bas[:,ANG_OF]
    for ia in range(auxmol.natm):
        for l in range(lmax+1):
            if not debug:
                inds = auxmol._bas[:,PTR_EXP][np.logical_and(l_lst==l,ia_lst==ia)]
                gmax = np.max(auxmol._env[inds], initial=1e-16)
                nstart = int(np.floor(np.log(alpha_max / gmax) / np.log(lambd)))
                nstart = np.maximum(nstart, 0)
                nstart = np.minimum(nstart, ngamma)
                if nstart == ngamma:
                    lmaxs.append(l-1)
                    break
            else:
                nstart = 0
            # contains the gammas and gcoefs for all ls and atoms
            all_gammas.append(gammas[nstart:])
            all_gcoefs.append(gcoefs[l,nstart:])
            gamma_ids.append(np.arange(nstart,ngamma))
            # gamma_loc contains the offsets for atom/l indexes
            # for the start of gammas and gcoefs
            ngamma_loc.append(ngamma - nstart)
        else:
            lmaxs.append(lmax)
        assert lmaxs[-1] >= 0 and lmaxs[-1] <= lmax
        # ag_loc[ia] is the index in gamma_loc where
        # atom ia starts
        ag_loc.append(len(ngamma_loc)-1)
    ngamma_loc = np.array(ngamma_loc, dtype=np.int32)
    gamma_loc = np.cumsum(ngamma_loc).astype(np.int32)
    gamma_loc2 = np.cumsum(ngamma_loc*ngamma_loc).astype(np.int32)
    ig_loc = []
    ng_loc = [0]
    for ish in range(auxmol.nbas):
        ia = auxmol._bas[ish,ATOM_OF]
        il = auxmol._bas[ish,ANG_OF]
        ind = ag_loc[ia]+il
        # ig_loc[ish] is the location of the exponents/coefs in the all_gammas
        # and all_gcoefs vectors corresponding to atm(ish), l(ish)
        ig_loc.append(gamma_loc[ind])
        ng_loc.append(gamma_loc[ind+1] - gamma_loc[ind])
    # atc_loc[ish] is the location of the overlaps between shell index ish
    # and the conv basis for l(ish), at(ish)
    ng_loc = np.array(ng_loc, dtype=np.int32)
    atc_loc = np.cumsum(ng_loc).astype(np.int32)
    atc_loc2 = np.cumsum(ng_loc*ng_loc).astype(np.int32)
    lmaxs = np.array(lmaxs, dtype=np.int32)
    all_gammas = np.concatenate(all_gammas)
    all_gcoefs = np.concatenate(all_gcoefs)
    gamma_ids = np.concatenate(gamma_ids).astype(np.int32)
    ag_loc = np.array(ag_loc, dtype=np.int32)
    atco = lib.c_null_ptr()
    libcider.make_atc_set(
        ctypes.byref(atco),
        atc_loc.ctypes.data_as(ctypes.c_void_p),
        ag_loc.ctypes.data_as(ctypes.c_void_p),
        gamma_loc.ctypes.data_as(ctypes.c_void_p),
        gamma_loc2.ctypes.data_as(ctypes.c_void_p),
        all_gammas.ctypes.data_as(ctypes.c_void_p),
        all_gcoefs.ctypes.data_as(ctypes.c_void_p),
        lmaxs.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(auxmol.natm),
        ctypes.c_int(auxmol.nbas),
        ctypes.c_char(b'U') if lower else ctypes.c_char(b'L'),
        gammas.ctypes.data_as(ctypes.c_void_p),
        gamma_ids.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(gammas.size),
    )
    return atco, atc_loc[-1]


def get_lm_max(mol):
    lmax = max(np.max(mol._bas[:,ANG_OF]), 2)
    return (lmax+1) * (lmax+1)


def get_diff_gaussian_ovlp(alphas, alpha_norms, flip=False):
    potrf = scipy.linalg.lapack.get_lapack_funcs('potrf')
    trtri = scipy.linalg.lapack.get_lapack_funcs('trtri')

    alpha_s3 = (np.pi / (alphas + alphas[:,None]))**1.5
    alpha_s4 = alpha_s3.copy()
    alpha_s4[1:,1:] += alpha_s3[:-1,:-1]
    alpha_s4[1:,:]  -= alpha_s3[:-1,:]
    alpha_s4[:,1:]  -= alpha_s3[:,:-1]
    ad = np.diag(alpha_s4).copy()
    norm4 = ad
    alpha_s4 /= np.sqrt(ad * ad[:,None])
    alpha_s4 = alpha_s4
    transform = np.identity(alpha_s4.shape[0])
    for i in range(alpha_s4.shape[0] - 1):
        transform[i+1,i] = -1
    transform /= np.sqrt(ad)[:,None]

    if flip:
        print("FLIP")
        alpha_s4 = np.flip(alpha_s4, axis=0)
        alpha_s4 = np.flip(alpha_s4, axis=1)
    tmp, info = potrf(alpha_s4, lower=True)
    tmp, info = trtri(tmp, lower=True)
    tmp = np.tril(tmp)
    alpha_sinv4 = tmp.T.dot(tmp)
    if flip:
        alpha_sinv4 = np.flip(alpha_sinv4, axis=0)
        alpha_sinv4 = np.flip(alpha_sinv4, axis=1)
    transform = np.identity(tmp.shape[0])
    for i in range(tmp.shape[0]-1):
        transform[i+1,i] = -1
    transform /= alpha_norms
    transform /= np.sqrt(norm4[:,None])
    transform = transform

    return alpha_s4, alpha_sinv4, norm4, transform, transform.T.dot(tmp.T)


class ATCO():

    def __init__(self, atco_ptr):
        self._atco = atco_ptr

    def __del__(self):
        libcider.free_atc_set(self._atco)

    @property
    def bas(self):
        nbas = libcider.get_atco_nshl_conv(self._atco)
        bas = np.zeros((nbas,8), order='C', dtype=np.int32)
        libcider.get_atco_bas(
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.cast(self._atco, ctypes.c_void_p)
        )
        return bas

    @property
    def env(self):
        nbas = libcider.get_atco_nshl_conv(self._atco)
        env = np.zeros((2*nbas,), order='C', dtype=np.float64)
        libcider.get_atco_env(
            env.ctypes.data_as(ctypes.c_void_p), self._atco
        )
        return env

    @property
    def loc(self):
        nbas = libcider.get_atco_nshl_conv(self._atco)
        loc = np.zeros((nbas+1,), order='C', dtype=np.int32)
        libcider.get_atco_aux_loc(
            loc.ctypes.data_as(ctypes.c_void_p), self._atco,
        )
        return loc


class SimpleMole():

    def __init__(self, atm, bas, env, loc, atom_coords):
        self._atm = atm
        self._atom_coords = atom_coords
        self._bas = bas
        self._env = env
        self._loc = loc
        self.natm = atm.shape[0]
        self.nbas = bas.shape[0]
        atom_loc = gto.mole.aoslice_by_atom(self)
        atom_loc = np.ascontiguousarray(
            np.append(atom_loc[:, 0], [atom_loc[-1, 1]])
        ).astype(np.int32)
        self._atom_loc = atom_loc
        #ls = []
        #for i in range(self.nbas):
        #    for m in range(self._loc[i+1] - self._loc[i]):
        #        ls.append(self._bas[i, ANG_OF])
        #self._ls = np.array(ls)

    def nao_nr(self):
        return self._loc[-1]

    def get_atom_loc(self):
        return self._atom_loc

    def ao_loc_nr(self):
        return self._loc

    def atom_coords(self, unit='Bohr'):
        '''np.asarray([mol.atom_coords(i) for i in range(mol.natm)])'''
        return self._atom_coords

    def get_deriv_mol(self):
        assert self._bas.ndim == 2 and self._bas.shape[1] == 8
        bas = []
        for b in self._bas:
            if b[ANG_OF] == 0:
                continue
            else:
                bas.append(b)
        bas = np.asarray(bas, order='C', dtype=np.int32)
        bas[:, ANG_OF] -= 1
        bas[:, 2:5] = 0
        loc = [0]
        for b in bas:
            loc.append(loc[-1] + 2 * b[ANG_OF] + 1)
        assert loc[-1] == np.sum(2*bas[:,ANG_OF]+1)
        loc = np.asarray(loc, order='C', dtype=np.int32)
        return SimpleMole(
            self._atm.copy(),
            bas,
            self._env.copy(),
            loc,
            self._atom_coords.copy(),
        )


def _generate_atc_integrals3(auxmol, alphas, lmaxs, alpha_norms):
    # TODO for future development, do not call
    raise NotImplementedError
    #for ia in range(auxmol.natm):
    #    pass



def _generate_atc_integrals2(auxmol, alphas, gammas, lmax, alpha_norms=None,
                             do_cholesky=False, debug=False, vg=False):
    alpha_loc = np.empty(0)
    shl_loc = np.empty(0)
    atm, bas, env = auxmol._atm, auxmol._bas, auxmol._env
    natm = atm.shape[0]
    nbas = bas.shape[0]
    nalpha = alphas.shape[0]
    ngamma = gammas.shape[0]
    gcoefs = np.empty((lmax+1, gammas.shape[0]), order='C')
    for l in range(lmax+1):
        gcoefs[l] = gto.mole.gto_norm(l, gammas)
    atco, natc = _gamma_loc_from_gammas(auxmol, alphas, gammas, gcoefs, lmax, debug=debug)
    if vg:
        ovlp_mats = np.empty((8, nalpha, natc), order='C')
    else:
        ovlp_mats = np.empty((nalpha, natc, nalpha), order='C')
    atom_loc = gto.mole.aoslice_by_atom(auxmol)
    atom_loc = np.ascontiguousarray(np.append(atom_loc[:,0], [atom_loc[-1,1]])).astype(np.int32)
    if alpha_norms is None:
        alpha_norms = np.ones_like(alphas)
    else:
        alpha_norms = np.ascontiguousarray(alpha_norms)
        assert alpha_norms.size == alphas.size
    fn = (libcider.VXCgenerate_atc_integrals4 if vg
          else libcider.VXCgenerate_atc_integrals2)
    fn(
        ovlp_mats.ctypes.data_as(ctypes.c_void_p),
        alpha_loc.ctypes.data_as(ctypes.c_void_p),
        shl_loc.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p),
        atom_loc.ctypes.data_as(ctypes.c_void_p),
        alphas.ctypes.data_as(ctypes.c_void_p),
        atco,
        ctypes.c_int(nalpha),
        ctypes.c_int(ngamma),
        alpha_norms.ctypes.data_as(ctypes.c_void_p),
    )
    if do_cholesky:
        lower = True
        fn = (libcider.VXCsolve_atc_coefs4 if vg
              else libcider.VXCsolve_atc_coefs2)
        fn(
            ovlp_mats.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            atom_loc.ctypes.data_as(ctypes.c_void_p),
            atco,
            ctypes.c_int(nalpha),
            ctypes.c_int(ngamma),
        )
    atco = ATCO(atco)
    return ovlp_mats, atco


class _CiderConv:

    def __init__(self, mol, amin=DEFAULT_AMIN, amax=DEFAULT_AMAX,
                 lambd=DEFAULT_ALPHA_BETA, aux_beta=DEFAULT_AUX_BETA,
                 **kwargs):
        N = int(np.ceil(np.log(amax/amin) / np.log(lambd))) + 1
        lambd = np.exp(np.log(amax/amin) / (N-1))
        print('AMAX AMIN LAMBD N', amax, amin, lambd, N)
        self.alphas = amax / lambd**np.arange(N)
        self.alpha_lambd = lambd
        self.aux_beta = aux_beta
        self.cider_lmax = kwargs.get('cider_lmax') or CIDER_DEFAULT_LMAX
        self.set_auxmols(mol)
        self.onsite_direct = DEFAULT_ONSITE_DIRECT
        self.__dict__.update(kwargs)

    def set_auxmols(self, mol):
        raise NotImplementedError

    def transform_orbital_feat(self, p_au):
        raise NotImplementedError

    def transform_orbital_feat_fwd(self, p_au):
        return self.transform_orbital_feat(p_au)

    def transform_orbital_feat_bwd(self, p_au):
        return self.transform_orbital_feat(p_au)

    def get_cider_coefs_fwd(self, rho, derivs=False, **gg_kwargs):
        return self.get_cider_coefs(rho, derivs=derivs, **gg_kwargs)

    def get_cider_coefs_bwd(self, rho, derivs=False, **gg_kwargs):
        return self.get_cider_coefs(rho, derivs=derivs, **gg_kwargs)

    def setup_aux(self):
        raise NotImplementedError

    def get_cider_coefs(self, rho, derivs=False, **gg_kwargs):
        alphas = self.alphas
        outputs = gg_kwargs['get_exponent'](
            rho, a0=0.5*gg_kwargs['a0'], fac_mul=0.5*gg_kwargs['fac_mul'],
            amin=0.5*gg_kwargs['amin'],
            nspin=gg_kwargs.get('nspin') or 1
        )
        cider_exp, derivs = outputs[0], outputs[1:]
        p_ag = np.empty((len(alphas), rho.shape[-1]))
        dp_ag = np.empty((len(alphas), rho.shape[-1]))
        libcider.VXCfill_coefs(
            p_ag.ctypes.data_as(ctypes.c_void_p),
            dp_ag.ctypes.data_as(ctypes.c_void_p),
            cider_exp.ctypes.data_as(ctypes.c_void_p),
            alphas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(p_ag.shape[-1]),
            ctypes.c_int(p_ag.shape[0]),
        )
        if derivs == 'wv':
            return (p_ag, dp_ag, cider_exp) + derivs
        elif derivs:
            return (p_ag, dp_ag,) + derivs
        else:
            return p_ag, dp_ag


class _CiderConvExtra(_CiderConv):

    def _setup_atomic_cho_factors(self):
        atom_loc = self.auxmol.aoslice_by_atom().astype(np.int32)
        ao_loc = self.auxmol.ao_loc_nr()
        sizes = atom_loc[:,3] - atom_loc[:,2]
        self.aux_shls_slices = np.ascontiguousarray(
            np.append(atom_loc[:,0], [atom_loc[-1,1]]).astype(np.int32)
        )
        self.aux_c = np.empty(np.sum(sizes*sizes), order='C', dtype=np.float64)
        self.cho_offsets = np.ascontiguousarray(
            np.append([0], np.cumsum(sizes*sizes)).astype(np.int32)
        )
        self.aux_offsets = np.ascontiguousarray(
            np.append([0], np.cumsum(sizes)).astype(np.int32)
        )
        comp = 1
        hermi = 1
        self.auxmol_ovlp_cintopt = moleintor.make_cintopt(
            self.auxmol._atm, self.auxmol._bas, self.auxmol._env, 'int1e_ovlp_sph'
        )
        atm, bas, env = self.auxmol._atm, self.auxmol._bas, self.auxmol._env
        natm = atm.shape[0]
        nbas = bas.shape[0]
        if False:
            fn = libcider.VXCfill_atomic_cho_factors
        else:
            fn = libcider.VXCfill_atomic_cho_factors2
            self.aux_c[:] = 0
        fn(
            getattr(moleintor.libcgto, 'int1e_ovlp_sph'),
            self.aux_c.ctypes.data_as(ctypes.c_void_p),
            self.aux_shls_slices.ctypes.data_as(ctypes.c_void_p),
            self.cho_offsets.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp),
            ctypes.c_int(hermi),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            self.auxmol_ovlp_cintopt,
            atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            self.cho_char,
        )

    def _fill_cho_solves_(self, p_au):
        libcider.VXCfill_atomic_cho_solves(
            p_au.ctypes.data_as(ctypes.c_void_p),
            self.aux_c.ctypes.data_as(ctypes.c_void_p),
            self.aux_offsets.ctypes.data_as(ctypes.c_void_p),
            self.cho_offsets.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.auxmol.natm),
            ctypes.c_int(p_au.shape[0]),
            ctypes.c_int(p_au.shape[1]),
            self.cho_char,
        )


class _OptATCInts():

    def _multiply_atc_integrals(ni, p_ua, vg=False, vk=False, ig=0):
        Nalpha = len(ni.alphas)
        auxmol = ni.auxmol
        atom_loc = gto.mole.aoslice_by_atom(auxmol)
        atom_loc = np.ascontiguousarray(np.append(atom_loc[:,0], [atom_loc[-1,1]])).astype(np.int32)
        ao_loc = auxmol.ao_loc_nr().astype(np.int32)
        assert p_ua.flags.c_contiguous
        vgk = vg or vk
        if vgk:
            o_au = np.zeros((Nalpha, ni.bigaux.nao_nr()), order='C')
        else:
            o_au = np.zeros((Nalpha, ni.bigaux.nao_nr()), order='C')
        atm, bas, env = auxmol._atm, auxmol._bas, auxmol._env
        natm = atm.shape[0]
        nbas = bas.shape[0]
        fn = (libcider.VXCmultiply_atc_integrals4 if vgk
              else libcider.VXCmultiply_atc_integrals2)
        ovlp_mats = (ni.ovlp_mats[ig] if vgk
                     else ni.ovlp_mats)
        fn(
            o_au.ctypes.data_as(ctypes.c_void_p),
            p_ua.ctypes.data_as(ctypes.c_void_p),
            ovlp_mats.ctypes.data_as(ctypes.c_void_p),
            atom_loc.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ni.atco._atco,
            ctypes.c_int(ni.alphas.shape[0]),
            ctypes.c_int(ni.gammas.shape[0]),
            atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
        )
        if vg:
            return o_au.sum(0)[np.newaxis, :]
        else:
            return o_au

    def _multiply_atc_integrals_bwd(ni, o_au, vg=False, vk=False, ig=0):
        Nalpha = len(ni.alphas)
        auxmol = ni.auxmol
        atom_loc = gto.mole.aoslice_by_atom(auxmol)
        atom_loc = np.ascontiguousarray(np.append(atom_loc[:,0], [atom_loc[-1,1]])).astype(np.int32)
        ao_loc = auxmol.ao_loc_nr().astype(np.int32)
        assert o_au.flags.c_contiguous
        if vg:
            assert o_au.shape == (1, ni.bigaux.nao_nr(),)
        else:
            assert o_au.shape == (Nalpha, ni.bigaux.nao_nr())
        p_ua = np.zeros((auxmol.nao_nr(), Nalpha), order='C')
        atm, bas, env = auxmol._atm, auxmol._bas, auxmol._env
        natm = atm.shape[0]
        nbas = bas.shape[0]
        if vk:
            fn = libcider.VXCmultiply_atc_integrals4_bwdv2
        elif vg:
            fn = libcider.VXCmultiply_atc_integrals4_bwd
        else:
            fn = libcider.VXCmultiply_atc_integrals2_bwd
        ovlp_mats = (ni.ovlp_mats[ig] if vg
                     else ni.ovlp_mats)
        fn(
            o_au.ctypes.data_as(ctypes.c_void_p),
            p_ua.ctypes.data_as(ctypes.c_void_p),
            ovlp_mats.ctypes.data_as(ctypes.c_void_p),
            atom_loc.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ni.atco._atco,
            ctypes.c_int(ni.alphas.shape[0]),
            ctypes.c_int(ni.gammas.shape[0]),
            atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
        )
        return p_ua


class _CiderConvBPBase(_OptATCInts, _CiderConvExtra):

    def get_cider_auxmol(self, mol):
        if mol._atom is None:
            mol.build()
        uniq_atoms = set([a[0] for a in mol._atom])
        basis, _, _ = aug_etb_for_cider(
            mol, beta=self.aux_beta, lmax=self.cider_lmax,
            upper_fac=1.0, lower_fac=1.0
        )
        return gto.M(
            atom=mol.atom, basis=basis, spin=mol.spin,
            charge=mol.charge, unit=mol.unit,
        )

    def set_auxmols(self, mol):
        self.cho_char = ctypes.c_char(b'U')
        alpha_norms = (np.pi / (2 * self.alphas))**-0.75
        self.auxmol = self.get_cider_auxmol(mol)
        self._setup_atomic_cho_factors()
        self.alpha_norms = alpha_norms
        self.gammas = 2 * self.alphas
        mol = self.auxmol
        self.lmax = mol._bas[:,ANG_OF].max()
        self.build_bigaux()
        self.alpha_s4, self.alpha_sinv4, self.norm4, self.transform, self.alpha_ci = \
            get_diff_gaussian_ovlp(self.alphas, self.alpha_norms)
        self.alpha_sinv = self.alpha_ci.dot(self.alpha_ci.T)

    def build_bigaux(self):
        self.ovlp_mats, self.atco = _generate_atc_integrals2(
            self.auxmol, self.alphas, self.gammas, self.lmax,
            do_cholesky=True, alpha_norms=self.alpha_norms,
            debug=False
        )
        self.bigaux = SimpleMole(
            self.auxmol._atm, self.atco.bas,
            self.atco.env, self.atco.loc,
            self.auxmol.atom_coords()
        )
        self.auxmol_ovlp_cintopt = moleintor.make_cintopt(
            self.auxmol._atm, self.auxmol._bas,
            self.auxmol._env, 'int1e_ovlp_sph'
        )

    def contract_three(self, p_au):
        p_au = self.alpha_sinv4.dot(p_au)
        p_au = self.transform.T.dot(p_au)
        p_au = np.ascontiguousarray(p_au.copy(order='C'))
        t = time.monotonic()
        self._fill_cho_solves_(p_au)
        return p_au

    def contract_three_bwd(self, p_au):
        p_au = np.ascontiguousarray(p_au.copy(order='C'))
        self._fill_cho_solves_(p_au)
        p_au = self.transform.dot(p_au)
        p_au = self.alpha_sinv4.dot(p_au)
        return p_au

    def transform_orbital_feat(ni, p_au):
        p_au = p_au / np.sqrt(ni.norm4[:,None])
        p_ua = np.ascontiguousarray(ni.contract_three(p_au).T)
        o_au = ni._multiply_atc_integrals(p_ua)
        o_au = ni.transform.dot(o_au)
        o_au = ni.alpha_sinv4.dot(o_au)
        return o_au / np.sqrt(ni.norm4[:,None])

    def transform_orbital_feat_bwd(ni, p_au):
        p_au = p_au / np.sqrt(ni.norm4[:,None])
        p_au = ni.alpha_sinv4.dot(p_au)
        p_au = ni.transform.T.dot(p_au)
        q_ua = ni._multiply_atc_integrals_bwd(np.ascontiguousarray(p_au))
        q_au = ni.contract_three_bwd(q_ua.T)
        return q_au / np.sqrt(ni.norm4[:,None])

    def get_cider_coefs(self, rho, derivs=False, **gg_kwargs):
        res = super(_CiderConvBPBase, self).get_cider_coefs(rho, derivs=derivs, **gg_kwargs)
        res[0][1:] -= res[0][:-1]
        res[1][1:] -= res[1][:-1]
        return res


class CiderConvSpline(_CiderConvBPBase):

    def set_auxmols(self, mol):
        super(CiderConvSpline, self).set_auxmols(mol)

        aparam = 0.03
        #dparam = 0.03
        #N = 250
        #dparam = 0.05
        #N = 150
        dparam = 0.04
        N = 200
        Rg = aparam * (np.exp(dparam * np.arange(N)) - 1)
        nrad = Rg.size
        nbas = self.bigaux.nbas
        w_rsp = np.zeros((nrad, nbas, 4), order='C')
        shls_slice = (0, self.bigaux.nbas)
        libcider.compute_spline_maps(
            w_rsp.ctypes.data_as(ctypes.c_void_p),
            Rg.ctypes.data_as(ctypes.c_void_p),
            self.bigaux._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            self.bigaux._env.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*2)(*shls_slice),
            ctypes.c_int(nrad),
        )
        self.w_rsp = w_rsp
        self.aparam = aparam
        self.dparam = dparam
        self.rspline = Rg
        nlm = get_lm_max(self.bigaux)
        self.nlm = nlm

        self.is_num_ai_setup = False

    def set_num_ai(self, grids, _set_ga_loc=True):
        # Warning: _set_ga_loc=False for debug/training data generation only
        all_coords = grids.coords
        if _set_ga_loc:
            self._ga_loc = np.ascontiguousarray(
                grids.rad_loc[grids.ra_loc]
            )
        else:
            assert not self.onsite_direct
        if self.onsite_direct:
            self._ga_loc_ptr = self._ga_loc.ctypes.data_as(ctypes.c_void_p)
        else:
            self._ga_loc_ptr = lib.c_null_ptr()
        all_coords = np.ascontiguousarray(all_coords)
        assert all_coords.shape[1] == 3
        ngrids_tot = all_coords.shape[0]
        nrad = self.rspline.size
        natm = self.auxmol.natm
        assert all_coords.flags.c_contiguous
        atm_coords = np.ascontiguousarray(self.auxmol.atom_coords())
        self.num_ai = np.empty((natm, nrad), dtype=np.int32, order='C')
        libcider.compute_num_spline_contribs(
            self.num_ai.ctypes.data_as(ctypes.c_void_p),
            all_coords.ctypes.data_as(ctypes.c_void_p),
            atm_coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
            ctypes.c_int(natm),
            ctypes.c_int(ngrids_tot),
            ctypes.c_int(nrad),
            self._ga_loc_ptr,
        )
        self.all_coords = all_coords
        self.loc_ai = np.ascontiguousarray(
            (np.append(np.zeros((natm,1), dtype=np.int32),
                np.cumsum(self.num_ai, axis=1), axis=1)
            ).astype(np.int32)
        )
        self.ind_ord_fwd = np.empty(ngrids_tot, dtype=np.int32, order='C') #buffer
        self.ind_ord_bwd = np.empty(ngrids_tot, dtype=np.int32, order='C') #buffer
        self.num_i_tmp = np.empty(nrad, dtype=np.int32, order='C') # buffer
        self.coords_ord = np.empty((ngrids_tot, 3), dtype=np.float64, order='C') # buffer
        self.maxg = np.max(self.num_ai)

        self.is_num_ai_setup = True

    def _compute_spline_ind_order(self, a):
        if not self.is_num_ai_setup:
            raise RuntimeError
        ngrids_tot = self.all_coords.shape[0]
        nrad = self.num_ai.shape[1]
        libcider.compute_spline_ind_order(
            self.loc_ai[a].ctypes.data_as(ctypes.c_void_p),
            self.all_coords.ctypes.data_as(ctypes.c_void_p),
            self.auxmol.atom_coords()[a].ctypes.data_as(ctypes.c_void_p),
            self.coords_ord.ctypes.data_as(ctypes.c_void_p),
            self.ind_ord_fwd.ctypes.data_as(ctypes.c_void_p),
            self.ind_ord_bwd.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
            ctypes.c_int(ngrids_tot),
            ctypes.c_int(nrad),
            self._ga_loc_ptr,
            ctypes.c_int(a),
        )
        return self.coords_ord, self.ind_ord_fwd

    def eval_spline_bas_single(self, a, out=None):
        self._compute_spline_ind_order(a)
        ngrids = self.coords_ord.shape[0]
        nlm = self.nlm
        nrad = self.rspline.size
        atm_coord = np.ascontiguousarray(self.auxmol.atom_coords())[a]
        auxo_gi = np.ndarray((ngrids, 4*nlm), order='C', buffer=out)
        ind_g = np.zeros((ngrids), dtype=np.int32, order='C')
        if self.onsite_direct:
            ngrids -= self._ga_loc[a+1] - self._ga_loc[a]
        libcider.compute_spline_bas(
            auxo_gi.ctypes.data_as(ctypes.c_void_p),
            ind_g.ctypes.data_as(ctypes.c_void_p),
            self.coords_ord.ctypes.data_as(ctypes.c_void_p),
            atm_coord.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1),
            ctypes.c_int(ngrids),
            ctypes.c_int(nrad),
            ctypes.c_int(nlm),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
        )
        return auxo_gi, ind_g

    def eval_spline_bas_single_grad(self, a, out=None):
        self._compute_spline_ind_order(a)
        ngrids = self.coords_ord.shape[0]
        nlm = self.nlm
        nrad = self.rspline.size
        atm_coord = np.ascontiguousarray(self.auxmol.atom_coords())[a]
        ind_g = np.zeros((ngrids), dtype=np.int32, order='C')
        if self.onsite_direct:
            ngrids -= self._ga_loc[a+1] - self._ga_loc[a]
        auxo_vgi = np.ndarray((3, ngrids, 4*nlm), order='C', buffer=out)
        libcider.compute_spline_bas_deriv(
            auxo_vgi.ctypes.data_as(ctypes.c_void_p),
            ind_g.ctypes.data_as(ctypes.c_void_p),
            self.coords_ord.ctypes.data_as(ctypes.c_void_p),
            atm_coord.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1),
            ctypes.c_int(ngrids),
            ctypes.c_int(nrad),
            ctypes.c_int(nlm),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
        )
        return auxo_vgi, ind_g

    def contract_grad_terms(self, excsum, f_g, a, v):
        ngrids = f_g.size
        assert self._ga_loc is not None
        libcider.contract_grad_terms(
            excsum.ctypes.data_as(ctypes.c_void_p),
            f_g.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.auxmol.natm),
            ctypes.c_int(a),
            ctypes.c_int(v),
            ctypes.c_int(ngrids),
            self._ga_loc.ctypes.data_as(ctypes.c_void_p),
        )

    def eval_spline_bas(self, coords, out=None):
        natm = self.auxmol.natm
        ngrids = coords.shape[0]
        nlm = self.nlm
        nrad = self.rspline.size
        atm_coords = np.ascontiguousarray(self.auxmol.atom_coords())
        #ylm_a = get_ylm_atm(coords, atm_coords, self.nlm)
        #assert ylm_a.flags.c_contiguous
        assert coords.flags.c_contiguous
        auxo_agi = np.ndarray((natm, ngrids, 4*nlm), order='C', buffer=out)
        #auxo_agi = np.zeros((natm, ngrids, 4*nlm), order='C')
        ind_ag = np.zeros((natm, ngrids), dtype=np.int32, order='C')
        libcider.compute_spline_bas(
            auxo_agi.ctypes.data_as(ctypes.c_void_p),
            ind_ag.ctypes.data_as(ctypes.c_void_p),
            #ylm_a.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            atm_coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            ctypes.c_int(ngrids),
            ctypes.c_int(nrad),
            ctypes.c_int(nlm),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
        )
        return auxo_agi, ind_ag

    def _project_spline_conv(self, f, auxmol, w_rsp, c2s=True):
        t = time.monotonic()
        natm = self.auxmol.natm
        nrad = self.rspline.size
        nlm = self.nlm
        nalpha = f.shape[0]
        shls_slice = (0, auxmol.nbas)
        atom_loc = gto.mole.aoslice_by_atom(auxmol)
        atom_loc = np.ascontiguousarray(np.append(atom_loc[:,0], [atom_loc[-1,1]])).astype(np.int32)
        if c2s:
            fn = libcider.project_conv_onto_splines
            p_au = np.ascontiguousarray(f)
            nu = p_au.shape[1]
            f_qarlp = np.zeros((nalpha, natm, nrad, nlm, 4), order='C')
        else:
            fn = libcider.project_spline_onto_convs
            f_qarlp = np.ascontiguousarray(f)
            p_au = np.zeros((nalpha, auxmol.nao_nr()), order='C')
        fn(
            f_qarlp.ctypes.data_as(ctypes.c_void_p),
            w_rsp.ctypes.data_as(ctypes.c_void_p),
            p_au.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*2)(*shls_slice),
            atom_loc.ctypes.data_as(ctypes.c_void_p),
            auxmol._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(auxmol.nbas),
            auxmol.ao_loc_nr().ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nalpha),
            ctypes.c_int(natm),
            ctypes.c_int(nrad),
            ctypes.c_int(nlm),
        )
        if c2s:
            return f_qarlp
        else:
            return p_au

    def project_spline_conv(self, f, c2s=True):
        return self._project_spline_conv(
            f, self.bigaux, self.w_rsp, c2s=c2s,
        )

    def compute_mol_convs_(self, f, auxo_agi, ind_ag, pot=False):
        natm = self.auxmol.natm
        nrad = self.rspline.size
        nlm = self.nlm
        nalpha = f.shape[0]
        ngrids = ind_ag.shape[-1]
        if pot:
            #print("POT")
            f_qg = f
            f_qarlp = np.zeros((nalpha, natm, nrad, nlm, 4), order='C')
            fn = libcider.compute_pot_convs
        else:
            #print("FEAT")
            f_qarlp = f
            f_qg = np.zeros((nalpha, ngrids), order='C')
            fn = libcider.compute_mol_convs
        t = time.monotonic()
        fn(
            f_qg.ctypes.data_as(ctypes.c_void_p),
            f_qarlp.ctypes.data_as(ctypes.c_void_p),
            auxo_agi.ctypes.data_as(ctypes.c_void_p),
            ind_ag.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nalpha),
            ctypes.c_int(natm),
            ctypes.c_int(nrad),
            ctypes.c_int(ngrids),
            ctypes.c_int(nlm),
        )
        print("MOL CONV TIME", time.monotonic()-t)
        if pot:
            return f_qarlp
        else:
            return f_qg.T

    def compute_mol_convs_single_(self, a, f_in, f_out, auxo_gi, ind_g, pot=False):
        nrad = self.rspline.size
        nlm = self.nlm
        nalpha = f_in.shape[1]
        ngrids = ind_g.size
        if pot:
            f_gq = f_in
            f_rqlp = f_out
            fn = libcider.compute_pot_convs_single
            if self.onsite_direct and len(self.loc_ai) == 1:
                f_rqlp[:] = 0.0
                return f_rqlp
        else:
            f_gq = f_out
            f_rqlp = f_in
            fn = libcider.compute_mol_convs_single
            if self.onsite_direct and len(self.loc_ai) == 1:
                return f_gq
        fn(
            f_gq.ctypes.data_as(ctypes.c_void_p),
            f_rqlp.ctypes.data_as(ctypes.c_void_p),
            auxo_gi.ctypes.data_as(ctypes.c_void_p),
            self.loc_ai[a].ctypes.data_as(ctypes.c_void_p),
            self.ind_ord_fwd.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nalpha),
            ctypes.c_int(nrad),
            ctypes.c_int(ngrids),
            ctypes.c_int(nlm),
            ctypes.c_int(self.maxg),
        )
        if pot:
            return f_rqlp
        else:
            return f_gq

    def _call_angc_to_orb(self, theta_ga, auxmol, grids):
        atom_loc = gto.mole.aoslice_by_atom(auxmol)
        atom_loc = np.ascontiguousarray(np.append(atom_loc[:,0], [atom_loc[-1,1]])).astype(np.int32)
        ao_loc = auxmol.ao_loc_nr().astype(np.int32)
        atm, bas, env = auxmol._atm, auxmol._bas, auxmol._env
        natm, nbas = auxmol.natm, auxmol.nbas

        nalpha = theta_ga.shape[1]
        theta_rlmq = np.zeros(
            (grids.rad_arr.size, grids.nlm, nalpha), order='C', dtype=np.float64
        )
        p_ua = np.zeros((auxmol.nao_nr(), nalpha), order='C', dtype=np.float64)
        grids._reduce_angc_ylm(theta_rlmq, theta_ga, a2y=True)
        libcider.contract_rad_to_orb(
            theta_rlmq.ctypes.data_as(ctypes.c_void_p),
            p_ua.ctypes.data_as(ctypes.c_void_p),
            grids.ra_loc.ctypes.data_as(ctypes.c_void_p),
            grids.rad_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(theta_rlmq.shape[0]),
            ctypes.c_int(theta_rlmq.shape[1]),
            atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            atom_loc.ctypes.data_as(ctypes.c_void_p),
            self.alphas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nalpha),
        )
        return p_ua

    def _call_orb_to_angc(self, p_ua, auxmol, grids):
        atom_loc = gto.mole.aoslice_by_atom(auxmol)
        atom_loc = np.ascontiguousarray(np.append(atom_loc[:,0], [atom_loc[-1,1]])).astype(np.int32)
        ao_loc = auxmol.ao_loc_nr().astype(np.int32)
        atm, bas, env = auxmol._atm, auxmol._bas, auxmol._env
        natm, nbas = auxmol.natm, auxmol.nbas
        nalpha = p_ua.shape[-1]
        theta_rlmq = np.zeros(
            (grids.rad_arr.size, grids.nlm, nalpha), order='C', dtype=np.float64
        )
        #print(theta_rlmq.shape, theta_rlmq.dtype, theta_rlmq.flags.c_contiguous)
        #print(p_ua.shape, p_ua.dtype, p_ua.flags.c_contiguous)
        libcider.contract_orb_to_rad(
            theta_rlmq.ctypes.data_as(ctypes.c_void_p),
            p_ua.ctypes.data_as(ctypes.c_void_p),
            grids.ar_loc.ctypes.data_as(ctypes.c_void_p),
            grids.rad_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(theta_rlmq.shape[0]),
            ctypes.c_int(theta_rlmq.shape[1]),
            atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            atom_loc.ctypes.data_as(ctypes.c_void_p),
            self.alphas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nalpha),
        )
        theta_ga = np.zeros((grids.weights.size, nalpha), order='C')
        grids._reduce_angc_ylm(theta_rlmq, theta_ga, a2y=False)
        return theta_ga

    def transform_orbital_feat_fwd(self, grids, theta_ga):
        p_ua = self._call_angc_to_orb(theta_ga, self.auxmol, grids)
        out = super(CiderConvSpline, self).transform_orbital_feat(p_ua.T)
        if self.onsite_direct:
            f_gq = self._call_orb_to_angc(
                np.ascontiguousarray(out.T), self.bigaux, grids
            )
            return self.project_spline_conv(out, c2s=True), f_gq
        else:
            return self.project_spline_conv(out, c2s=True)

    def transform_orbital_feat_bwd(self, grids, f_qarlp, f_gq=None):
        out = self.project_spline_conv(f_qarlp, c2s=False)
        if self.onsite_direct:
            assert f_gq is not None
            out += self._call_angc_to_orb(f_gq, self.bigaux, grids).T
        p_au = super(CiderConvSpline, self).transform_orbital_feat_bwd(out)
        p_ua = np.ascontiguousarray(p_au.T)
        theta_ga = self._call_orb_to_angc(p_ua, self.auxmol, grids)
        return np.ascontiguousarray(theta_ga.T).T


class CiderConvSplineG(CiderConvSpline):
    """
    Experimental class that works for the version g features.
    """

    def _transform_orbital_feat_f(ni, p_au, ap_au):
        p_au = p_au / np.sqrt(ni.norm4[:,None])
        p_ua = np.ascontiguousarray(ni.contract_three(p_au).T)
        o_au = ni._multiply_atc_integrals(p_ua, vg=True)
        o1_au = ni._multiply_atc_integrals(p_ua, vg=True, ig=1)
        o2_au = ni._multiply_atc_integrals(p_ua, vg=True, ig=2)
        p_au = ap_au / np.sqrt(ni.norm4[:, None])
        p_ua = np.ascontiguousarray(ni.contract_three(p_au).T)
        o3_au = ni._multiply_atc_integrals(p_ua, vg=True, ig=2)
        #return o_au
        return np.concatenate([o_au, o2_au, o3_au, o1_au], axis=0)

    def _transform_orbital_feat_b(ni, p_au):
        q_ua = ni._multiply_atc_integrals_bwd(np.ascontiguousarray(p_au), vg=True)
        q_au = ni.contract_three_bwd(q_ua.T)
        return q_au / np.sqrt(ni.norm4[:,None])

    def transform_orbital_feat_fwd(self, grids, theta_ga, atheta_ga):
        p_ua = self._call_angc_to_orb(theta_ga, self.auxmol, grids)
        ap_ua = self._call_angc_to_orb(atheta_ga, self.auxmol, grids)
        out = self._transform_orbital_feat_f(p_ua.T, ap_ua.T)
        if self.onsite_direct:
            raise NotImplementedError
            #f_gq = self._call_orb_to_angc(
            #    np.ascontiguousarray(out.T), self.bigaux, grids
            #)
            #return self.project_spline_conv(out, c2s=True), f_gq
        else:
            return self.project_spline_conv(out, c2s=True)

    def transform_orbital_feat_bwd(self, grids, f_qarlp, f_gq=None):
        out = self.project_spline_conv(f_qarlp, c2s=False)
        if self.onsite_direct:
            assert f_gq is not None
            out += self._call_angc_to_orb(f_gq, self.bigaux, grids).T
        p_au = self._transform_orbital_feat_b(out)
        p_ua = np.ascontiguousarray(p_au.T)
        theta_ga = self._call_orb_to_angc(p_ua, self.auxmol, grids)
        return np.ascontiguousarray(theta_ga.T).T

    def _get_cider_coefs(self, rho, derivs=False, **gg_kwargs):
        alphas = self.alphas
        outputs = gg_kwargs['get_exponent'](
            rho, a0=gg_kwargs['a0'], fac_mul=gg_kwargs['fac_mul'],
            amin=gg_kwargs['amin'],
            nspin=gg_kwargs.get('nspin') or 1
        )
        cider_exp, derivs = outputs[0], outputs[1:]
        p_ag = np.empty((len(alphas), rho.shape[-1]))
        dp_ag = np.empty((len(alphas), rho.shape[-1]))
        libcider.VXCfill_coefs(
            p_ag.ctypes.data_as(ctypes.c_void_p),
            dp_ag.ctypes.data_as(ctypes.c_void_p),
            cider_exp.ctypes.data_as(ctypes.c_void_p),
            alphas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(p_ag.shape[-1]),
            ctypes.c_int(p_ag.shape[0]),
        )
        if derivs:
            return (p_ag, dp_ag,) + outputs
        else:
            return p_ag, dp_ag

    def get_cider_coefs(self, rho, derivs=False, **gg_kwargs):
        res = self._get_cider_coefs(rho, derivs=derivs, **gg_kwargs)
        res[0][1:] -= res[0][:-1]
        res[1][1:] -= res[1][:-1]
        return res

    def get_cider_coefs_bwd(self, rho, derivs=False, **gg_kwargs):
        raise NotImplementedError

    def build_bigaux(self):
        self.ovlp_mats, self.atco = _generate_atc_integrals2(
            self.auxmol, self.alphas, self.gammas, self.lmax,
            do_cholesky=True, alpha_norms=self.alpha_norms,
            debug=False, vg=True
        )
        self.bigaux = SimpleMole(
            self.auxmol._atm, self.atco.bas,
            self.atco.env, self.atco.loc,
            self.auxmol.atom_coords()
        )
        self.auxmol_ovlp_cintopt = moleintor.make_cintopt(
            self.auxmol._atm, self.auxmol._bas,
            self.auxmol._env, 'int1e_ovlp_sph'
        )


class CiderConvSplineH(CiderConvSplineG):

    def _transform_orbital_feat_f(ni, p_au, ap_au):
        p_au = p_au / np.sqrt(ni.norm4[:,None])
        p_ua = np.ascontiguousarray(ni.contract_three(p_au).T)
        o0_au = ni._multiply_atc_integrals(p_ua, vg=True, ig=0)
        o1_au = ni._multiply_atc_integrals(p_ua, vg=True, ig=1)
        p_au = ap_au / np.sqrt(ni.norm4[:, None])
        p_ua = np.ascontiguousarray(ni.contract_three(p_au).T)
        ao0_au = ni._multiply_atc_integrals(p_ua, vg=True, ig=0)
        ao1_au = ni._multiply_atc_integrals(p_ua, vg=True, ig=1)
        ao2_au = ni._multiply_atc_integrals(p_ua, vg=True, ig=2)
        #return o_au
        # F1, F1T, F2T, FX, FXT
        return np.concatenate([o0_au, ao0_au, ao2_au, o1_au, ao1_au], axis=0)

    def transform_orbital_feat_fwd(self, grids, theta_ga, atheta_ga):
        p_ua = self._call_angc_to_orb(theta_ga, self.auxmol, grids)
        ap_ua = self._call_angc_to_orb(atheta_ga, self.auxmol, grids)
        out = self._transform_orbital_feat_f(p_ua.T, ap_ua.T)
        if self.onsite_direct:
            raise NotImplementedError
            #f_gq = self._call_orb_to_angc(
            #    np.ascontiguousarray(out.T), self.bigaux, grids
            #)
            #return self.project_spline_conv(out, c2s=True), f_gq
        else:
            return self.project_spline_conv(out, c2s=True)

    def get_feat_from_f(self, f_qg, rho, a0, a_i):
        a15 = a0 ** 1.5
        p = 1.5
        nfeat = 7
        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        tfac = np.pi / 2 ** (2. / 3) / CFC
        feat_i = np.zeros((nfeat, rho.shape[-1]))
        feat_i[0] = a15 * f_qg[0]
        feat_i[1] = 2.0 / 3 * a15 * a0 * f_qg[2] * tfac
        f_xg = f_qg[3::2]
        ft_xg = f_qg[4::2]
        fxdx = np.einsum('qg,qg->g', f_xg, f_xg)
        fxtdxt = np.einsum('qg,qg->g', ft_xg, ft_xg)
        fxdxt = np.einsum('qg,qg->g', f_xg, ft_xg)
        feat_i[2] = a15 * a15 * a_i * fxdx
        feat_i[3] = a0 ** (3 * p + 2) * tfac * fxdx * f_qg[1]
        feat_i[4] = a0 ** (2 * p + 2) * tfac * tfac / (a_i + 1e-16) * fxtdxt
        feat_i[5] = a0 ** (p + 0.5) * np.einsum(
            'qg,qg->g', rho[1:4] / (rho[0] + 1e-16), f_xg
        )
        feat_i[6] = a0 ** (2 * p + 1) * tfac * fxdxt
        return feat_i, f_xg, ft_xg, fxdx, fxtdxt, fxdxt

    def fill_dfeat_from_f(
            self, dfeatj_ig, df_qg, rho, drho, a0, a_i, dadphi,
            feat_i, f_qg, f_xg, ft_xg, fxdx, fxtdxt
    ):
        a15 = a0 ** 1.5
        p = 1.5
        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        tfac = np.pi / 2 ** (2. / 3) / CFC
        dfeatj_ig[0] = a15 * df_qg[0]
        dfeatj_ig[1] = (2.0 / 3 * a15 * a0 * tfac) * df_qg[2]
        df_xg = df_qg[3::2]
        dft_xg = df_qg[4::2]
        dfxdx = 2 * np.einsum('qg,qg->g', f_xg, df_xg)
        dfxtdxt = 2 * np.einsum('qg,qg->g', ft_xg, dft_xg)
        dfxdxt = (np.einsum('qg,qg->g', f_xg, dft_xg) +
                  np.einsum('qg,qg->g', df_xg, ft_xg))
        dfeatj_ig[2] = a15 * a15 * (dadphi * fxdx + a_i * dfxdx)
        dfeatj_ig[3] = df_qg[1] * fxdx + f_qg[1] * dfxdx
        dfeatj_ig[3] *= a0 ** (3 * p + 2) * tfac
        dfeatj_ig[4] = dfxtdxt / (a_i + 1e-16) - fxtdxt / (a_i + 1e-16) ** 2 * dadphi
        dfeatj_ig[4] *= a0 ** (2 * p + 2) * tfac * tfac
        dfeatj_ig[5] = a0 ** (p + 0.5) * (np.einsum(
            'qg,qg->g', drho[1:4] / (rho[0] + 1e-16), f_xg
        ) + np.einsum(
            'qg,qg->g', rho[1:4] / (rho[0] + 1e-16), df_xg
        ))
        dfeatj_ig[5] -= feat_i[5] / (rho[0] + 1e-16) * drho[0]
        dfeatj_ig[6] = a0 ** (2 * p + 1) * tfac * dfxdxt


class _VHMixin():

    def _fill_deriv_coeff(self, f, d, bwd=False):
        nlm = self._gaunt_coeff.shape[-1]
        assert f.flags.c_contiguous
        assert d.flags.c_contiguous
        assert self._gaunt_coeff.flags.c_contiguous
        assert self.bigaux.get_atom_loc().flags.c_contiguous
        assert self.bigaux.ao_loc_nr().flags.c_contiguous
        assert self.derivaux.get_atom_loc().flags.c_contiguous
        assert self.derivaux.ao_loc_nr().flags.c_contiguous
        fn = libcider.fill_gaussian_deriv_coeff_bwd if bwd \
             else libcider.fill_gaussian_deriv_coeff
        fn(
            f.ctypes.data_as(ctypes.c_void_p),
            d.ctypes.data_as(ctypes.c_void_p),
            self._gaunt_coeff.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nlm),
            self.bigaux.get_atom_loc().ctypes.data_as(ctypes.c_void_p),
            self.bigaux.ao_loc_nr().ctypes.data_as(ctypes.c_void_p),
            self.bigaux._atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.bigaux.natm),
            self.bigaux._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.bigaux.nbas),
            self.bigaux._env.ctypes.data_as(ctypes.c_void_p),
            self.derivaux.get_atom_loc().ctypes.data_as(ctypes.c_void_p),
            self.derivaux.ao_loc_nr().ctypes.data_as(ctypes.c_void_p),
            self.derivaux._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.derivaux.nbas),
            self.derivaux._env.ctypes.data_as(ctypes.c_void_p),
        )

    def _fill_onsite_grad_nldf(self, f_gq, gxyz, bwd=False):
        atom_coords = np.ascontiguousarray(
            self.auxmol.atom_coords()
        )
        fn = libcider.add_vh_grad_onsite_bwd if bwd \
             else libcider.add_vh_grad_onsite
        assert f_gq.flags.c_contiguous
        fn(
            f_gq.ctypes.data_as(ctypes.c_void_p),
            self.all_coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(len(atom_coords)),
            atom_coords.ctypes.data_as(ctypes.c_void_p),
            self._ga_loc_ptr,
            ctypes.c_int(gxyz[0]),
            ctypes.c_int(gxyz[1]),
            ctypes.c_int(gxyz[2]),
            ctypes.c_int(gxyz[3]),
            ctypes.c_int(f_gq.shape[1]),
        )

    def _fill_nldf_grad_term(self, f, a, gxyz, bwd=False):
        acoord = np.ascontiguousarray(self.auxmol.atom_coords()[a])
        assert self.all_coords.flags.c_contiguous and self.all_coords.shape[-1] == 3
        assert self.all_coords.shape[0] == f.shape[0]
        assert f.flags.c_contiguous
        fn = libcider.add_vh_grad_term_bwd if bwd else libcider.add_vh_grad_term
        fn(
            f.ctypes.data_as(ctypes.c_void_p),
            self.all_coords.ctypes.data_as(ctypes.c_void_p),
            acoord.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(f.shape[0]),
            ctypes.c_int(gxyz[0]),
            ctypes.c_int(gxyz[1]),
            ctypes.c_int(gxyz[2]),
            ctypes.c_int(gxyz[3]),
            ctypes.c_int(f.shape[1]),
        )

    def project_deriv_spline_conv(self, f, c2s=True):
        return self._project_spline_conv(
            f, self.derivaux, self.dw_rsp, c2s=c2s,
        )


class CiderConvSplineHv2(_VHMixin, CiderConvSplineG):

    def set_auxmols(self, mol):
        super(CiderConvSplineHv2, self).set_auxmols(mol)
        self._gaunt_coeff = get_deriv_ylm_coeff(self.lmax)
        self.derivaux = self.bigaux.get_deriv_mol()
        nbas = self.derivaux.nbas
        nrad = self.w_rsp.shape[0]
        w_rsp = np.zeros((nrad, nbas, 4), order='C')
        shls_slice = (0, nbas)
        libcider.compute_spline_maps(
            w_rsp.ctypes.data_as(ctypes.c_void_p),
            self.rspline.ctypes.data_as(ctypes.c_void_p),
            self.derivaux._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            self.derivaux._env.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int * 2)(*shls_slice),
            ctypes.c_int(nrad),
        )
        self.dw_rsp = w_rsp

    def _transform_orbital_feat_f(ni, p_au, ap_au):
        out_au = np.zeros((7, ni.bigaux.nao_nr()))
        outd_au = np.ascontiguousarray(
            np.zeros((6, ni.derivaux.nao_nr()))
        )
        p_au = p_au / np.sqrt(ni.norm4[:,None])
        p_ua = np.ascontiguousarray(ni.contract_three(p_au).T)
        # o0_au
        out_au[0:1] = ni._multiply_atc_integrals(p_ua, vg=True, ig=0)
        # o1_au
        out_au[5:6] = ni._multiply_atc_integrals(p_ua, vg=True, ig=1)
        # o3_au
        out_au[3:4] = ni._multiply_atc_integrals(p_ua, vg=True, ig=3)
        p_au = ap_au / np.sqrt(ni.norm4[:, None])
        p_ua = np.ascontiguousarray(ni.contract_three(p_au).T)
        # ao0_au
        out_au[1:2] = ni._multiply_atc_integrals(p_ua, vg=True, ig=0)
        # ao1_au
        out_au[6:7] = ni._multiply_atc_integrals(p_ua, vg=True, ig=1)
        # ao2_au
        out_au[2:3] = ni._multiply_atc_integrals(p_ua, vg=True, ig=2)
        # ao3_au
        out_au[4:5] = ni._multiply_atc_integrals(p_ua, vg=True, ig=3)
        ni._fill_deriv_coeff(out_au[5], outd_au[:3])
        ni._fill_deriv_coeff(out_au[6], outd_au[3:])
        # F1, F1T, F2T, FX_rad, FXT_rad, FX_ang, FXT_ang, (FX_prim, FXT_prim)
        return out_au[:5], outd_au

    def _transform_orbital_feat_b(ni, out_au, outd_au):
        naux = ni.auxmol.nao_nr()
        out_buf = np.zeros((2, out_au.shape[1]))
        ni._fill_deriv_coeff(out_buf[0], outd_au[:3], bwd=True)
        ni._fill_deriv_coeff(out_buf[1], outd_au[3:], bwd=True)
        shape = (naux, len(ni.alphas))
        p_ua = np.zeros(shape)
        ap_ua = np.zeros(shape)
        p_ua[:] += ni._multiply_atc_integrals_bwd(out_au[0:1], vg=True, ig=0)
        p_ua[:] += ni._multiply_atc_integrals_bwd(out_buf[0:1], vg=True, ig=1)
        p_ua[:] += ni._multiply_atc_integrals_bwd(out_au[3:4], vg=True, ig=3)
        ap_ua[:] += ni._multiply_atc_integrals_bwd(out_au[1:2], vg=True, ig=0)
        ap_ua[:] += ni._multiply_atc_integrals_bwd(out_buf[1:2], vg=True, ig=1)
        ap_ua[:] += ni._multiply_atc_integrals_bwd(out_au[2:3], vg=True, ig=2)
        ap_ua[:] += ni._multiply_atc_integrals_bwd(out_au[4:5], vg=True, ig=3)
        p_au = ni.contract_three_bwd(p_ua.T) / np.sqrt(ni.norm4[:, None])
        ap_au = ni.contract_three_bwd(ap_ua.T) / np.sqrt(ni.norm4[:, None])
        return p_au, ap_au

    def transform_orbital_feat_fwd(self, grids, theta_ga, atheta_ga):
        p_ua = self._call_angc_to_orb(theta_ga, self.auxmol, grids)
        ap_ua = self._call_angc_to_orb(atheta_ga, self.auxmol, grids)
        # out has shape (nfeat, norb)
        out, outd = self._transform_orbital_feat_f(p_ua.T, ap_ua.T)
        if self.onsite_direct:
            f_gq = np.zeros((grids.weights.size, 11))
            f_gq[:, :5] = self._call_orb_to_angc(
                np.ascontiguousarray(out.T), self.bigaux, grids
            )
            f_gq[:, 5:] = self._call_orb_to_angc(
                np.ascontiguousarray(outd.T), self.derivaux, grids
            )
            self._fill_onsite_grad_nldf(f_gq, (3, 5, 6, 7))
            self._fill_onsite_grad_nldf(f_gq, (4, 8, 9, 10))
            out1 = self.project_spline_conv(out, c2s=True)
            out2 = self.project_deriv_spline_conv(outd, c2s=True)
            return np.concatenate([out1, out2], axis=0), f_gq
        else:
            out1 = self.project_spline_conv(out, c2s=True)
            out2 = self.project_deriv_spline_conv(outd, c2s=True)
            return np.concatenate([out1, out2], axis=0)

    def transform_orbital_feat_bwd(self, grids, f_qarlp, f_gq=None):
        assert f_qarlp.flags.c_contiguous
        assert f_qarlp.shape[0] == 11
        out = self.project_spline_conv(f_qarlp[:5], c2s=False)
        outd = self.project_deriv_spline_conv(f_qarlp[5:], c2s=False)
        if self.onsite_direct:
            assert f_gq is not None
            self._fill_onsite_grad_nldf(f_gq, (3, 5, 6, 7), bwd=True)
            self._fill_onsite_grad_nldf(f_gq, (4, 8, 9, 10), bwd=True)
            out += self._call_angc_to_orb(
                np.ascontiguousarray(f_gq[:, :5]), self.bigaux, grids
            ).T
            outd += self._call_angc_to_orb(
                np.ascontiguousarray(f_gq[:, 5:]), self.derivaux, grids
            ).T
        p_au, ap_au = self._transform_orbital_feat_b(out, outd)
        p_ua = np.ascontiguousarray(p_au.T)
        ap_ua = np.ascontiguousarray(ap_au.T)
        theta_ga = self._call_orb_to_angc(p_ua, self.auxmol, grids)
        atheta_ga = self._call_orb_to_angc(ap_ua, self.auxmol, grids)
        return (np.ascontiguousarray(theta_ga.T).T,
                np.ascontiguousarray(atheta_ga.T).T)

    def compute_mol_convs_single_(self, a, f_in, f_out, auxo_gi, ind_g, pot=False):
        #f_out[:, 3:5] = 0.0
        if pot:
            self._fill_nldf_grad_term(f_in, a, (3, 5, 6, 7), bwd=True)
            self._fill_nldf_grad_term(f_in, a, (4, 8, 9, 10), bwd=True)
            super(CiderConvSplineHv2, self).compute_mol_convs_single_(
                a, f_in, f_out, auxo_gi, ind_g, pot=pot
            )
        else:
            super(CiderConvSplineHv2, self).compute_mol_convs_single_(
                a, f_in, f_out, auxo_gi, ind_g, pot=pot
            )
            self._fill_nldf_grad_term(f_out, a, (3, 5, 6, 7), bwd=False)
            self._fill_nldf_grad_term(f_out, a, (4, 8, 9, 10), bwd=False)

    @staticmethod
    def get_feat_from_f(f_qg, rho, a0, a_i, nspin=1):
        # 0,  1,   2,   3,      4,       5:8,    8:11
        # F1, F1T, F2T, FX_rad, FXT_rad, FX_ang, FXT_ang
        ddd = 1e-10
        a15 = a0 ** 1.5
        p = 1.5
        nfeat = 7
        #f_qg *= nspin
        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        tfac = np.pi / 2 ** (2. / 3) / CFC
        feat_i = np.zeros((nfeat, rho.shape[-1]))
        feat_i[0] = a15 * f_qg[0]
        feat_i[1] = 2.0 / 3 * a15 * a0 * f_qg[2] * tfac
        f_xg = f_qg[5:8]
        ft_xg = f_qg[8:11]
        fxdx = np.einsum('qg,qg->g', f_xg, f_xg)
        fxtdxt = np.einsum('qg,qg->g', ft_xg, ft_xg)
        fxdxt = np.einsum('qg,qg->g', f_xg, ft_xg)
        feat_i[2] = a15 * a15 * a_i * fxdx
        feat_i[3] = a0 ** (3 * p + 2) * tfac * fxdx * f_qg[1]
        feat_i[4] = a0 ** (2 * p + 2) * tfac * tfac / (a_i + ddd) * fxtdxt
        feat_i[5] = a0 ** (p + 0.5) * np.einsum(
            'qg,qg->g', rho[1:4] / (rho[0] + ddd), f_xg
        )
        feat_i[6] = a0 ** (2 * p + 1) * tfac * fxdxt
        #feat_i /= nspin
        feat_i[[2, 4, 6]] *= nspin
        feat_i[3] *= nspin * nspin
        return feat_i, f_xg, ft_xg, fxdx, fxtdxt, fxdxt

    @staticmethod
    def get_vfeat_scf(vfeat_i, rho, a0, a_i, feat_i, f_qg,
                      f_xg, ft_xg, fxdx, fxtdxt, nspin=1):
        #vfeat_i /= nspin
        #feat_i *= nspin
        vfeat_i[[2, 4, 6]] *= nspin
        vfeat_i[3] *= nspin * nspin
        feat_i[[2, 4, 6]] /= nspin
        feat_i[3] /= nspin * nspin
        ddd = 1e-10
        a15 = a0 ** 1.5
        p = 1.5
        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        tfac = np.pi / 2 ** (2. / 3) / CFC
        vf_qg = np.zeros_like(f_qg)
        C1 = 2.0 / 3 * a15 * a0 * tfac
        C2 = a15 * a15
        C3 = a0 ** (3 * p + 2) * tfac
        C4 = a0 ** (2 * p + 2) * tfac * tfac
        C5 = a0 ** (p + 0.5)
        C6 = a0 ** (2 * p + 1) * tfac
        vf_qg[0] = a15 * vfeat_i[0]
        vf_qg[2] = C1 * vfeat_i[1]
        vfxdx = (C2 * a_i * vfeat_i[2]
                 + C3 * vfeat_i[3] * f_qg[1])
        vf_qg[1] = C3 * vfeat_i[3] * fxdx
        vf_qg[5:8] = (2 * vfxdx * f_xg
                      + C6 * vfeat_i[6] * ft_xg
                      + C5 * vfeat_i[5] * rho[1:4] / (rho[0] + ddd))
        vf_qg[8:11] = (2 * C4 / (a_i + ddd) * vfeat_i[4] * ft_xg
                       + C6 * vfeat_i[6] * f_xg)
        deda = (C2 * vfeat_i[2] * fxdx
                - vfeat_i[4] * feat_i[4] / (a_i + ddd))
        dedrho = -1 * vfeat_i[5] * feat_i[5] / (rho[0] + ddd)
        dedgrad = C5 * f_xg / (rho[0] + ddd) * vfeat_i[5]
        #vf_qg *= nspin
        #feat_i /= nspin
        return vf_qg, deda, dedrho, dedgrad

    @staticmethod
    def fill_dfeat_from_f(
            dfeatj_ig, df_qg, rho, drho, a0, a_i, dadphi,
            feat_i, f_qg, f_xg, ft_xg, fxdx, fxtdxt
    ):
        a15 = a0 ** 1.5
        p = 1.5
        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        tfac = np.pi / 2 ** (2. / 3) / CFC
        dfeatj_ig[0] = a15 * df_qg[0]
        dfeatj_ig[1] = (2.0 / 3 * a15 * a0 * tfac) * df_qg[2]
        df_xg = df_qg[5:8]
        dft_xg = df_qg[8:11]
        dfxdx = 2 * np.einsum('qg,qg->g', f_xg, df_xg)
        dfxtdxt = 2 * np.einsum('qg,qg->g', ft_xg, dft_xg)
        dfxdxt = (np.einsum('qg,qg->g', f_xg, dft_xg) +
                  np.einsum('qg,qg->g', df_xg, ft_xg))
        dfeatj_ig[2] = a15 * a15 * (dadphi * fxdx + a_i * dfxdx)
        dfeatj_ig[3] = df_qg[1] * fxdx + f_qg[1] * dfxdx
        dfeatj_ig[3] *= a0 ** (3 * p + 2) * tfac
        dfeatj_ig[4] = dfxtdxt / (a_i + 1e-16) - fxtdxt / (a_i + 1e-16) ** 2 * dadphi
        dfeatj_ig[4] *= a0 ** (2 * p + 2) * tfac * tfac
        dfeatj_ig[5] = a0 ** (p + 0.5) * (np.einsum(
            'qg,qg->g', drho[1:4] / (rho[0] + 1e-16), f_xg
        ) + np.einsum(
            'qg,qg->g', rho[1:4] / (rho[0] + 1e-16), df_xg
        ))
        dfeatj_ig[5] -= feat_i[5] / (rho[0] + 1e-16) * drho[0]
        dfeatj_ig[6] = a0 ** (2 * p + 1) * tfac * dfxdxt


class CiderConvSplineI(_VHMixin, CiderConvSplineG):

    def set_auxmols(self, mol):
        super(CiderConvSplineI, self).set_auxmols(mol)
        self._gaunt_coeff = get_deriv_ylm_coeff(self.lmax)
        self.derivaux = self.bigaux.get_deriv_mol()
        nbas = self.derivaux.nbas
        nrad = self.w_rsp.shape[0]
        w_rsp = np.zeros((nrad, nbas, 4), order='C')
        shls_slice = (0, nbas)
        libcider.compute_spline_maps(
            w_rsp.ctypes.data_as(ctypes.c_void_p),
            self.rspline.ctypes.data_as(ctypes.c_void_p),
            self.derivaux._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            self.derivaux._env.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int * 2)(*shls_slice),
            ctypes.c_int(nrad),
        )
        self.dw_rsp = w_rsp

    def _transform_orbital_feat_f(ni, p_au):
        out_au = np.zeros((5, ni.bigaux.nao_nr()))
        outd_au = np.ascontiguousarray(
            np.zeros((3, ni.derivaux.nao_nr()))
        )
        p_au = p_au / np.sqrt(ni.norm4[:,None])
        p_ua = np.ascontiguousarray(ni.contract_three(p_au).T)
        out_au[0:1] = ni._multiply_atc_integrals(p_ua, vg=True, ig=0)
        out_au[4:5] = ni._multiply_atc_integrals(p_ua, vg=True, ig=4)
        out_au[3:4] = ni._multiply_atc_integrals(p_ua, vg=True, ig=7)
        out_au[2:3] = ni._multiply_atc_integrals(p_ua, vg=True, ig=5)
        out_au[1:2] = ni._multiply_atc_integrals(p_ua, vg=True, ig=6)
        ni._fill_deriv_coeff(out_au[4], outd_au)
        return out_au[:4], outd_au

    def _transform_orbital_feat_b(ni, out_au, outd_au):
        naux = ni.auxmol.nao_nr()
        out_buf = np.zeros((1, out_au.shape[1]))
        ni._fill_deriv_coeff(out_buf[0], outd_au, bwd=True)
        shape = (naux, len(ni.alphas))
        p_ua = np.zeros(shape)
        p_ua[:] += ni._multiply_atc_integrals_bwd(out_au[0:1], vg=True, ig=0)
        p_ua[:] += ni._multiply_atc_integrals_bwd(out_buf[0:1], vg=True, ig=4)
        p_ua[:] += ni._multiply_atc_integrals_bwd(out_au[3:4], vg=True, ig=7)
        p_ua[:] += ni._multiply_atc_integrals_bwd(out_au[2:3], vg=True, ig=5)
        p_ua[:] += ni._multiply_atc_integrals_bwd(out_au[1:2], vg=True, ig=6)
        p_au = ni.contract_three_bwd(p_ua.T) / np.sqrt(ni.norm4[:, None])
        return p_au

    def transform_orbital_feat_fwd(self, grids, theta_ga):
        p_ua = self._call_angc_to_orb(theta_ga, self.auxmol, grids)
        # out has shape (nfeat, norb)
        out, outd = self._transform_orbital_feat_f(p_ua.T)
        if self.onsite_direct:
            f_gq = np.zeros((grids.weights.size, 7))
            f_gq[:, :4] = self._call_orb_to_angc(
                np.ascontiguousarray(out.T), self.bigaux, grids
            )
            f_gq[:, 4:] = self._call_orb_to_angc(
                np.ascontiguousarray(outd.T), self.derivaux, grids
            )
            self._fill_onsite_grad_nldf(f_gq, (3, 4, 5, 6))
            out1 = self.project_spline_conv(out, c2s=True)
            out2 = self.project_deriv_spline_conv(outd, c2s=True)
            return np.concatenate([out1, out2], axis=0), f_gq
        else:
            out1 = self.project_spline_conv(out, c2s=True)
            out2 = self.project_deriv_spline_conv(outd, c2s=True)
            return np.concatenate([out1, out2], axis=0)

    def transform_orbital_feat_bwd(self, grids, f_qarlp, f_gq=None):
        assert f_qarlp.flags.c_contiguous
        assert f_qarlp.shape[0] == 7
        out = self.project_spline_conv(f_qarlp[:4], c2s=False)
        outd = self.project_deriv_spline_conv(f_qarlp[4:], c2s=False)
        if self.onsite_direct:
            assert f_gq is not None
            self._fill_onsite_grad_nldf(f_gq, (3, 4, 5, 6), bwd=True)
            out += self._call_angc_to_orb(
                np.ascontiguousarray(f_gq[:, :4]), self.bigaux, grids
            ).T
            outd += self._call_angc_to_orb(
                np.ascontiguousarray(f_gq[:, 4:]), self.derivaux, grids
            ).T
        p_au = self._transform_orbital_feat_b(out, outd)
        p_ua = np.ascontiguousarray(p_au.T)
        theta_ga = self._call_orb_to_angc(p_ua, self.auxmol, grids)
        return np.ascontiguousarray(theta_ga.T).T

    def compute_mol_convs_single_(self, a, f_in, f_out, auxo_gi, ind_g, pot=False):
        #f_out[:, 3:5] = 0.0
        if pot:
            self._fill_nldf_grad_term(f_in, a, (3, 4, 5, 6), bwd=True)
            CiderConvSplineG.compute_mol_convs_single_(
                self, a, f_in, f_out, auxo_gi, ind_g, pot=pot
            )
        else:
            CiderConvSplineG.compute_mol_convs_single_(
                self, a, f_in, f_out, auxo_gi, ind_g, pot=pot
            )
            self._fill_nldf_grad_term(f_out, a, (3, 4, 5, 6), bwd=False)

    @staticmethod
    def get_feat_from_f(f_qg, rho, a0, a_i, flapl_prho, nspin=1):
        # 0,  1,   2,   3,      4,       5:8,    8:11
        # F1, F1T, F2T, FX_rad, FXT_rad, FX_ang, FXT_ang
        ddd = 1e-10
        a15 = a0 ** 1.5
        nfeat = 8
        feat_i = np.zeros((nfeat, rho.shape[-1]))
        feat_i[0] = a15 * f_qg[0]
        feat_i[1] = 2.0 / 3 * a15 * f_qg[1]
        feat_i[2] = a15 / (a_i + ddd) * f_qg[2]
        f_xg = f_qg[4:7]
        fxdx = np.einsum('qg,qg->g', f_xg, f_xg)
        feat_i[3] = a15 * fxdx / (f_qg[2] + ddd)
        feat_i[4] = a15 * a15 * fxdx / (a_i + ddd)
        fgdx = np.einsum(
            'qg,qg->g', rho[1:4] / (rho[0] + ddd), f_xg
        )
        feat_i[5] = a15 * fgdx / (a_i + ddd)

        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        pow1 = (2 * FRAC_LAPL_POWER / 3.0) + 1
        pow2 = (4 * FRAC_LAPL_POWER / 3.0) + 1
        feat_i[6] = flapl_prho[1] / (2 * CFC * (rho[0] + ddd) ** pow2 + ddd)
        feat_i[7] = flapl_prho[2] / ((rho[0] + ddd) ** pow1)

        feat_i[4] *= nspin
        feat_i[6] /= nspin**pow2
        feat_i[7] /= nspin**pow1
        return feat_i, f_xg, fxdx, fgdx

    @staticmethod
    def get_vfeat_scf(vfeat_i, rho, a0, a_i, feat_i, f_qg,
                      f_xg, flapl_rho, nspin=1):
        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        pow1 = (2 * FRAC_LAPL_POWER / 3.0) + 1
        pow2 = (4 * FRAC_LAPL_POWER / 3.0) + 1
        vfeat_i[4] *= nspin
        vfeat_i[6] /= nspin ** pow2
        vfeat_i[7] /= nspin ** pow1
        feat_i[4] /= nspin
        feat_i[6] *= nspin ** pow2
        feat_i[7] *= nspin ** pow1
        ddd = 1e-10
        a15 = a0 ** 1.5
        vf_qg = np.zeros_like(f_qg)
        vf_qg[0] = a15 * vfeat_i[0]
        vf_qg[1] = 2.0 / 3 * a15 * vfeat_i[1]
        vf_qg[2] = a15 / (a_i + ddd) * vfeat_i[2]
        vf_qg[2] -= feat_i[3] * vfeat_i[3] / (f_qg[2] + ddd)
        vfxdx = (a15 / (f_qg[2] + ddd) * vfeat_i[3]
                 + a15 * a15 / (a_i + ddd) * vfeat_i[4])
        vfgdx = a15 / (a_i + ddd) * vfeat_i[5]
        vf_qg[4:7] = 2 * vfxdx * f_xg + vfgdx * rho[1:4] / (rho[0] + ddd)
        deda = (feat_i[2] * vfeat_i[2] + feat_i[4] * vfeat_i[4]
                + feat_i[5] * vfeat_i[5])
        deda *= -1 / (a_i + ddd)
        dedrho = -1 * vfeat_i[5] * feat_i[5] / (rho[0] + ddd)
        dedgrad = vfgdx * f_xg / (rho[0] + ddd)

        vflapl = np.empty((2, rho.shape[-1]))
        dedrho -= (pow2 * vfeat_i[6] * feat_i[6] + pow1 * vfeat_i[7] * feat_i[7]) / (rho[0] + ddd)
        vflapl[0] = vfeat_i[6] / (2 * CFC * (rho[0] + ddd) ** pow2)
        vflapl[1] = vfeat_i[7] / (rho[0] + ddd) ** pow1

        feat_i[4] *= nspin
        feat_i[6] /= nspin ** pow2
        feat_i[7] /= nspin ** pow1
        return vf_qg, deda, dedrho, dedgrad, vflapl

    @staticmethod
    def fill_dfeat_from_f(
            dfeatj_ig, df_qg, rho, drho, a0, a_i, dadphi,
            feat_i, f_qg, f_xg, fxdx, fgdx, flapl_prho, dflapl_prho,
    ):
        ddd = 1e-10
        a15 = a0 ** 1.5
        dfeatj_ig[0] = a15 * df_qg[0]
        dfeatj_ig[1] = (2.0 / 3 * a15) * df_qg[1]
        df_xg = df_qg[4:7]
        dfxdx = 2 * np.einsum('qg,qg->g', f_xg, df_xg)
        dfgdx = np.einsum('qg,qg->g', f_xg, drho[1:4] / (rho[0] + ddd))
        dfgdx += np.einsum('qg,qg->g', df_xg, rho[1:4] / (rho[0] + ddd))
        dfgdx -= fgdx / (rho[0] + ddd) * drho[0]
        # feat_i[2] = a15 / (a_i + ddd) * f_qg[2]
        dfeatj_ig[2] = a15 / (a_i + ddd) * df_qg[2]
        dfeatj_ig[2] -= feat_i[2] / (a_i + ddd) * dadphi
        # feat_i[3] = a15 * fxdx / (f_qg[2] + ddd)
        dfeatj_ig[3] = a15 * dfxdx / (f_qg[2] + ddd)
        dfeatj_ig[3] -= feat_i[3] / (f_qg[2] + ddd) * df_qg[2]
        # feat_i[4] = a15 * a15 * fxdx / (a_i + ddd)
        dfeatj_ig[4] = a15 * a15 * dfxdx / (a_i + ddd)
        dfeatj_ig[4] -= feat_i[4] / (a_i + ddd) * dadphi
        # feat_i[5] = a15 * fgdx / (a_i + ddd)
        dfeatj_ig[5] = a15 * dfgdx / (a_i + ddd)
        dfeatj_ig[5] -= feat_i[5] / (a_i + ddd) * dadphi

        pow1 = (2 * FRAC_LAPL_POWER / 3.0) + 1
        pow2 = (4 * FRAC_LAPL_POWER / 3.0) + 1
        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        dfeatj_ig[6] = (dflapl_prho[1] - pow2 * flapl_prho[1] * drho[0] / rho[0]) / rho[0] ** pow2
        dfeatj_ig[7] = (dflapl_prho[2] - pow1 * flapl_prho[2] * drho[0] / rho[0]) / rho[0] ** pow1
        dfeatj_ig[6] /= 2 * CFC


class CiderConv2Spline(CiderConvSpline):
    """
    Experimental class that uses a spline for the auxiliary expansion
    in addition to projectiing the convolved features back onto grids.
    This is akin to the solid-state version.
    """

    def set_auxmols(self, mol):
        nalpha = len(self.alphas)
        self.cho_char = ctypes.c_char(b'U')
        self.alpha_norms = np.ones(nalpha)
        self.auxmol = self.get_cider_auxmol(mol)
        self._setup_atomic_cho_factors()
        self.gammas = 2 * self.alphas
        mol = self.auxmol
        self.lmax = mol._bas[:,ANG_OF].max()
        self.build_bigaux()

        self.w_iap = np.empty((nalpha, nalpha, 4))
        libcider.compute_alpha_splines(
            self.w_iap.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nalpha)
        )

        aparam = 0.03
        dparam = 0.04
        N = 200
        Rg = aparam * (np.exp(dparam * np.arange(N)) - 1)
        nrad = Rg.size
        nbas = self.bigaux.nbas
        w_rsp = np.zeros((nrad, nbas, 4), order='C')
        shls_slice = (0, self.bigaux.nbas)
        libcider.compute_spline_maps(
            w_rsp.ctypes.data_as(ctypes.c_void_p),
            Rg.ctypes.data_as(ctypes.c_void_p),
            self.bigaux._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            self.bigaux._env.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*2)(*shls_slice),
            ctypes.c_int(nrad),
        )
        self.w_rsp = w_rsp
        self.aparam = aparam
        self.dparam = dparam
        self.rspline = Rg
        nlm = get_lm_max(self.bigaux)
        self.nlm = nlm

        self.is_num_ai_setup = False

    def _transform_orbital_feat_fwd(ni, p_au):
        p_au = np.ascontiguousarray(p_au)
        ni._fill_cho_solves_(p_au)
        p_ua = np.ascontiguousarray(p_au.T)
        o_au = ni._multiply_atc_integrals(p_ua)
        return o_au

    def _transform_orbital_feat_bwd(ni, p_au):
        q_au = np.ascontiguousarray(
            ni._multiply_atc_integrals_bwd(np.ascontiguousarray(p_au)).T
        )
        ni._fill_cho_solves_(q_au)
        return q_au

    def transform_orbital_feat_fwd(self, grids, theta_ga):
        p_ua = self._call_angc_to_orb(theta_ga, self.auxmol, grids)
        #return super(CiderConvSpline, self).transform_orbital_feat_fwd(p_ua.T)
        out = self._transform_orbital_feat_fwd(p_ua.T)
        if self.onsite_direct:
            f_gq = self._call_orb_to_angc(
                np.ascontiguousarray(out.T), self.bigaux, grids
            )
            return self.project_spline_conv(out, c2s=True), f_gq
        else:
            return self.project_spline_conv(out, c2s=True)

    def transform_orbital_feat_bwd(self, grids, f_qarlp, f_gq=None):
        #p_au = super(CiderConvSpline, self).transform_orbital_feat_bwd(f_qarlp)
        out = self.project_spline_conv(f_qarlp, c2s=False)
        if self.onsite_direct:
            assert f_gq is not None
            out += self._call_angc_to_orb(f_gq, self.bigaux, grids).T
        p_au = self._transform_orbital_feat_bwd(out)
        p_ua = np.ascontiguousarray(p_au.T)
        theta_ga = self._call_orb_to_angc(p_ua, self.auxmol, grids)
        return np.ascontiguousarray(theta_ga.T).T

    def get_cider_coefs(self, rho, derivs=False, **gg_kwargs):
        alphas = self.alphas
        outputs = gg_kwargs['get_exponent'](
            rho, a0=0.5*gg_kwargs['a0'], fac_mul=0.5*gg_kwargs['fac_mul'],
            amin=0.5*gg_kwargs['amin'],
            nspin=gg_kwargs.get('nspin') or 1
        )
        cider_exp, derivs = outputs[0], outputs[1:]
        ucond = cider_exp >= self.alphas[0]
        lcond = cider_exp < self.alphas[-1]
        cider_exp[ucond] = self.alphas[0]
        cider_exp[lcond] = self.alphas[-1]
        if ucond.any() or lcond.any():
            raise ValueError('cider_exp out of bounds')
        nalpha = len(alphas)
        ngrids = rho.shape[-1]
        p_ga = np.empty((ngrids, nalpha))
        dp_ga = np.empty((ngrids, nalpha))
        libcider.VXCfill_coefs_spline_t(
            p_ga.ctypes.data_as(ctypes.c_void_p),
            dp_ga.ctypes.data_as(ctypes.c_void_p),
            cider_exp.ctypes.data_as(ctypes.c_void_p),
            self.w_iap.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(nalpha),
            ctypes.c_double(self.alphas[0]),
            ctypes.c_double(self.alpha_lambd),
        )
        dp_ga[ucond] = 0
        dp_ga[lcond] = 0
        if derivs:
            return (p_ga.T, dp_ga.T,) + derivs
        else:
            return p_ga.T, dp_ga.T


class CiderConvSplineK(CiderConv2Spline):

    def get_cider_coefs_bwd(self, rho, derivs=False, **gg_kwargs):
        alphas = self.alphas
        outputs = gg_kwargs['get_exponent'](
            rho, a0=gg_kwargs['a0'], fac_mul=gg_kwargs['fac_mul'],
            amin=gg_kwargs['amin'],
            nspin=gg_kwargs.get('nspin') or 1
        )
        cider_exp, derivs = outputs[0], outputs[1:]
        ucond = cider_exp >= self.alphas[0]
        lcond = cider_exp < self.alphas[-1]
        cider_exp[ucond] = self.alphas[0]
        cider_exp[lcond] = self.alphas[-1]
        if ucond.any() or lcond.any():
            raise ValueError('cider_exp out of bounds')
        nalpha = len(alphas)
        ngrids = rho.shape[-1]
        p_ga = np.empty((ngrids, nalpha))
        dp_ga = np.empty((ngrids, nalpha))
        libcider.VXCfill_coefs_spline_t(
            p_ga.ctypes.data_as(ctypes.c_void_p),
            dp_ga.ctypes.data_as(ctypes.c_void_p),
            cider_exp.ctypes.data_as(ctypes.c_void_p),
            self.w_iap.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(nalpha),
            ctypes.c_double(self.alphas[0]),
            ctypes.c_double(self.alpha_lambd),
        )
        dp_ga[ucond] = 0
        dp_ga[lcond] = 0
        if derivs:
            return (p_ga.T, dp_ga.T,) + derivs
        else:
            return p_ga.T, dp_ga.T

    def get_cider_coefs_fwd(self, rho, derivs=False, **gg_kwargs):
        alphas = self.alphas
        a0 = gg_kwargs['a0']
        outputs = gg_kwargs['get_exponent'](
            rho, a0=gg_kwargs['a0'] / a0, fac_mul=gg_kwargs['fac_mul'] / a0,
            amin=gg_kwargs['amin'] / a0,
            nspin=gg_kwargs.get('nspin') or 1
        )
        cider_exp, derivs = outputs[0], outputs[1:]
        nalpha = len(alphas)
        ngrids = rho.shape[-1]
        p_ga = np.empty((ngrids, nalpha))
        dp_ga = np.empty((ngrids, nalpha))
        libcider.VXCfill_coefs_kv1_t(
            p_ga.ctypes.data_as(ctypes.c_void_p),
            dp_ga.ctypes.data_as(ctypes.c_void_p),
            cider_exp.ctypes.data_as(ctypes.c_void_p),
            alphas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(nalpha),
        )
        if derivs:
            return (p_ga.T, dp_ga.T,) + outputs
        else:
            return p_ga.T, dp_ga.T

    def build_bigaux(self):
        self.ovlp_mats, self.atco = _generate_atc_integrals2(
            self.auxmol, self.alphas, self.gammas, self.lmax,
            do_cholesky=True, alpha_norms=self.alpha_norms,
            debug=False, vg=True
        )
        self.bigaux = SimpleMole(
            self.auxmol._atm, self.atco.bas,
            self.atco.env, self.atco.loc,
            self.auxmol.atom_coords()
        )
        self.auxmol_ovlp_cintopt = moleintor.make_cintopt(
            self.auxmol._atm, self.auxmol._bas,
            self.auxmol._env, 'int1e_ovlp_sph'
        )

    def transform_orbital_feat_fwd(self, grids, theta_ga):
        p_ua = self._call_angc_to_orb(theta_ga, self.auxmol, grids)
        p_au = np.ascontiguousarray(p_ua.T)
        self._fill_cho_solves_(p_au)
        p_ua = np.ascontiguousarray(p_au.T)
        out = self._multiply_atc_integrals(p_ua, vk=True)
        if self.onsite_direct:
            f_gq = self._call_orb_to_angc(
                np.ascontiguousarray(out.T), self.bigaux, grids
            )
            return self.project_spline_conv(out, c2s=True), f_gq
        else:
            return self.project_spline_conv(out, c2s=True)

    def transform_orbital_feat_bwd(self, grids, f_qarlp, f_gq=None):
        out = self.project_spline_conv(f_qarlp, c2s=False)
        if self.onsite_direct:
            assert f_gq is not None
            out += self._call_angc_to_orb(f_gq, self.bigaux, grids).T
        p_au = np.ascontiguousarray(
            self._multiply_atc_integrals_bwd(np.ascontiguousarray(out), vk=True).T
        )
        self._fill_cho_solves_(p_au)
        p_ua = np.ascontiguousarray(p_au.T)
        theta_ga = self._call_orb_to_angc(p_ua, self.auxmol, grids)
        return np.ascontiguousarray(theta_ga.T).T

    def get_feat_from_f(self, f_qg, rho, a_i, flapl_prho, gg_kwargs, nspin=1):
        nfeat = 7
        ngrid = rho.shape[-1]
        feat_i = np.zeros((nfeat, ngrid))
        dfeat_i = np.zeros((nfeat, ngrid))
        p_iqg = np.zeros((nfeat, f_qg.shape[0], ngrid))
        dadn_i = np.zeros((nfeat, ngrid))
        dadsigma_i = np.zeros((nfeat, ngrid))
        dadtau_i = np.zeros((nfeat, ngrid))
        ax = 0.5 * gg_kwargs['a0']
        for i in range(5):
            gg_kwargs_tmp = {
                'get_exponent': gg_kwargs['get_exponent'],
                'a0': 2**(i-1) * gg_kwargs['a0'],
                'fac_mul': 2**(i-1) * gg_kwargs['fac_mul'],
                'amin': 2**(i-1) * gg_kwargs['amin'],
                'nspin': nspin,
            }
            p_iqg[i], dp_qg, dadn_i[i], dadsigma_i[i], dadtau_i[i] = (
                self.get_cider_coefs_bwd(rho, derivs=True, **gg_kwargs_tmp)
            )
            a15 = ax ** 1.5
            const = a15 * np.exp(1.5 / ax)
            feat_i[i] = const * np.einsum('qg,qg->g', f_qg, p_iqg[i])
            dfeat_i[i] = const * np.einsum('qg,qg->g', f_qg, dp_qg)
            ax *= 2
        #feat_i[5] = flapl_prho[1] / (a_i + 1e-10)**2.5
        #feat_i[6] = flapl_prho[2] / (a_i + 1e-10)**2

        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        #feat_i[5] = (flapl_prho[1] - 2 * rho[5]) / (flapl_prho[1] + 2 * rho[5] + 1e-10)
        #sigma = np.einsum('qg,qg->g', rho[1:4], rho[1:4])
        #feat_i[6] = (flapl_prho[2] * flapl_prho[2] - sigma) / (flapl_prho[2] * flapl_prho[2] + sigma + 1e-10)

        pow1 = (2 * FRAC_LAPL_POWER / 3.0) + 1
        pow2 = (4 * FRAC_LAPL_POWER / 3.0) + 1
        feat_i[5] = flapl_prho[1] / (2 * CFC * rho[0]**pow2)
        feat_i[6] = flapl_prho[2] / rho[0]**pow1

        return feat_i, dfeat_i, p_iqg, dadn_i, dadsigma_i, dadtau_i

    @staticmethod
    def get_vfeat_scf(vfeat_i, dfeat_i, p_iqg, gg_kwargs, nspin=1):
        vf_qg = np.zeros((p_iqg.shape[1], p_iqg.shape[2]))
        ax = 0.5 * gg_kwargs['a0']
        deda = 0
        for i in range(5):
            a15 = ax ** 1.5
            const = a15 * np.exp(1.5 / ax)
            deda += vfeat_i[i] * dfeat_i[i] * 2**(i-1)
            vf_qg[:] += const * vfeat_i[i] * p_iqg[i]
            ax *= 2
        return vf_qg, deda

    @staticmethod
    def fill_dfeat_from_f(
            dfeatj_ig, dfeat_i, a_i, dadphi, p_iqg, df_qg, rho, drho, flapl_prho,
            dflapl_prho, gg_kwargs
    ):
        ax = 0.5 * gg_kwargs['a0']
        for i in range(5):
            a15 = ax ** 1.5
            const = a15 * np.exp(1.5 / ax)
            # TODO
            # This 2**(i-1) as well as the normalizations assume
            # a certain structure to the exponents (each 2x greater
            # than the previous). Should fix this assumption.
            # TODO
            # Need to test non-default settings for derivatives.
            dfeatj_ig[i] = dadphi * dfeat_i[i] * 2**(i-1)
            dfeatj_ig[i] += const * np.einsum('qg,qg->g', df_qg, p_iqg[i])
            ax *= 2

        #dfeatj_ig[5] = (dflapl_prho[1] - 2 * drho[4]) / (flapl_prho[1] + 2 * rho[5] + 1e-10)
        #dfeatj_ig[5] -= (flapl_prho[1] - 2 * rho[5]) / (flapl_prho[1] + 2 * rho[5] + 1e-10)**2 * (dflapl_prho[1] + 2 * drho[4])
        #sigma = np.einsum('qg,qg->g', rho[1:4], rho[1:4])
        #dsigma = 2 * np.einsum('qg,qg->g', rho[1:4], drho[1:4])
        #dfeatj_ig[6] = (2 * dflapl_prho[2] * flapl_prho[2] - dsigma) / (flapl_prho[2] * flapl_prho[2] + sigma + 1e-10)
        #dfeatj_ig[6] -= (flapl_prho[2] * flapl_prho[2] - sigma) / (flapl_prho[2] * flapl_prho[2] + sigma + 1e-10)**2 * (2 * dflapl_prho[2] * flapl_prho[2] + dsigma)

        #dfeatj_ig[5] = dflapl_prho[1] / (a_i + 1e-10)**2.5
        #dfeatj_ig[5] -= 2.5 * flapl_prho[1] / (a_i + 1e-10)**3.5 * dadphi
        #dfeatj_ig[6] = dflapl_prho[2] / (a_i + 1e-10)**2
        #dfeatj_ig[6] -= 2 * flapl_prho[2] / (a_i + 1e-10)**3 * dadphi

        pow1 = (2 * FRAC_LAPL_POWER / 3.0) + 1
        pow2 = (4 * FRAC_LAPL_POWER / 3.0) + 1
        CFC = (3.0 / 10) * (3 * np.pi ** 2) ** (2.0 / 3)
        dfeatj_ig[5] = (dflapl_prho[1] - pow2 * flapl_prho[1] * drho[0] / rho[0]) / rho[0]**pow2
        dfeatj_ig[6] = (dflapl_prho[2] - pow1 * flapl_prho[2] * drho[0] / rho[0]) / rho[0]**pow1
        dfeatj_ig[5] /= 2 * CFC

if __name__ == '__main__':
    Lmax = 121
    t = np.random.normal(size=(3,80)) + 1e-8
    t /= np.linalg.norm(t, axis=0)
    from ciderpress.dft.futil import fast_sph_harm as fsh
    Y_true = fsh.recursive_sph_harm_t2_pyscf(Lmax, t)
    tmpx = Y_true[3].copy()
    tmpy = Y_true[1].copy()
    tmpz = Y_true[2].copy()
    Y_true[1] = tmpx
    Y_true[2] = tmpy
    Y_true[3] = tmpz
    Y_test = np.zeros((t[0].size, Lmax))
    print(t.shape, t[0].size)
    print(type(t.T), type(Y_test))
    libcider.recursive_sph_harm_vec(
        ctypes.c_int(Lmax),
        ctypes.c_int(t[0].size),
        np.ascontiguousarray(t.T).ctypes.data_as(ctypes.c_void_p),
        Y_test.ctypes.data_as(ctypes.c_void_p),
    )
    for L in range(Lmax):
        print('C VERSION', L, np.linalg.norm(Y_true[L]-Y_test[:,L]))
