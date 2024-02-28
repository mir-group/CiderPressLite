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

from pyscf.dft.gen_grid import *
from pyscf.dft.gen_grid import _padding_size, _default_rad, _default_ang
from ciderpress.dft.cider_conv import libcider, CIDER_DEFAULT_LMAX
import numpy as np

LMAX_DICT = {v : k // 2 for k, v in LEBEDEV_ORDER.items()}

def gen_atomic_grids_cider(mol, atom_grid={}, radi_method=radi.gauss_chebyshev,
                           level=3, prune=nwchem_prune, full_lmax=CIDER_DEFAULT_LMAX, **kwargs):
    if isinstance(atom_grid, (list, tuple)):
        atom_Grid = dict([(mol.atom_symbol(ia), atom_grid)
                          for ia in range(mol.natm)])
    atom_grids_tab = {}
    lmax_tab = {}
    rad_loc_tab = {}
    ylm_tab = {}
    ylm_loc_tab = {}
    rad_tab = {}
    dr_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
                if n_ang not in LEBEDEV_NGRID:
                    raise ValueError('Unsupported angular grids %d' % n_ang)
            else:
                n_rad = _default_rad(chg, level)
                n_ang = _default_ang(chg, level)
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)

            rad_weight = 4*numpy.pi * rad**2 * dr

            if callable(prune):
                angs = prune(chg, rad, n_ang)
            else:
                angs = [n_ang] * n_rad
            logger.debug(mol, 'atom %s rad-grids = %d, ang-grids = %s',
                         symb, n_rad, angs)

            angs = numpy.array(angs)
            coords = []
            vol = []
            rad_loc = np.append([0], np.cumsum(angs)).astype(np.int32)
            rad_loc = np.array([0], dtype=np.int32)
            lmaxs = np.array([LMAX_DICT[ang] for ang in angs]).astype(np.int32)
            lmaxs = np.minimum(lmaxs, full_lmax)
            nlm = (full_lmax+1)*(full_lmax+1)
            ylm_full = np.empty((0,nlm), order='C')
            ylm_loc = []
            rads = []
            drs = []
            for n in sorted(set(angs)):
                grid = numpy.empty((n,4))
                libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(n))
                idx = numpy.where(angs==n)[0]
                yloc_curr = ylm_full.shape[0]
                ylm = np.zeros((n, nlm), order='C')
                sphgd = np.ascontiguousarray(grid[:,:3])
                libcider.recursive_sph_harm_vec(
                    ctypes.c_int(nlm),
                    ctypes.c_int(n),
                    sphgd.ctypes.data_as(ctypes.c_void_p),
                    ylm.ctypes.data_as(ctypes.c_void_p),
                )
                lmax_shl = LMAX_DICT[n]
                nlm_shl = (lmax_shl+1)*(lmax_shl+1)
                ylm[:,nlm_shl:] = 0.0
                ylm_full = np.append(ylm_full, ylm, axis=0)
                coords.append(numpy.einsum('i,jk->ijk',rad[idx],
                                           grid[:,:3]).reshape(-1,3))
                vol.append(numpy.einsum('i,j->ij', rad_weight[idx],
                                        grid[:,3]).ravel())
                rads.append(rad[idx])
                drs.append(dr[idx])
                rad_loc = np.append(rad_loc, rad_loc[-1] + n*np.arange(1,len(idx)+1))
                ylm_loc.append(yloc_curr*np.ones(idx.size, dtype=np.int32))
            atom_grids_tab[symb] = (numpy.vstack(coords), numpy.hstack(vol))
            lmax_tab[symb] = lmaxs
            rad_loc_tab[symb] = rad_loc
            ylm_tab[symb] = ylm_full
            ylm_loc_tab[symb] = np.concatenate(ylm_loc).astype(np.int32)
            rad_tab[symb] = np.concatenate(rads).astype(np.float64)
            dr_tab[symb] = np.concatenate(drs).astype(np.float64)

    return atom_grids_tab, lmax_tab, rad_loc_tab, ylm_tab, ylm_loc_tab, rad_tab, dr_tab


class CiderGrids(Grids):

    def __init__(self, mol, lmax=CIDER_DEFAULT_LMAX):
        super(CiderGrids, self).__init__(mol)
        #self.becke_scheme = becke_lko
        self.lmax = lmax

    def gen_atomic_grids(self, mol, atom_grid=None, radi_method=None,
                         level=None, prune=None, **kwargs):
        if atom_grid is None: atom_grid = self.atom_grid
        if radi_method is None: radi_method = self.radi_method
        if level is None: level = self.level
        if prune is None: prune = self.prune
        atom_grids_tab, lmax_tab, rad_loc_tab, ylm_tab, \
            ylm_loc_tab, rad_tab, dr_tab = \
                gen_atomic_grids_cider(
                    mol, atom_grid, self.radi_method,
                    level, prune, **kwargs
                )
        self.atom_grids_tab = atom_grids_tab
        self.lmax_tab = lmax_tab
        self.rad_loc_tab = rad_loc_tab
        self.ylm_tab = ylm_tab
        self.ylm_loc_tab = ylm_loc_tab
        self.rad_tab = rad_tab
        self.dr_tab = dr_tab
        return self.atom_grids_tab

    def _reduce_angc_ylm(self, theta_rlmq, theta_gq, a2y=True):
        assert theta_rlmq.flags.c_contiguous
        assert theta_gq.flags.c_contiguous
        if a2y:
            fn = libcider.reduce_angc_to_ylm
        else:
            fn = libcider.reduce_ylm_to_angc
        fn(
            theta_rlmq.ctypes.data_as(ctypes.c_void_p),
            self.ylm.ctypes.data_as(ctypes.c_void_p),
            theta_gq.ctypes.data_as(ctypes.c_void_p),
            self.rad_loc.ctypes.data_as(ctypes.c_void_p),
            self.ylm_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(theta_gq.shape[-1]),
            ctypes.c_int(theta_rlmq.shape[0]),
            ctypes.c_int(theta_gq.shape[0]),
            ctypes.c_int(theta_rlmq.shape[1]),
        )

    def build(self, mol=None, with_non0tab=False, **kwargs):
        if with_non0tab:
            with_non0tab = False
            import warnings
            msg = 'non0tab grids screening not yet supported for CIDER.\n'
            msg+= 'Setting non0tab to False.'
            warnings.warn(msg)
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.nlm = (self.lmax+1)*(self.lmax+1)
        atom_grids_tab = self.gen_atomic_grids(mol, self.atom_grid,
                                               self.radi_method,
                                               self.level, self.prune,
                                               full_lmax=self.lmax, **kwargs)
        # TODO cleaner version of this way of calling VXCgen_grid_lko
        #tmp = libdft.VXCgen_grid
        #libdft.VXCgen_grid = libcider.VXCgen_grid_lko
        self.coords, self.weights = \
                self.get_partition(mol, atom_grids_tab,
                                   self.radii_adjust, self.atomic_radii,
                                   self.becke_scheme)
        #libdft.VXCgen_grid = tmp
        full_rad_loc = np.array([0], dtype=np.int32)
        ar_loc = []
        ra_loc = [0]
        rads = []
        full_ylm_loc = np.array([], dtype=np.int32)
        full_ylm = np.empty((0, self.nlm), dtype=np.float64)
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            nrad = self.rad_loc_tab[symb].size - 1
            start = full_rad_loc[-1]
            full_rad_loc = np.append(full_rad_loc, self.rad_loc_tab[symb][1:] + start)
            full_ylm_loc = np.append(full_ylm_loc, self.ylm_loc_tab[symb] + full_ylm.shape[0])
            full_ylm = np.append(full_ylm, self.ylm_tab[symb], axis=0)
            rads.append(self.rad_tab[symb])
            rad_loc = full_rad_loc.size-1
            ra_loc.append(rad_loc)
            ar_loc.append(ia*np.ones(nrad, dtype=np.int32))
        self.rad_loc = np.ascontiguousarray(full_rad_loc.astype(np.int32))
        self.ylm = np.ascontiguousarray(full_ylm.astype(np.float64))
        self.ylm_loc = np.ascontiguousarray(full_ylm_loc.astype(np.int32))
        self.ar_loc = np.ascontiguousarray(
            np.concatenate(ar_loc).astype(np.int32)
        )
        self.rad_arr = np.ascontiguousarray(
            np.concatenate(rads).astype(np.float64)
        )
        self.ra_loc = np.array(ra_loc, dtype=np.int32)
        assert self.rad_loc[-1] == self.weights.size
        """
        print("GRID DATA")
        print(self.rad_loc)
        print(self.ylm_loc)
        print(self.ylm_loc_tab)
        print(self.ar_loc.tolist())
        print(self.ra_loc)
        """

        if False: # TODO implement grid sorting if sort_grids=True
            idx = arg_group_grids(mol, self.coords)
            self.coords = self.coords[idx]
            self.weights = self.weights[idx]

        if self.alignment > 1:
            padding = _padding_size(self.size, self.alignment)
            logger.debug(self, 'Padding %d grids', padding)
            if padding > 0:
                self.coords = numpy.vstack(
                    [self.coords, numpy.repeat([[1e4]*3], padding, axis=0)])
                self.weights = numpy.hstack([self.weights, numpy.zeros(padding)])

        if with_non0tab:
            self.non0tab = self.make_mask(mol, self.coords)
            self.screen_index = self.non0tab
        else:
            self.screen_index = self.non0tab = None
        logger.info(self, 'tot grids = %d', len(self.weights))
        return self


class CiderRadGrids():

    def __init__(self, a=0.03, N=250, d=0.03):
        self.a = a
        self.d = d
        self.r = a * (np.exp(d*np.arange(N)) - 1)

    def get_i_and_dr(r):
        i = (np.log(r / a + 1) / d).astype(np.int32)
        dr = r - i
        return i, dr
