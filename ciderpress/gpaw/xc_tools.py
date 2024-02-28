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
from ase.units import Ha

from gpaw.xc import XC
from gpaw.xc.kernel import XCNull
from gpaw.utilities import unpack
from gpaw import GPAW
from gpaw.calculator import GPAW as GPAWOld
from gpaw.new.ase_interface import ASECalculator
from ciderpress.gpaw.cider_paw import _CiderPASDW_MPRoutines

from pathlib import Path


def non_self_consistent_eigenvalues(
        calc, xc, n1=0, n2=0, kpt_indices=None,
        snapshot=None):
    if not isinstance(calc, (GPAWOld, ASECalculator)):
        if calc == '<gpw-file>':  # for doctest
            return (np.zeros((1, 1, 1)),
                    np.zeros((1, 1, 1)),
                    np.zeros((1, 1, 1)))
        calc = GPAW(Path(calc), txt=None, parallel={'band': 1, 'kpt': 1})

    assert isinstance(calc, (GPAWOld, ASECalculator))
    wfs = calc.wfs

    if isinstance(xc, str):
        xc = XC(xc)

    if n2 <= 0:
        n2 += wfs.bd.nbands

    if kpt_indices is None:
        kpt_indices = np.arange(wfs.kd.nibzkpts).tolist()

    if snapshot is not None:
        raise NotImplementedError
    #path = Path(snapshot) if snapshot is not None else None

    e_dft_sin = np.zeros(0)
    v_dft_sin = np.zeros(0)

    # sl=semilocal, nl=nonlocal
    v_hyb_sl_sin = np.zeros(0)

    if v_dft_sin.size == 0:
        e_dft_sin, v_dft_sin, v_hyb_sl_sin = _semi_local(
            calc, xc, n1, n2, kpt_indices)

    return (e_dft_sin * Ha,
            v_dft_sin * Ha,
            v_hyb_sl_sin * Ha)


def _semi_local(calc, xc, n1, n2, kpt_indices):
    wfs = calc.wfs
    nspins = wfs.nspins
    e_dft_sin = np.array([[calc.get_eigenvalues(k, spin)[n1:n2]
                           for k in kpt_indices]
                          for spin in range(nspins)])
    v_dft_sin = vxc(calc.gs_adapter(), n1=n1, n2=n2)[:, kpt_indices]
    if isinstance(xc.kernel, XCNull):
        v_hyb_sl_sin = np.zeros_like(v_dft_sin)
    else:
        v_hyb_sl_sin = vxc(calc.gs_adapter(), xc, n1=n1, n2=n2)[:, kpt_indices]
    return e_dft_sin / Ha, v_dft_sin / Ha, v_hyb_sl_sin / Ha


def vxc(gs, xc=None, coredensity=True, n1=0, n2=0):
    """Calculate XC-contribution to eigenvalues."""

    if n2 <= 0:
        n2 += gs.bd.nbands

    ham = gs.hamiltonian
    dens = gs.density

    if xc is None:
        xc = ham.xc
    elif isinstance(xc, str):
        xc = XC(xc)

    if dens.nt_sg is None:
        dens.interpolate_pseudo_density()

    thisisatest = not True

    #if xc.orbital_dependent:
    xc.set_grid_descriptor(gs._density.finegd)
    xc.initialize(gs._density, gs._hamiltonian, gs._wfs)
    xc.set_positions(gs.spos_ac)
    gs._hamiltonian.get_xc_difference(xc, gs._density)
    #gs.get_xc_difference(xc)

    # Calculate XC-potential:
    vxct_sg = ham.finegd.zeros(gs.nspins)
    xc.calculate(dens.finegd, dens.nt_sg, vxct_sg)
    vxct_sG = ham.restrict_and_collect(vxct_sg)
    if thisisatest:
        vxct_sG[:] = 1

    # ... and PAW corrections:
    dvxc_asii = {}
    if isinstance(xc, _CiderPASDW_MPRoutines):
        xc._collect_paw_corrections()
    for a, D_sp in dens.D_asp.items():
        dvxc_sp = np.zeros_like(D_sp)
        xc.calculate_paw_correction(gs.setups[a], D_sp, dvxc_sp, a=a,
                                    addcoredensity=coredensity)
        dvxc_asii[a] = [unpack(dvxc_p) for dvxc_p in dvxc_sp]
        if thisisatest:
            dvxc_asii[a] = [gs.setups[a].dO_ii]

    vxc_un = np.empty((gs.kd.mynk * gs.nspins, gs.bd.mynbands))
    for u, vxc_n in enumerate(vxc_un):
        kpt = gs.kpt_u[u]
        vxct_G = vxct_sG[kpt.s]
        if xc.type == 'MGGA':
            G_Gv = gs._wfs.pd.get_reciprocal_vectors(q=kpt.q)
        for n in range(gs.bd.mynbands):
            if n1 <= n + gs.bd.beg < n2:
                psit_G = gs.get_wave_function_array(u, n)
                vxc_n[n] = gs.gd.integrate((psit_G * psit_G.conj()).real,
                                           vxct_G, global_integral=False)
                if xc.type == 'MGGA':
                    psit_k = kpt.psit_nG[n]
                    dpsit2_G = np.zeros_like(vxct_G)
                    for v in range(3):
                        tmp_G = gs.pd.ifft(
                            1j * G_Gv[:, v] * psit_k, q=kpt.q
                        )
                        dpsit2_G += 0.5 * (tmp_G * tmp_G.conj()).real
                    vxc_n[n] += gs.gd.integrate(dpsit2_G, xc.dedtaut_sG[kpt.s],
                                                global_integral=False)

        for a, dvxc_sii in dvxc_asii.items():
            m1 = max(n1, gs.bd.beg) - gs.bd.beg
            m2 = min(n2, gs.bd.end) - gs.bd.beg
            if m1 < m2:
                P_ni = kpt.P_ani[a][m1:m2]
                vxc_n[m1:m2] += ((P_ni @ dvxc_sii[kpt.s]) *
                                 P_ni.conj()).sum(1).real

    gs.gd.comm.sum(vxc_un)
    vxc_skn = gs.kd.collect(vxc_un, broadcast=True)

    vxc_skn = gs.bd.collect(vxc_skn.T.copy(), broadcast=True).T
    return vxc_skn[:, :, n1:n2] * Ha