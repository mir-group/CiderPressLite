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

import numpy
import numpy as np
from pyscf import gto
import logging
import ctypes
from ciderpress.lib import load_library as load_cider_library
from ciderpress.dft.cider_kernel import get_exponent_d, get_exponent_b
from pyscf.dft.numint import NumInt
from pyscf.dft.gen_grid import Grids


LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)
GG_SMUL = 1.0
GG_AMUL = 1.0
GG_AMIN = 1.0 / 18
CFC = (3.0/10) * (3*np.pi**2)**(2.0/3)
DESC_VERSION_LIST = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

UEG_VECTORS = {
    'b': np.array([1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0]),
    'd': np.array([1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0]),
}


def get_exchange_descriptors(analyzer, restricted=True, version='a', **kwargs):
    """
    A length-21 descriptor containing semi-local information
    and a few Gaussian integrals. The descriptors are
    normalized to be scale-invariant and rotation-invariant.

    Args:
        analyzer (RHFAnalyzer or UHFAnalyzer): The analyzer containing
            the density matrix from which the features are computed

    g1 order: x, y, z
    g2 order: xy, yz, z^2, xz, x^2-y^2
    """
    assert version in DESC_VERSION_LIST
    if version == 'a':
        _get_x_helper = _get_x_helper_a
    elif version not in ['a', 'c']:
        if version == 'b':
            get_exponent = get_exponent_b
            feat_helper = _bfeat_helper
        elif version == 'd':
            get_exponent = get_exponent_d
            feat_helper = _bfeat_helper
        elif version == 'f':
            get_exponent = get_exponent_b
            feat_helper = _bfeat_helper
        elif version == 'g':
            get_exponent = get_exponent_b
            feat_helper = _gfeat_helper
        elif version == 'h':
            get_exponent = get_exponent_b
            feat_helper = _hfeat_helper
        elif version == 'i':
            get_exponent = get_exponent_b
            feat_helper = _ifeat_helper
        elif version == 'j':
            get_exponent = get_exponent_b
            feat_helper = _jfeat_helper
        elif version == 'k':
            get_exponent = get_exponent_b
            feat_helper = _kfeat_helper
        elif version == 'e':
            get_exponent = get_exponent_d
            feat_helper = _efeat_helper
        else:
            raise ValueError('unknown descriptor version {}'.format(version))
        if restricted:
            return _get_x_helper_b(analyzer.mol, analyzer.grids, analyzer.rdm1,
                                   get_exponent=get_exponent, feat_helper=feat_helper,
                                   version=version, **kwargs)
        else:
            desc0 = _get_x_helper_b(analyzer.mol, analyzer.grids, 2 * analyzer.rdm1[0],
                                    get_exponent=get_exponent, feat_helper=feat_helper,
                                    version=version, **kwargs)
            desc1 = _get_x_helper_b(analyzer.mol, analyzer.grids, 2 * analyzer.rdm1[1],
                                    get_exponent=get_exponent, feat_helper=feat_helper,
                                    version=version, **kwargs)
            return desc0, desc1
    elif version == 'c':
        _get_x_helper = _get_x_helper_c
    else:
        raise ValueError('unknown descriptor version {}'.format(version))

    from ciderpress.external.df_rho import get_df_rho_from_mol_and_auxbasis
    density, auxmol = get_df_rho_from_mol_and_auxbasis(
        analyzer.mol, 'weigend+etb', analyzer.rdm1
    )

    rho_data = analyzer.get_rho_data()

    if restricted:
        return _get_x_helper(auxmol, rho_data, analyzer.grids,
                             density[0], **kwargs)
    else:
        desc0 = _get_x_helper(auxmol, 2 * rho_data[0], analyzer.grids,
                              2 * density[0], **kwargs)
        desc1 = _get_x_helper(auxmol, 2 * rho_data[1], analyzer.grids,
                              2 * density[1], **kwargs)
        return desc0, desc1


def ldax(n):
    """
    LDA exchange energy density
    Args:
        n: Density
    """
    return LDA_FACTOR * n**(4.0/3)


def ldaxp(n):
    """
    Fully spin-polarized LDA exchange energy density
    Args:
        n: Density
    """
    return 0.5 * ldax(2 * n)


def lsda(nu, nd):
    """
    LSDA exchange energy density
    Args:
        nu: Spin-up Density
        nd: Spin-down Density
    """
    return ldaxp(nu) + ldaxp(nd)


def get_ldax_dens(n):
    """
    LDA exchange energy density
    Args:
        n: Density
    """
    return LDA_FACTOR * n**(4.0/3)


def get_ldax(n):
    """
    LDA Exchange energy per particle
    Args:
        n: Density
    """
    return LDA_FACTOR * n**(1.0/3)


def get_xed_from_y(y, rho):
    """
    Get the exchange energy density (n * epsilon_x)
    from the exchange enhancement factor y
    and density rho.
    """
    return rho * get_x(y, rho)


def get_x(y, rho):
    return (y + 1) * get_ldax(rho)


def get_y_from_xed(xed, rho):
    """
    Get the exchange enhancement factor minus one.
    """
    return xed / (get_ldax_dens(rho) - 1e-12) - 1


def get_gradient_magnitude(rho_data):
    return np.linalg.norm(rho_data[1:4,:], axis=0)


def get_normalized_grad(rho, mag_grad):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    s = mag_grad / (sprefac * n43 + 1e-16)
    return s


def get_single_orbital_tau(rho, mag_grad):
    return mag_grad**2 / (8 * rho + 1e-16)


def get_uniform_tau(rho):
    return (3.0/10) * (3*np.pi**2)**(2.0/3) * rho**(5.0/3)


def get_regularized_tau(tau, tau_w, tau_unif):
    alpha = (tau - tau_w) / (tau_unif + 1e-4)
    return alpha**3 / (alpha**2 + 1e-3)


def get_normalized_tau(tau, tau_w, tau_unif):
    return (tau - tau_w) / (tau_unif + 1e-16)


def get_dft_input(rho_data):
    rho = rho_data[0,:]
    mag_grad = get_gradient_magnitude(rho_data)
    s = get_normalized_grad(rho, mag_grad)
    tau_w = get_single_orbital_tau(rho, mag_grad)
    tau_unif = get_uniform_tau(rho)
    alpha = get_normalized_tau(rho_data[5], tau_w, tau_unif)
    return rho, s, alpha, tau_w, tau_unif


def get_dft_input_gga(rho_data):
    rho = rho_data[0,:]
    mag_grad = get_gradient_magnitude(rho_data)
    s = get_normalized_grad(rho, mag_grad)
    return rho, s

