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

"""
This is a small file with some baseline functionals and other
utilities. The file name is for legacy purposes and will be
changed later.
"""
from ciderpress.density import get_ldax_dens, get_ldax, \
                               get_xed_from_y, get_y_from_xed
import numpy as np


SCALE_FAC = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)

def xed_to_y_scan(xed, rho_data):
    pbex = eval_xc('SCAN,', rho_data)[0] * rho_data[0]
    return (xed - pbex) / (ldax(rho_data[0]) - 1e-7)

def y_to_xed_scan(y, rho_data):
    yp = y * ldax(rho_data[0])
    pbex = eval_xc('SCAN,', rho_data)[0] * rho_data[0]
    return yp + pbex

def xed_to_y_lda(xed, rho, s2=None):
    return get_y_from_xed(xed, rho)

def y_to_xed_lda(y, rho, s2=None):
    return get_xed_from_y(y, rho)

def chachiyo_fx(s2):
    c = 4 * np.pi / 9
    x = c * np.sqrt(s2)
    dx = c / (2 * np.sqrt(s2))
    Pi = np.pi
    Log = np.log
    chfx = (3*x**2 + Pi**2*Log(1 + x))/((Pi**2 + 3*x)*Log(1 + x))
    dchfx = (-3*x**2*(Pi**2 + 3*x) + 3*x*(1 + x)*(2*Pi**2 + 3*x)*Log(1 + x) - 3*Pi**2*(1 + x)*Log(1 + x)**2)/((1 + x)*(Pi**2 + 3*x)**2*Log(1 + x)**2)
    dchfx *= dx
    chfx[s2<1e-8] = 1 + 8 * s2[s2<1e-8] / 27
    dchfx[s2<1e-8] = 8.0 / 27
    return chfx, dchfx


def xed_to_y_chachiyo(xed, rho, s2):
    return xed / get_ldax_dens(rho) - chachiyo_fx(s2)[0]

def y_to_xed_chachiyo(y, rho, s2):
    return (y + chachiyo_fx(s2)[0]) * get_ldax_dens(rho)

def pbe_fx(s2):
    kappa = 0.804
    # mu = 0.2195149727645171
    mu = 0.21951
    mk = mu / kappa
    fac = 1.0 / (1 + mk * s2)
    fx = 1 + kappa - kappa * fac
    dfx = mu * fac * fac
    return fx, dfx

def xed_to_y_pbe(xed, rho, s2):
    return xed / get_ldax_dens(rho) - pbe_fx(s2)[0]

def y_to_xed_pbe(y, rho, s2):
    return (y + pbe_fx(s2)[0]) * get_ldax_dens(rho)

# For ex/elec training
def xed_to_dex_chachiyo(xed, rho, s2):
    return xed / rho - chachiyo_fx(s2)[0] * get_ldax(rho)

def dex_to_xed_chachiyo(dex, rho, s2):
    return dex * rho + get_ldax_dens(rho) * chachiyo_fx(s2)[0]

def get_unity():
    return 1

def get_identity(x):
    return x

XED_Y_CONVERTERS = {
    # method_name: (xed_to_y, y_to_xed, fx_baseline, nfeat--rho, s2, alpha...)
    'LDA': (xed_to_y_lda, y_to_xed_lda, get_unity, 1),
    'PBE': (xed_to_y_pbe, y_to_xed_pbe, pbe_fx, 2),
    'CHACHIYO': (xed_to_y_chachiyo, y_to_xed_chachiyo, chachiyo_fx, 2),
    'CHACHIYO_EX': (xed_to_dex_chachiyo, dex_to_xed_chachiyo, chachiyo_fx, 2),
}

