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

from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import yaml


class FeatureNormalizer(ABC):

    @abstractproperty
    def num_arg(self):
        pass

    @abstractmethod
    def bounds(self):
        pass

    @abstractmethod
    def fill_feat_(self, y, x):
        pass

    @abstractmethod
    def fill_deriv_(self, dfdx, dfdy, x):
        pass

    @classmethod
    def from_dict(cls, d):
        if d['code'] == 'L':
            return LMap.from_dict(d)
        elif d['code'] == 'T':
            return TMap.from_dict(d)
        elif d['code'] == 'U':
            return UMap.from_dict(d)
        elif d['code'] == 'V':
            return VMap.from_dict(d)
        elif d['code'] == 'V2':
            return V2Map.from_dict(d)
        elif d['code'] == 'V3':
            return V3Map.from_dict(d)
        elif d['code'] == 'V4':
            return V4Map.from_dict(d)
        elif d['code'] == 'W':
            return WMap.from_dict(d)
        elif d['code'] == 'X':
            return XMap.from_dict(d)
        elif d['code'] == 'Y':
            return YMap.from_dict(d)
        elif d['code'] == 'Z':
            return ZMap.from_dict(d)
        elif d['code'] == 'SU':
            return SignedUMap.from_dict(d)
        else:
            raise ValueError('Unrecognized code')


class LMap(FeatureNormalizer):

    def __init__(self, n, i, bounds=None):
        self.n = n
        self.i = i
        self._bounds = bounds or (-np.inf, np.inf)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = x[i]

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, = self.n, self.i
        dfdx[i] += dfdy[n]

    def as_dict(self):
        return {
            'code': 'L',
            'n': self.n,
            'i': self.i,
            'bounds': self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return LMap(d['n'], d['i'], bounds=d.get('bounds'))


class UMap(FeatureNormalizer):

    def __init__(self, n, i, gamma, bounds=None):
        self.n = n
        self.i = i
        self.gamma = gamma
        self._bounds = bounds or (0, 1)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = self.gamma * x[i] / (1 + self.gamma * x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i = self.n, self.i
        dfdx[i] += dfdy[n] * self.gamma / (1 + self.gamma * x[i])**2

    def as_dict(self):
        return {
            'code': 'U',
            'n': self.n,
            'i': self.i,
            'gamma': self.gamma,
            'bounds': self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return UMap(d['n'], d['i'], d['gamma'], bounds=d.get('bounds'))


class TMap(FeatureNormalizer):

    def __init__(self, n, i, j, bounds=None):
        self.n = n
        self.i = i
        self.j = j
        self._bounds = bounds or (0, 1)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i, j = self.n, self.i, self.j
        tmp = x[j] + 5.0 / 3 * x[i]
        y[n] = tmp / (1 + tmp)

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j = self.n, self.i, self.j
        tmp = x[j] + 5.0 / 3 * x[i]
        dtmp = 1.0 / ((1 + tmp) * (1 + tmp))
        dtmp *= dfdy[n]
        dfdx[i] += 5.0 / 3 * dtmp
        dfdx[j] += dtmp

    def as_dict(self):
        return {
            'code': 'T',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'bounds': self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['n'], d['i'], d['j'], bounds=d.get('bounds'))


def get_vmap_heg_value(heg, gamma):
    return (heg * gamma) / (1 + heg * gamma)


class VMap(FeatureNormalizer):

    def __init__(self, n, i, gamma, scale=1.0, center=0.0, bounds=None):
        self.n = n
        self.i = i
        self.gamma = gamma
        self.scale = scale
        self.center = center
        self._bounds = bounds or (-self.center, self.scale-self.center)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = -self.center + self.scale * self.gamma * x[i] / (1 + self.gamma * x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i = self.n, self.i
        dfdx[i] += dfdy[n] * self.scale * self.gamma / (1 + self.gamma * x[i])**2

    def as_dict(self):
        return {
            'code': 'V',
            'n': self.n,
            'i': self.i,
            'gamma': self.gamma,
            'scale': self.scale,
            'center': self.center,
            'bounds': self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return VMap(d['n'], d['i'], d['gamma'], d['scale'], d['center'],
                    bounds=d.get('bounds'))


class V2Map(FeatureNormalizer):

    def __init__(self, n, i, j):
        self.n = n
        self.i = i
        self.j = j

    @property
    def bounds(self):
        return (-1.0, 1.0)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i, j = self.n, self.i, self.j
        a = 2**1.5
        b = 0.5 / (a-1)
        tmp = b * (a * x[i] - x[j])
        y[n] = tmp / (1 + tmp)
        tmp = 0.5 * x[j]
        y[n] -= tmp / (1 + tmp)

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j = self.n, self.i, self.j
        a = 2**1.5
        b = 0.5 / (a-1)
        tmp = (a * x[i] - x[j])
        tmp = b / (1 + b * tmp)**2
        dfdx[i] += dfdy[n] * tmp * a
        dfdx[j] -= dfdy[n] * tmp
        tmp = 0.5 / (1 + 0.5 * x[j])**2
        dfdx[j] -= dfdy[n] * tmp

    def as_dict(self):
        return {
            'code': 'V2',
            'n': self.n,
            'i': self.i,
            'j': self.j
        }

    @classmethod
    def from_dict(cls, d):
        return V2Map(d['n'], d['i'], d['j'])


class V3Map(FeatureNormalizer):

    def __init__(self, n, i, j, gamma):
        self.n = n
        self.i = i
        self.j = j
        self.gamma = gamma

    @property
    def bounds(self):
        return (-1.0, 1.0)

    @property
    def num_arg(self):
        return 2

    def fill_feat_(self, y, x):
        n, i, j = self.n, self.i, self.j
        y[n] = self.gamma * x[i] / (1 + self.gamma * x[i])
        y[n] -= self.gamma * x[j] / (1 + self.gamma * x[j])

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j = self.n, self.i, self.j
        dfdx[i] += dfdy[n] * self.gamma / (1 + self.gamma * x[i])**2
        dfdx[j] -= dfdy[n] * self.gamma / (1 + self.gamma * x[j])**2

    def as_dict(self):
        return {
            'code': 'V3',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'gamma': self.gamma,
        }

    @classmethod
    def from_dict(cls, d):
        return V3Map(d['n'], d['i'], d['j'], d['gamma'])


class V4Map(FeatureNormalizer):

    def __init__(self, n, i, j, gamma):
        self.n = n
        self.i = i
        self.j = j
        self.gamma = gamma

    @property
    def bounds(self):
        return (-0.5, 0.5)

    @property
    def num_arg(self):
        return 2

    def fill_feat_(self, y, x):
        n, i, j = self.n, self.i, self.j
        tmp = np.exp(self.gamma * (x[i]-x[j]))
        y[n] = 1 / (1 + tmp) - 0.5

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j = self.n, self.i, self.j
        tmp = np.exp(self.gamma * (x[i]-x[j]))
        tmp = dfdy[n] * self.gamma * tmp / (1 + tmp)**2
        dfdx[i] -= tmp
        dfdx[j] += tmp

    def as_dict(self):
        return {
            'code': 'V4',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'gamma': self.gamma,
        }

    @classmethod
    def from_dict(cls, d):
        return V4Map(d['n'], d['i'], d['j'], d['gamma'])


class WMap(FeatureNormalizer):

    def __init__(self, n, i, j, k, gammai, gammaj):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.gammai = gammai
        self.gammaj = gammaj

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x):
        n, i, j, k = self.n, self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        y[n] = (gammai*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k])/(1 + gammai*x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j, k = self.n, self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        dfdx[i] -= dfdy[n] * ((gammai**2*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k])/(1 + gammai*x[i])**2)
        dfdx[j] -= dfdy[n] * (gammai*(gammaj/(1 + gammaj*x[j]))**1.5*x[k])/(2.*(1 + gammai*x[i]))
        dfdx[k] += dfdy[n] * (gammai*np.sqrt(gammaj/(1 + gammaj*x[j])))/(1 + gammai*x[i])

    def as_dict(self):
        return {
            'code': 'W',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'k': self.k,
            'gammai': self.gammai,
            'gammaj': self.gammaj
        }

    @classmethod
    def from_dict(cls, d):
        return WMap(d['n'], d['i'], d['j'], d['k'],
                    d['gammai'], d['gammaj'])


class XMap(FeatureNormalizer):

    def __init__(self, n, i, j, k, gammai, gammaj):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.gammai = gammai
        self.gammaj = gammaj

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x):
        n, i, j, k = self.n, self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        y[n] = np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k]

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j, k = self.n, self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        dfdx[i] -= dfdy[n] * ((gammai*np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k])/(2 + 2*gammai*x[i]))
        dfdx[j] -= dfdy[n] * ((gammaj*np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k])/(2 + 2*gammaj*x[j]))
        dfdx[k] += dfdy[n] * np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))

    def as_dict(self):
        return {
            'code': 'X',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'k': self.k,
            'gammai': self.gammai,
            'gammaj': self.gammaj
        }

    @classmethod
    def from_dict(cls, d):
        return XMap(d['n'], d['i'], d['j'], d['k'],
                    d['gammai'], d['gammaj'])


class YMap(FeatureNormalizer):

    def __init__(self, n, i, j, k, l, gammai, gammaj, gammak):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.gammai = gammai
        self.gammaj = gammaj
        self.gammak = gammak

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 4

    def fill_feat_(self, y, x):
        n, i, j, k, l = self.n, self.i, self.j, self.k, self.l
        gammai, gammaj, gammak = self.gammai, self.gammaj, self.gammak
        y[n] = np.sqrt(gammai/(1 + gammai*x[i]))*x[l]*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k]))

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j, k, l = self.n, self.i, self.j, self.k, self.l
        gammai, gammaj, gammak = self.gammai, self.gammaj, self.gammak
        dfdx[i] -= dfdy[n] * ((gammai*np.sqrt(gammai/(1 + gammai*x[i]))*x[l]*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k])))/(2 + 2*gammai*x[i]))
        dfdx[j] -= dfdy[n] * ((gammaj*np.sqrt(gammai/(1 + gammai*x[i]))*x[l]*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k])))/(2 + 2*gammaj*x[j]))
        dfdx[k] -= dfdy[n] * ((gammak*np.sqrt(gammai/(1 + gammai*x[i]))*x[l]*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k])))/(2 + 2*gammak*x[k]))
        dfdx[l] += dfdy[n] * np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k]))

    def as_dict(self):
        return {
            'code': 'Y',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'k': self.k,
            'l': self.l,
            'gammai': self.gammai,
            'gammaj': self.gammaj,
            'gammak': self.gammak
        }

    @classmethod
    def from_dict(cls, d):
        return YMap(d['n'], d['i'], d['j'], d['k'], d['l'],
                    d['gammai'], d['gammaj'], d['gammak'])


class ZMap(FeatureNormalizer):

    def __init__(self, n, i, gamma, scale=1.0, center=0.0,
                 bounds=None):
        self.n = n
        self.i = i
        self.gamma = gamma
        self.scale = scale
        self.center = center
        self._bounds = bounds or (-self.center, self.scale-self.center)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = -self.center + self.scale / (1 + self.gamma * x[i]**2)

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i = self.n, self.i
        dfdx[i] -= 2 * dfdy[n] * self.scale * self.gamma * x[i] / (1 + self.gamma * x[i]**2)**2

    def as_dict(self):
        return {
            'code': 'Z',
            'n': self.n,
            'i': self.i,
            'gamma': self.gamma,
            'scale': self.scale,
            'center': self.center,
            'bounds': self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return ZMap(d['n'], d['i'], d['gamma'], d['scale'], d['center'],
                    bounds=d.get('bounds'))


class SignedUMap(FeatureNormalizer):

    def __init__(self, n, i, gamma):
        self.n = n
        self.i = i
        self.gamma = gamma

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = x[i] / np.sqrt(self.gamma + x[i] * x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i = self.n, self.i
        dfdx[i] += dfdy[n] * self.gamma / (self.gamma + x[i] * x[i])**1.5

    def as_dict(self):
        return {
            'code': 'SU',
            'n': self.n,
            'i': self.i,
            'gamma': self.gamma
        }

    @classmethod
    def from_dict(cls, d):
        return SignedUMap(d['n'], d['i'], d['gamma'])


class FeatureList():

    def __init__(self, feat_list):
        self.feat_list = feat_list
        self.nfeat = len(self.feat_list)

    def __getitem__(self, index):
        return self.feat_list[index]

    def __call__(self, xdesc):
        # xdesc (nsamp, ninp)
        xdesc = xdesc.T
        # now xdesc (ninp, nsamp)
        tdesc = np.zeros((self.nfeat, xdesc.shape[1]))
        for i in range(self.nfeat):
            self.feat_list[i].fill_feat_(tdesc, xdesc)
        return tdesc.T

    @property
    def bounds_list(self):
        return [f.bounds for f in self.feat_list]

    def fill_vals_(self, tdesc, xdesc):
        for i in range(self.nfeat):
            self.feat_list[i].fill_feat_(tdesc, xdesc)

    def fill_derivs_(self, dfdx, dfdy, xdesc):
        """
        Fill dfdx wth the derivatives of f with respect
        to the raw features x, given the derivative
        with respect to transformed features dfdy.

        Args:
            dfdx (nraw, nsamp): Output. Derivative of output wrt
                raw features x
            dfdy (ntrans, nsamp): Input. Derivative of output wrt transformed
                features y
            xdesc (nraw, nsamp): Raw features x
        """
        for i in range(self.nfeat):
            self.feat_list[i].fill_deriv_(dfdx, dfdy, xdesc)

    def as_dict(self):
        d = {
            'nfeat': self.nfeat,
            'feat_list': [f.as_dict() for f in self.feat_list]
        }
        return d

    def dump(self, fname):
        d = self.as_dict()
        with open(fname, 'w') as f:
            yaml.dump(d, f)

    @classmethod
    def from_dict(cls, d):
        return cls([FeatureNormalizer.from_dict(d['feat_list'][i])\
                    for i in range(len(d['feat_list']))])

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            d = yaml.load(f, Loader=yaml.Loader)
        return cls.from_dict(d)


"""
Examples of FeatureList objects:
center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(8.0, gamma0b)
center0c = get_vmap_heg_value(0.5, gamma0c)
lst = [
        UMap(0, 1, gammax),
        VMap(1, 2, 1, scale=2.0, center=1.0),
        VMap(2, 3, gamma0a, scale=1.0, center=center0a),
        UMap(3, 4, gamma1),
        UMap(4, 5, gamma2),
        WMap(5, 1, 5, 6, gammax, gamma2),
        XMap(6, 1, 4, 7, gammax, gamma1),
        VMap(7, 8, gamma0b, scale=1.0, center=center0b),
        VMap(8, 9, gamma0c, scale=1.0, center=center0c),
        YMap(9, 1, 4, 5, 10, gammax, gamma1, gamma2),
]
flst = FeatureList(lst)

center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(3.0, gamma0b)
UMap(0, 1, gammax)
VMap(1, 2, 1, scale=2.0, center=1.0)
VMap(2, 3, gamma0a, scale=1.0, center=center0a)
UMap(3, 4, gamma1)
UMap(4, 5, gamma2)
WMap(5, 1, 5, 6, gammax, gamma2)
XMap(6, 1, 4, 7, gammax, gamma1)
VMap(7, 8, gamma0b, scale=1.0, center=center0b)
YMap(8, 1, 4, 5, 9, gammax, gamma1, gamma2)
"""

