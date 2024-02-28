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
from interpolation.splines.eval_cubic_numba import vec_eval_cubic_splines_G_1,\
                                                   vec_eval_cubic_splines_G_2,\
                                                   vec_eval_cubic_splines_G_3,\
                                                   vec_eval_cubic_splines_G_4
from ciderpress.dft.cider_kernel import ds2, dalpha, contract_descriptors_l0, \
                                        contract_descriptors_l0_d, \
                                        contract_descriptors_h
import yaml


def get_vec_eval(grid, coeffs, X, N):
    """
    Call the numba-accelerated spline evaluation routines from the
    interpolation package. Also returns derivatives
    Args:
        grid: start and end points + number of grids in each dimension
        coeffs: coefficients of the spline
        X: coordinates to interpolate
        N: dimension of the interpolation (between 1 and 4, inclusive)
    """
    coeffs = np.expand_dims(coeffs, coeffs.ndim)
    y = np.zeros((X.shape[0], 1))
    dy = np.zeros((X.shape[0], N, 1))
    a_, b_, orders = zip(*grid)
    if N == 1:
        vec_eval_cubic_splines_G_1(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 2:
        vec_eval_cubic_splines_G_2(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 3:
        vec_eval_cubic_splines_G_3(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 4:
        vec_eval_cubic_splines_G_4(a_, b_, orders,
                                   coeffs, X, y, dy)
    else:
        raise ValueError('invalid dimension N')
    return np.squeeze(y, -1), np.squeeze(dy, -1)


def functional_derivative_loop_b(dEddesc, rho, sigma, tau):
    v_npalpha = dEddesc[:3]
    v_nst = np.zeros(v_npalpha.shape)
    # dE/dn lines 1-3
    v_nst[0] = v_npalpha[0]
    dpdn, dpdsigma = ds2(rho, sigma)
    # dE/dn line 4 term 1
    v_nst[0] += v_npalpha[1] * dpdn
    # dE/dsigma term 1
    v_nst[1] += v_npalpha[1] * dpdsigma
    dadn, dadsigma, dadtau = dalpha(rho, sigma, tau)
    # dE/dn line 4 term 2
    v_nst[0] += v_npalpha[2] * dadn
    # dE/dsigma term 2
    v_nst[1] += v_npalpha[2] * dadsigma
    # dE/dtau
    v_nst[2] = v_npalpha[2] * dadtau
    return v_nst, dEddesc[3:].copy()

functional_derivative_loop_h = functional_derivative_loop_b

def functional_derivative_loop_d(dEddesc, rho, sigma):
    v_np = dEddesc[:2]
    v_ns = np.zeros(v_np.shape)
    # dE/dn lines 1-3
    v_ns[0] = v_np[0]
    dpdn, dpdsigma = ds2(rho, sigma)
    # dE/dn line 4 term 1
    v_ns[0] += v_np[1] * dpdn
    # dE/dsigma term 1
    v_ns[1] += v_np[1] * dpdsigma
    return v_ns, dEddesc[2:].copy()


class XCEvalSerializable:

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError

    def dump(self, fname):
        """
        Save the Evaluator to a file name fname as yaml format.
        """
        state_dict = self.to_dict()
        with open(fname, 'w') as f:
            yaml.dump(state_dict, f)

    @classmethod
    def load(cls, fname):
        """
        Load an instance of this class from yaml
        """
        with open(fname, 'r') as f:
            state_dict = yaml.load(f, Loader=yaml.CLoader)
        return cls.from_dict(state_dict)


class KernelEvalBase:

    mode = None
    feature_list = None
    _mul_basefunc = None
    _add_basefunc = None

    @property
    def N1(self):
        raise NotImplementedError

    def get_descriptors(self, X0T):
        """
        Compute and return transformed descriptor matrix X1.

        Args:
            rho: Density/Kinetic energy density representation.
            X0 (nspin, N0, Nsamp): Raw features
            nspin (1 or 2): Number of spins

        Returns:
            X1 (Nsamp, N1)
        """
        nspin, N0, Nsamp = X0T.shape
        N1 = self.N1
        if self.mode == 'SEP':
            X1 = np.zeros((nspin, Nsamp, N1))
            for s in range(nspin):
                self.feature_list.fill_vals_(X1[s].T, X0T[s])
            X1 = X1.reshape(nspin * Nsamp, N1)
        elif self.mode == 'NPOL':
            X0T_sum = X0T.mean(0)
            X1 = np.zeros((Nsamp, N1))
            self.feature_list.fill_vals_(X1.T, X0T_sum)
        else:
            raise NotImplementedError
        return X1

    def get_descriptors_with_mul(self, X0T, multiplier):
        """
        Compute and return transformed descriptor matrix X1.

        Args:
            rho: Density/Kinetic energy density representation.
            X0 (nspin, N0, Nsamp): Raw features
            nspin (1 or 2): Number of spins

        Returns:
            X1 (Nsamp, N1)
        """
        nspin, N0, Nsamp = X0T.shape
        N1 = self.N1
        if self.mode == 'SEP':
            X1 = np.zeros((nspin, Nsamp, N1))
            for s in range(nspin):
                self.feature_list.fill_vals_(X1[s].T, X0T[s])
                X1[s] *= multiplier[:, None]
            X1 = X1.reshape(nspin * Nsamp, N1)
        elif self.mode == 'NPOL':
            X0T_sum = X0T.mean(0)
            X1 = np.zeros((Nsamp, N1))
            self.feature_list.fill_vals_(X1.T, X0T_sum)
            X1[:] *= multiplier[:, None]
        else:
            raise NotImplementedError
        return X1

    def _baseline(self, X0T, base_func):
        nspin, N0, Nsamp = X0T.shape
        if self.mode == 'SEP':
            ms = []
            dms = []
            for s in range(nspin):
                m, dm = base_func(X0T[s : s + 1])
                ms.append(m / nspin)
                dms.append(dm / nspin)
            return np.stack(ms), np.concatenate(dms, axis=0)
        else:
            raise NotImplementedError

    def multiplicative_baseline(self, X0T):
        return self._baseline(X0T, self._mul_basefunc)

    def additive_baseline(self, X0T):
        return self._baseline(X0T, self._add_basefunc)

    def apply_descriptor_grad(self, X0T, dfdX1):
        """

        Args:
            X0T (nspin, N0, Nsamp): raw feature descriptors
            dfdX1 (nspin * Nsamp, N1): derivative with respect to transformed
                descriptors X1

        Returns:
            dfdX0T (nspin, N0, Nsamp): derivative with respect
                to raw descriptors X0.
        """
        nspin, N0, Nsamp = X0T.shape
        N1 = self.N1
        if self.mode == 'SEP':
            dfdX0T = np.zeros_like(X0T)
            dfdX1 = dfdX1.reshape(nspin, Nsamp, N1)
            for s in range(nspin):
                self.feature_list.fill_derivs_(dfdX0T[s], dfdX1[s].T, X0T[s])
        elif self.mode == 'NPOL':
            dfdX0T = np.zeros_like(X0T)
            self.feature_list.fill_derivs_(dfdX0T[0], dfdX1.T, X0T.mean(0))
            for s in range(1, nspin):
                dfdX0T[s] = dfdX0T[0]
            dfdX0T /= nspin
        else:
            raise NotImplementedError
        return dfdX0T

    def apply_baseline(self, X0T, f, dfdX0T=None, add_base=True):
        """

        Args:
            f:
            dfdX0T:
            add_base:

        Returns:

        """
        m, dm = self.multiplicative_baseline(X0T)
        add_base = add_base and self.additive_baseline is not None
        if add_base:
            a, da = self.additive_baseline(X0T)

        res = f * m
        if add_base:
            res += a
        if dfdX0T is not None:
            dres = dfdX0T * m[:, np.newaxis] + f[:, np.newaxis] * dm
            if add_base:
                dres += da
            return res, dres
        return res


class FuncEvaluator:

    def build(self, *args, **kwargs):
        pass

    def __call__(self, X1, res=None, dres=None):
        raise NotImplementedError


class KernelEvaluator(FuncEvaluator, XCEvalSerializable):

    def __init__(self, kernel, X1ctrl, alpha):
        self.X1ctrl = X1ctrl
        self.kernel = kernel
        self.alpha = alpha

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        elif res.shape != X1.shape[:1]:
            raise ValueError
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        N = X1.shape[0]
        dn = 2000
        for i0 in range(0, N, dn):
            i1 = min(N, i0 + dn)
            k, dk = self.kernel.k_and_deriv(X1[i0:i1], self.X1ctrl)
            res[i0:i1] += k.dot(self.alpha)
            dres[i0:i1] += np.einsum('gcn,c->gn', dk, self.alpha)
        return res, dres


class SplineSetEvaluator(FuncEvaluator, XCEvalSerializable):

    def __init__(self, scale, ind_sets, spline_grids, coeff_sets,
                 const=0):
        self.scale = scale
        self.nterms = len(self.scale)
        assert len(ind_sets) == self.nterms
        self.ind_sets = ind_sets
        assert len(spline_grids) == self.nterms
        self.spline_grids = spline_grids
        assert len(coeff_sets) == self.nterms
        self.coeff_sets = coeff_sets
        self.const = const

    def __call__(self, X1, res=None, dres=None):
        """
        Note: This function adds to res and dres if they are passed,
        rather than writing them from scratch.

        Args:
            X1:
            res:
            dres:

        Returns:

        """
        if res is None:
            res = np.zeros(X1.shape[0]) + self.const
        elif res.shape != X1.shape[:1]:
            raise ValueError
        else:
            res[:] += self.const
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        for t in range(self.nterms):
            ind_set = self.ind_sets[t]
            y, dy = get_vec_eval(self.spline_grids[t],
                                 self.coeff_sets[t],
                                 X1[:, ind_set],
                                 len(ind_set))
            res[:] += y * self.scale[t]
            dres[:, ind_set] += dy * self.scale[t]
        return res, dres

    def to_dict(self):
        return {
            'scale': np.array(self.scale),
            'ind_sets': self.ind_sets,
            'spline_grids': self.spline_grids,
            'coeff_sets': self.coeff_sets,
            'const': self.const,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d['scale'],
            d['ind_sets'],
            d['spline_grids'],
            d['coeff_sets'],
            const=d['const'],
        )


class NNEvaluator(FuncEvaluator, XCEvalSerializable):

    def __init__(self, model, device=None):
        self.model = model
        self.device = device
        import torch
        from torch.autograd import grad
        self.cuda_is_available = torch.cuda.is_available
        self.device_init = torch.device
        self.tensor_init = torch.tensor
        self.eval_grad = grad
        self.set_device(device=device)

    def set_device(self, device=None):
        if device is None:
            if self.cuda_is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        if isinstance(device, str):
            device = self.device_init(device)
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

    def build(self, *args, **kwargs):
        self.set_device()

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        elif res.shape != X1.shape[:1]:
            raise ValueError
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        X1_torch = self.tensor_init(X1, device=self.device, requires_grad=True)
        self.model.zero_grad()
        output = self.model(X1_torch)
        output_grad = self.eval_grad(output.sum(), X1_torch, retain_graph=False)[0]
        res[:] += output.cpu().detach().numpy()
        dres[:] += output_grad.cpu().detach().numpy()
        return res, dres


class GlobalLinearEvaluator(FuncEvaluator, XCEvalSerializable):

    def __init__(self, consts):
        self.consts = np.asarray(consts, dtype=np.float64, order='C')

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        elif res.shape != X1.shape[:1]:
            raise ValueError
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        res[:] += X1.dot(self.consts)
        dres[:] += self.consts
        return res, dres


class MappedDFTKernel(KernelEvalBase, XCEvalSerializable):
    """
    This class evaluates the XC term arising from a single DFTKernel
    object, using one or a list of FuncEvaluator objects.
    """
    def __init__(self, fevals, feature_list, mode,
                 multiplicative_baseline, additive_baseline=None):
        self.fevals = fevals
        if not isinstance(self.fevals, list):
            self.fevals = [self.fevals]
        for feval in self.fevals:
            if not isinstance(feval, FuncEvaluator):
                raise ValueError
        self.mode = mode
        self.feature_list = feature_list
        self._mul_basefunc = multiplicative_baseline
        self._add_basefunc = additive_baseline

    @property
    def N1(self):
        return self.feature_list.nfeat

    def __call__(self, X0T, add_base=True, rhocut=0):
        X1 = self.get_descriptors(X0T)
        Nsamp_internal = X1.shape[0]
        f = np.zeros(Nsamp_internal)
        df = np.zeros((Nsamp_internal, self.N1))
        for feval in self.fevals:
            feval(X1, f, df)
        if self.mode == 'SEP':
            f = f.reshape(X0T.shape[0], -1)
        dfdX0T = self.apply_descriptor_grad(X0T, df)
        res, dres = self.apply_baseline(X0T, f, dfdX0T)
        if rhocut > 0:
            if self.mode == 'SEP':
                cond = X0T[:, 0] < rhocut
                for s in range(X0T.shape[0]):
                    res[s][cond[s]] = 0.0
                    dres[s][:, cond[s]] = 0.0
            else:
                cond = X0T[:, 0].sum(0) < rhocut
                res[..., cond] = 0.0
                dres[..., cond] = 0.0
        if self.mode == 'SEP':
            res = res.sum(0)
        return res, dres

    def to_dict(self):
        return {
            'fevals': [fe.to_dict() for fe in self.fevals],
            'feature_list': self.feature_list.to_dict(),
            'mode': self.mode,
        }

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError


class MappedFunctional:

    def __init__(
            self,
            mapped_kernels,
            desc_params,
            libxc_baseline=None
    ):
        self.kernels = mapped_kernels
        self.desc_params = desc_params
        self.libxc_baseline = libxc_baseline

    @property
    def desc_version(self):
        return self.desc_params.version[0]

    @property
    def vvmul(self):
        return self.desc_params.vvmul

    @property
    def a0(self):
        return self.desc_params.a0

    @property
    def fac_mul(self):
        return self.desc_params.fac_mul

    @property
    def amin(self):
        return self.desc_params.amin

    @property
    def feature_list(self):
        # TODO misleading because this is X functional only
        return self.kernels[0].feature_list

    def set_baseline_mode(self, mode):
        # TODO baseline can be GPAW or PySCF mode.
        # Need to implement for more complicated XC.
        raise NotImplementedError

    def __call__(self, X0T, rhocut=0):
        res, dres = 0, 0
        for kernel in self.kernels:
            tmp, dtmp = kernel(X0T, rhocut=rhocut)
            res += tmp
            dres += dtmp
        return res, dres


class _FunctionalEvaluatorBase:

    version_list = None

    def __init__(self, gpxc, amix=1.0, rhocut=1e-8, mode='pyscf'):
        if gpxc.desc_params.version[0] not in self.version_list:
            raise ValueError
        self.amix = amix
        self.gpxc = gpxc
        self.rhocut = rhocut
        self.mode = mode
        # TODO not sure if this is the best way to handle small density
        self.numerical_epsilon = rhocut**2


class GGAFunctionalEvaluator(_FunctionalEvaluatorBase):

    version_list = ['d']

    def __call__(self, e_g, nt_sg, sigma_xg, feat_sg,
                 v_sg, dedsigma_xg, vfeat_sg, RHOCUT=None):

        rhocut = RHOCUT or self.rhocut

        nspin = nt_sg.shape[0]
        N = e_g.size
        Nfeat = self.gpxc.desc_params.size
        X0T = np.zeros((nspin, Nfeat, N))
        gshape = nt_sg[0].shape
        if self.gpxc.libxc_baseline is not None:
            raise NotImplementedError
        for s in range(nspin):
            rho = nspin * nt_sg[s].reshape(-1) + self.numerical_epsilon
            sigma = nspin * nspin * sigma_xg[2 * s].reshape(-1)
            feat = nspin * feat_sg[s].reshape(feat_sg[s].shape[0],-1)
            X0T[s, :] = contract_descriptors_l0_d(rho, sigma, feat)[:Nfeat]
            rho = sigma = feat = None

        rho = nt_sg.sum(0)
        cond = rho > rhocut
        gcond = cond.reshape(gshape)
        exc_ml, vxc_ml = self.gpxc(X0T, rhocut=rhocut)
        vxc_ml *= self.amix
        e_g[gcond] += self.amix * exc_ml.reshape(gshape)[gcond]

        for s in range(nspin):
            rho = nspin * nt_sg[s].reshape(-1) + self.numerical_epsilon
            sigma = nspin * nspin * sigma_xg[2 * s].reshape(-1)
            v_nst, vfeat = functional_derivative_loop_d(
                vxc_ml[s], rho, sigma
            )
            cond = rho > rhocut
            gcond = cond.reshape(gshape)
            vfeat_sg[s][:, gcond] += nspin * vfeat.reshape(Nfeat - 2, *gshape)[:, gcond]
            v_sg[s, gcond] += nspin * v_nst[0].reshape(*gshape)[gcond]
            dedsigma_xg[2 * s, gcond] += nspin * nspin * v_nst[1].reshape(*gshape)[gcond]
            rho = sigma = feat = None


class MGGAFunctionalEvaluator(_FunctionalEvaluatorBase):

    version_list = ['b', 'h', 'i', 'k']

    def __call__(self, e_g, nt_sg, sigma_xg, tau_sg, feat_sg,
                 v_sg, dedsigma_xg, dedtau_sg, vfeat_sg, RHOCUT=None):

        rhocut = RHOCUT or self.rhocut

        nspin = nt_sg.shape[0]
        N = e_g.size
        Nfeat = self.gpxc.desc_params.size
        X0T = np.zeros((nspin, Nfeat, N))
        gshape = nt_sg[0].shape
        if self.gpxc.libxc_baseline is not None:
            raise NotImplementedError
        for s in range(nspin):
            rho = nspin * nt_sg[s].reshape(-1) + self.numerical_epsilon
            sigma = nspin * nspin * sigma_xg[2 * s].reshape(-1)
            tau = nspin * tau_sg[s].reshape(-1)
            feat = nspin * feat_sg[s].reshape(feat_sg[s].shape[0],-1)
            X0T[s, :] = contract_descriptors_h(rho, sigma, tau, feat)[:Nfeat]
            rho = sigma = tau = feat = None

        rho = nt_sg.sum(0)
        cond = rho > rhocut
        gcond = cond.reshape(gshape)
        exc_ml, vxc_ml = self.gpxc(X0T, rhocut=rhocut)
        vxc_ml *= self.amix
        e_g[gcond] += self.amix * exc_ml.reshape(gshape)[gcond]

        for s in range(nspin):
            rho = nspin * nt_sg[s].reshape(-1) + self.numerical_epsilon
            sigma = nspin * nspin * sigma_xg[2 * s].reshape(-1)
            tau = nspin * tau_sg[s].reshape(-1)
            v_nst, vfeat = functional_derivative_loop_h(
                vxc_ml[s], rho, sigma, tau
            )
            cond = rho > rhocut
            gcond = cond.reshape(gshape)
            vfeat_sg[s][:, gcond] += nspin * vfeat.reshape(Nfeat - 3, *gshape)[:, gcond]
            v_sg[s, gcond] += nspin * v_nst[0].reshape(*gshape)[gcond]
            dedsigma_xg[2 * s, gcond] += nspin * nspin * v_nst[1].reshape(*gshape)[gcond]
            dedtau_sg[s, gcond] += nspin * v_nst[2].reshape(*gshape)[gcond]
            rho = sigma = tau = feat = None
