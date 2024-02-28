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

from ciderpress.dft.futil import sph_nlxc_mod as fnlxc
from ciderpress.gpaw.atom_utils import PASDWCiderKernel, AtomPASDWSlice

from gpaw.calculator import GPAW
from gpaw.xc.libxc import LibXC
from gpaw.xc.gga import gga_vars
from gpaw.xc.mgga import MGGA

from ciderpress.dft.xc_models import NormGPFunctional
from ciderpress.gpaw.interp_paw import DiffGGA, DiffMGGA
from ciderpress.gpaw.cider_fft import CiderGGAHybridKernelPBEPot, \
    CiderGGAHybridKernel, CiderMGGAHybridKernel, LDict, \
    DEFAULT_RHO_TOL, CiderGGA, CiderMGGA
from ciderpress.dft.xc_evaluator import MappedFunctional

import numpy as np
import joblib
import yaml


class CiderGPAW(GPAW):
    """
    This class is equivalent to the GPAW calculator object
    provded in the gpaw pacakage, except that it is able to load
    and save CIDER calculations. The GPAW object can run CIDER but
    not save/load CIDER calculations.

    One can also provide CIDER XC functional in dictionary format,
    but this is not advised since one also has to explicitly provide
    by the mlfunc_data parameter, which is the str of the yaml file
    representing the mlfunc object.
    """

    def __init__(self,
                 restart=None,
                 *,
                 label=None,
                 timer=None,
                 communicator=None,
                 txt='?',
                 parallel=None,
                 **kwargs):
        if (isinstance(kwargs.get('xc'), dict)
                and '_cider_type' in kwargs['xc']):
            if 'mlfunc_data' not in kwargs:
                raise ValueError('Must provide mlfunc_data for CIDER.')
            self._mlfunc_data = kwargs.pop('mlfunc_data')
        super(CiderGPAW, self).__init__(
            restart=restart,
            label=label,
            timer=timer,
            communicator=communicator,
            txt=txt,
            parallel=parallel,
            **kwargs
        )

    def _write(self, writer, mode):

        writer = super(CiderGPAW, self)._write(writer, mode)
        if hasattr(self.hamiltonian.xc, 'get_mlfunc_data'):
            mlfunc_data = self.hamiltonian.xc.get_mlfunc_data()
            writer.child('mlfunc').write(mlfunc_data=mlfunc_data)
        return writer


    def initialize(self, atoms=None, reading=False):
        xc = self.parameters.xc
        if isinstance(xc, dict) and '_cider_type' in xc:
            is_cider = 'Cider' in xc['_cider_type']
            if is_cider:
                if reading:
                    self._mlfunc_data = self.reader.mlfunc.get('mlfunc_data')
                xc['mlfunc'] = self._mlfunc_data
            self.parameters.xc = cider_functional_from_dict(xc)
            if is_cider:
                del self._mlfunc_data
        GPAW.initialize(self, atoms, reading)


def get_cider_functional(
    mlfunc,
    mlfunc_format=None,
    slx='GGA_X_PBE',
    slc='GGA_C_PBE',
    xmix=0.25,
    use_paw=True,
    pasdw_ovlp_fit=True,
    pasdw_store_funcs=False,
    Nalpha=None,
    qmax=300,
    lambd=1.8,
    debug=False,
    no_paw_atom_kernel=False,
    _force_nonlocal=False,
):
    """
    Initialize a CIDER surrogate hybrid exchange function of the form
    E_xc = E_x^sl * (1 - xmix) + E_c^sl + xmix * E_x^CIDER
    where E_x^sl is given by the slx parameter, E_c^sl is given by the slc
    parameter, and E_x^CIDER is the ML exchange energy contains in mlfunc.

    NOTE: Do not use CIDER with ultrasoft pseudopotentials (PAW only).
    At your own risk, you can use CIDER with norm-conserving pseudopotentials,
    but correctness is not guaranteed because the nonlocal features will
    not be correct due to the lack of core electrons.

    NOTE: If the mlfunc is determined to be semi-local, all the
    internal settings are ignored, and a simpler, more efficient
    class is returned to evaluate the semi-local functional.

    Args:
        mlfunc (NormGPFunctional, str): An ML functional object or a str
            corresponding to the file name of a yaml or joblib file
            containing it.
        mlfunc_format (str, None): 'joblib' or 'yaml', specifies the format
            of mlfunc if it is a string corresponding to a file name.
            If unspecified, infer from file extension and raise error
            if file type cannot be determined.
        slx (str, GGA_X_PBE): libxc code str for semi-local X functional
        slc (str, GGA_C_PBE): libxc code str for semi-local C functional
        xmix (float, 0.25): Mixing parameter for CIDER exchnange.
        use_paw (bool, True): Whether to compute PAW corrections. This
            should be True unless you plan to use norm-conserving
            pseudopotentials (NCPPs). Note, the use of NCPPs is allowed
            but not advised, as large errors might arise due to the use
            of incorrect nonlocal features.

    The following parameters are only used if use_paw is True. They
    control internal behavior of the PASDW algorithm used to
    evaluate the nonlocal density features in CIDER, but users might
    want to set them to address numerical stability or memory/performance
    tradeoff issues.

    PASDW args:
        pasdw_ovlp_fit (bool, True): Whether to use overlap fitting to
            attempt to improve numerical precision of PASDW projections.
            Default to True. Impact of this parameter should be minor.
        pasdw_store_funcs (bool, False): Whether to store projector
            functions used in the PASDW routine. Defaults to False
            because this can be memory intensive, but if you have the
            space, the computational cost of the atomic corrections
            can be greatly reduced by setting to True.

    The following additional arguments can be set but are considered
    internal parameters and should usually be kept as their defaults,
    which should be reasonable for most cases.

    Internal args:
        Nalpha (int, None):
            Number of interpolation points for the nonlocal feature kernel.
            If None, set automatically based on lambd, qmax, and qmin.
        qmax (float, 300):
            Maximum value of q to use for kernel interpolation on FFT grid.
            Default should be fine for most cases.
        * qmin (not currently a parameter, set automatically):
            Mininum value of q to use for kernel interpolation.
            Currently, qmin is set automatically based
            on the minimum regularlized value of the kernel exponent.
        lambd (float, 1.8):
            Density of interpolation points. q_alpha=q_0 * lambd**alpha.
            Smaller lambd is more expensive and more precise.

    The following options are for debugging only. Do not set them unless
    you want to test numerical stability issues, as the energies will
    not be variational. Only supported for GGA-type functionals.

    Debug args:
        debug (bool, False): Use CIDER energy and semi-local XC potential.
        no_paw_atom_kernel: Use CIDER energy and semi-local XC potential
            for PAW corrections, and CIDER energy/potential on FFT grid.
        _force_nonlocal (bool, False): Use nonlocal kernel even if nonlocal
            features are not required. For debugging use.

    Returns:
        A _CiderBase object (specific class depends on the parameters
        provided above, specifically on whether use_paw is True,
        whether the functional is GGA or MGGA form, and whether the
        functional is semi-local or nonlocal.)
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
    if not isinstance(mlfunc, (NormGPFunctional, MappedFunctional)):
        raise ValueError('mlfunc must be NormGPFunctional')

    if mlfunc.desc_version == 'b':
        if mlfunc.feature_list.nfeat == 2 and not _force_nonlocal:
            # functional is semi-local MGGA
            cider_kernel = SLCiderMGGAHybridWrapper(mlfunc, xmix, slx, slc)
            return SLCiderMGGA(cider_kernel)
        if debug:
            raise NotImplementedError
        else:
            cider_kernel = CiderMGGAHybridKernel(mlfunc, xmix, slx, slc)
        if use_paw:
            cls = CiderMGGAPASDW
        else:
            cls = CiderMGGA
    elif mlfunc.desc_version == 'd':
        if mlfunc.feature_list.nfeat == 1 and not _force_nonlocal:
            # functional is semi-local GGA
            cider_kernel = SLCiderGGAHybridWrapper(mlfunc, xmix, slx, slc)
            return SLCiderGGA(cider_kernel)
        if debug:
            cider_kernel = CiderGGAHybridKernelPBEPot(mlfunc, xmix, slx, slc)
        else:
            cider_kernel = CiderGGAHybridKernel(mlfunc, xmix, slx, slc)
        if use_paw:
            cls = CiderGGAPASDW
        else:
            cls = CiderGGA
    else:
        msg = "Only implemented for b and d version, found {}"
        raise ValueError(msg.format(mlfunc.desc_version))

    const_list = _get_const_list(mlfunc)
    nexp = 4

    if use_paw:
        xc = cls(cider_kernel, nexp, const_list, Nalpha=Nalpha,
                 lambd=lambd, encut=qmax,
                 pasdw_ovlp_fit=pasdw_ovlp_fit,
                 pasdw_store_funcs=pasdw_store_funcs)
    else:
        xc = cls(cider_kernel, nexp, const_list, Nalpha=Nalpha,
                 lambd=lambd, encut=qmax)

    if no_paw_atom_kernel:
        if mlfunc.desc_version == 'b':
            raise NotImplementedError
        xc.debug_kernel = CiderGGAHybridKernelPBEPot(
            mlfunc, xmix, slx, slc
        )

    return xc


def cider_functional_from_dict(d):
    """
    This is a function for reading CIDER functionals (and also
    standard functionals using the DiffPAW approach) from a dictionary
    that can be stored in a .gpw file. This is primarily a helper function
    for internal use, as it just reads the functional type and calls
    the from_dict() function of the respective class.

    Args:
        d: XC data

    Returns:
        XC Functional object for use in GPAW.
    """
    cider_type = d.pop('_cider_type')
    if cider_type == 'CiderGGAPASDW':
        cls = CiderGGAPASDW
        kcls = CiderGGAHybridKernel
    elif cider_type == 'CiderMGGAPASDW':
        cls = CiderMGGAPASDW
        kcls = CiderMGGAHybridKernel
    elif cider_type == 'CiderGGA':
        cls = CiderGGA
        kcls = CiderGGAHybridKernel
    elif cider_type == 'CiderMGGA':
        cls = CiderMGGA
        kcls = CiderMGGAHybridKernel
    elif cider_type == 'SLCiderGGA':
        cls = SLCiderGGA
        kcls = SLCiderGGAHybridWrapper
    elif cider_type == 'SLCiderMGGA':
        cls = SLCiderMGGA
        kcls = SLCiderMGGAHybridWrapper
    elif cider_type == 'DiffGGA':
        cls = DiffGGA
        kcls = None
    elif cider_type == 'DiffMGGA':
        cls = DiffMGGA
        kcls = None
    else:
        raise ValueError('Unsupported functional type')

    if 'Cider' in cider_type:
        mlfunc_dict = yaml.load(d['mlfunc'], Loader=yaml.CLoader)
        mlfunc = NormGPFunctional.from_dict(mlfunc_dict)
        const_list = _get_const_list(mlfunc)
        nexp = 4
        # kernel_params should be xmix, slx, slc
        cider_kernel = kcls(mlfunc, **(d['kernel_params']))
        if 'SLCider' in cider_type:
            xc = cls(cider_kernel)
        else:
            # xc_params should have Nalpha, lambd, encut.
            # For PAW, it should also have pasdw_ovlp_fit, pasdw_store_funcs.
            xc = cls(cider_kernel, nexp, const_list, **(d['xc_params']))
    else:
        xc = cls(LibXC(d['name']))

    return xc


def _get_const_list(mlfunc):
    consts = np.array([0.00, mlfunc.a0, mlfunc.fac_mul, mlfunc.amin])
    const_list = np.stack([
        0.5 * consts,
        1.0 * consts,
        2.0 * consts,
        consts * mlfunc.vvmul
    ])
    return const_list



class _CiderPASDW_MPRoutines():

    has_paw = True

    def __init__(self, *args, **kwargs):
        # This is a base class to store the common
        # functions between GGA and MGGA PASDW.
        # Most routines are the same between the two.
        raise NotImplementedError

    def get_D_asp(self):
        return self.atomdist.to_work(self.dens.D_asp)

    def initialize_paw_kernel(self, cider_kernel_inp, Nalpha_atom, encut_atom):
        self.paw_kernel = PASDWCiderKernel(
            cider_kernel_inp,
            self.nexp,
            self.consts,
            Nalpha_atom,
            self.lambd,
            encut_atom,
            self.world,
            self.timer,
            self.Nalpha,
            self.cut_xcgrid,
            gd=self.gd,
            bas_exp_fit=self.bas_exp,
            is_mgga=(self.type == 'MGGA'),
        )
        self.paw_kernel.initialize(self.dens, self.atomdist, self.atom_partition, self.setups)
        self.paw_kernel.initialize_more_things(setups=self.setups)
        self.paw_kernel.interpolate_dn1 = self.interpolate_dn1
        self.atom_slices = None

    def setup_aalist(self):
        # Set up list of atoms to handle on this process
        if self.a_comm is not None:
            aa_range = self.par_cider.get_aa_range(self.comms['d'].rank)
            aalist = np.arange(aa_range[0], aa_range[1])
        else:
            aalist = []
        self.aalist = aalist
        return aalist

    def _setup_atom_slices(self):
        self.timer.start('ATOM SLICE SETUP')
        aalist = self.setup_aalist()
        self.atom_slices = {}
        for a in aalist:
            self.atom_slices[a] = AtomPASDWSlice.from_gd_and_setup(
                    self.gd,
                    self.spos_ac[a],
                    self.setups[a].pasdw_setup,
                    rmax=self.setups[a].pasdw_setup.rcut_func,
                    order='C',
                    sphere=True,
                    ovlp_fit=self.pasdw_ovlp_fit,
                    store_funcs=self.pasdw_store_funcs,
                )
        self.atom_slices_2 = {}
        for a in aalist:
            self.atom_slices_2[a] = AtomPASDWSlice.from_gd_and_setup(
                    self.gd,
                    self.spos_ac[a],
                    self.setups[a].paonly_setup,
                    rmax=self.setups[a].paonly_setup.rcut_func,
                    order='C',
                    sphere=True,
                    ovlp_fit=self.pasdw_ovlp_fit,
                    store_funcs=self.pasdw_store_funcs,
                )
        self.timer.stop()

    def _setup_extra_buffers(self):
        self.Freal_sag = LDict()
        self.dedtheta_sag = LDict()
        for s in range(self.nspin):
            self.Freal_sag[s] = LDict()
            self.dedtheta_sag[s] = LDict()
        self.Freal_sag.lock()
        self.dedtheta_sag.lock()
        for s in range(self.nspin):
            for a in self.alphas:
                self.Freal_sag[s][a] = self.gdfft.empty()
                self.dedtheta_sag[s][a] = self.gdfft.empty()
            self.Freal_sag[s].lock()
            self.dedtheta_sag[s].lock()

    def initialize_more_things(self):
        self.setups = self._hamiltonian.setups
        self.atomdist = self._hamiltonian.atomdist
        self.atom_partition = self.get_D_asp().partition

        natom = len(self.setups)
        rank_a = self.atom_partition.rank_a
        self.atoms = np.arange(natom)[rank_a==self.world.rank]

        encut_atom = self.encut
        Nalpha_atom = self.Nalpha
        while encut_atom < self.encut_atom_min:
            encut_atom *= self.lambd
            Nalpha_atom += 1

        if self.verbose:
            print('CIDER: density array size:',
                  self.gd.get_size_of_global_array())
            print('CIDER: zero-padded array size:', self.shape)

        cider_kernel_inp = self.cider_kernel \
            if (not hasattr(self, 'debug_kernel')) else self.debug_kernel

        self.initialize_paw_kernel(cider_kernel_inp, Nalpha_atom, encut_atom)

        self._setup_kgrid()
        self._setup_cd_xd_ranks()
        self._setup_6d_integral_buffer()
        self._setup_extra_buffers()

        self.par_cider.setup_atom_comm_data(
            self.atom_partition.rank_a, self.setups
        )

        if not (self.atom_partition.rank_a[1:] >= self.atom_partition.rank_a[:-1]).all():
            raise ValueError('Bad atom_partition!')

    def get_dH_asp(self, setups, D_asp, D_sabi, deltaV):
        dH_asp = self.paw_kernel.calculate_paw_feat_corrections(
            setups, D_asp, vc_sabi=D_sabi
        )
        for a in D_asp.keys():
            dH_asp[a] += deltaV[a]
        return dH_asp

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        if a is None:
            raise ValueError('Atom index a not provided.')
        if dEdD_sp is not None:
            dEdD_sp += self.dH_asp_tmp[a]
        return self.E_a_tmp[a]

    def _collect_paw_corrections(self):
        """
        For use in xc_tools, only after calculate_paw_correction
        is called. Collects potential using from_work
        """
        dH_asp_tmp = self.dH_asp_tmp
        E_a_tmp = self.E_a_tmp
        dH_asp_new = self.dens.setups.empty_atomic_matrix(
            self.nspin, self.atom_partition
        )
        E_a_new = self.atom_partition.arraydict(
            [(1,)] * len(self.dens.setups), float
        )
        for a, E in E_a_tmp.items():
            E_a_new[a][:] = E
            dH_asp_new[a][:] = dH_asp_tmp[a]
        dist = self.atomdist
        self.E_a_tmp = dist.from_work(E_a_new)
        self.dH_asp_tmp = dist.from_work(dH_asp_new)

    def initialize(self, density, hamiltonian, wfs):
        self.dens = density
        self._hamiltonian = hamiltonian
        self.timer = wfs.timer
        self.world = wfs.world
        self.get_alphas()
        self.setups = None
        self.gdfft = None
        self.pwfft = None
        self.rbuf_ag = None

    def get_paw_atom_contribs(self, n_sg, sigma_xg, ae=True):
        return self.paw_kernel.get_paw_atom_contribs(n_sg, sigma_xg, ae)

    def write_augfeat_to_rbuf(self, c_abi, pot=False):
        if self.atom_slices is None:
            self._setup_atom_slices()
        for b in self.alphas:
            self.rbuf_ag[b][:] = 0.0
        Nalpha_r = len(self.alphas)
        
        if pot:
            atom_slices = self.atom_slices_2
            atom_comm = self.par_cider.pa_comm
            nilist = [s.paonly_setup.ni for s in self.setups]
        else:
            atom_slices = self.atom_slices
            atom_comm = self.par_cider.ps_comm
            nilist = [s.pasdw_setup.ni for s in self.setups]

        coefs_abi = {}
        for a in self.aalist:
            coefs_abi[a] = np.empty((Nalpha_r, nilist[a]))
        atom_comm.send_atomic_coefs(c_abi, coefs_abi, alpha2atom=False)

        if Nalpha_r == 0:
            pass
        elif Nalpha_r == 1: # For large systems w/ atom par
            b = self.alphas[0]
            feat_g = self.gdfft.zeros(global_array=True)
            for a in self.aalist:
                atom_slice = atom_slices[a]
                funcs_gi = atom_slice.get_funcs()
                ni = funcs_gi.shape[1]
                coefs_bi = coefs_abi[a]
                if atom_slice.sinv_pf is not None:
                    coefs_bi = coefs_bi.dot(atom_slice.sinv_pf.T)
                fnlxc.pasdw_reduce_i(
                    coefs_bi[0],
                    funcs_gi,
                    feat_g.T,
                    atom_slice.indset,
                )
            self.gdfft.comm.sum(feat_g, root=0)
            self.rbuf_ag[b][:] += self.gdfft.distribute(feat_g)
        elif len(self.aalist) == len(self.setups): # For small systems w/o atom par
            for a in self.aalist:
                atom_slice = atom_slices[a]
                funcs_gi = atom_slice.get_funcs()
                ni = funcs_gi.shape[1]
                coefs_bi = coefs_abi[a]
                if atom_slice.sinv_pf is not None:
                    coefs_bi = coefs_bi.dot(atom_slice.sinv_pf.T)
                for ib, b in enumerate(self.alphas):
                    assert self.rbuf_ag[b].flags.c_contiguous
                    fnlxc.pasdw_reduce_i(
                        coefs_bi[ib],
                        funcs_gi,
                        self.rbuf_ag[b].T,
                        atom_slice.indset,
                    )
        else:
            raise RuntimeError('CIDER: Parallelization over atoms only supported for 1 alpha per core')

    def interpolate_rbuf_onto_atoms(self, pot=False):
        if self.atom_slices is None:
            self._setup_atom_slices()
        comm = self.world
        coefs_abi = {}
        Nalpha_r = len(self.alphas)

        if pot:
            atom_slices = self.atom_slices
            atom_comm = self.par_cider.ps_comm
            nilist = [s.pasdw_setup.ni for s in self.setups]
        else:
            atom_slices = self.atom_slices_2
            atom_comm = self.par_cider.pa_comm
            nilist = [s.paonly_setup.ni for s in self.setups]

        if Nalpha_r == 0:
            pass
        elif Nalpha_r == 1: # For large systems w/ atom par
            feat_g = self.gdfft.collect(self.rbuf_ag[self.alphas[0]], broadcast=True)
            for a in self.aalist:
                atom_slice = atom_slices[a]
                funcs_gi = atom_slice.get_funcs()
                ni = funcs_gi.shape[1]
                coefs_bi = np.zeros((1, ni))
                fnlxc.pasdw_reduce_g(
                    coefs_bi[0],
                    funcs_gi,
                    feat_g.T,
                    atom_slice.indset,
                )
                coefs_bi *= self.gd.dv
                if atom_slice.sinv_pf is not None:
                    coefs_bi = coefs_bi.dot(atom_slice.sinv_pf)
                coefs_abi[a] = coefs_bi
        elif len(self.aalist) == len(self.setups): # For small systems w/o atom par
            for a in self.aalist:
                atom_slice = atom_slices[a]
                funcs_gi = atom_slice.get_funcs()
                ni = funcs_gi.shape[1]
                coefs_bi = np.zeros((Nalpha_r, ni))
                for ib, b in enumerate(self.alphas):
                    fnlxc.pasdw_reduce_g(
                        coefs_bi[ib],
                        funcs_gi,
                        self.rbuf_ag[b].T,
                        atom_slice.indset,
                    )
                coefs_bi *= self.gd.dv
                if atom_slice.sinv_pf is not None:
                    coefs_bi = coefs_bi.dot(atom_slice.sinv_pf)
                coefs_abi[a] = coefs_bi
        else:
            raise RuntimeError('CIDER: Parallelization over atoms only supported for 1 alpha per core')

        c_abi = {}
        for a in self.atoms:
            c_abi[a] = np.empty((self.Nalpha, nilist[a]))
        # self.send_coefs_from_atoms()
        atom_comm.send_atomic_coefs(
            coefs_abi, c_abi, alpha2atom=True
        )
        return c_abi

    def construct_grad_terms(self, c_ag, c_abi, aug=False, stress=False):
        # aug specifies that these are the augmentation terms
        if self.atom_slices is None:
            self._setup_atom_slices()
        Nalpha_r = len(self.alphas)

        if aug:
            atom_slices = self.atom_slices_2
            atom_comm = self.par_cider.pa_comm
            nilist = [s.paonly_setup.ni for s in self.setups]
        else:
            atom_slices = self.atom_slices
            atom_comm = self.par_cider.ps_comm
            nilist = [s.pasdw_setup.ni for s in self.setups]

        coefs_abi = {}
        for a in self.aalist:
            coefs_abi[a] = np.empty((Nalpha_r, nilist[a]))
        atom_comm.send_atomic_coefs(c_abi, coefs_abi, alpha2atom=False)

        if stress:
            P = 0
            stress_vv = np.zeros((3,3))
        else:
            F_av = np.zeros((len(self.setups), 3))

        if Nalpha_r == 0:
            pass
        elif Nalpha_r == 1:
            b = self.alphas[0]
            feat_g = self.gdfft.collect(c_ag[self.alphas[0]], broadcast=True)
            for a in self.aalist:
                atom_slice = atom_slices[a]
                grads_vgi = atom_slice.get_grads()
                ni = grads_vgi.shape[-1]
                coefs_bi = coefs_abi[a]
                has_ofit = atom_slice.sinv_pf is not None
                if stress or has_ofit:
                    funcs_gi = atom_slice.get_funcs()
                if has_ofit:
                    coefs_bi = coefs_bi.dot(atom_slice.sinv_pf.T)
                    X_xii = atom_slice.get_ovlp_deriv(
                        funcs_gi, grads_vgi, stress=stress
                    )
                    ft_xbi = np.einsum('bi,...ji->...bj', coefs_bi, X_xii)
                    ft_xbg = np.einsum('...bi,gi->...bg', ft_xbi, funcs_gi)
                    ft_xbg *= self.gd.dv
                # NOTE account for dv here
                intb_vg = grads_vgi.dot(self.gd.dv * coefs_bi[0])
                if stress:
                    if aug:
                        tmp_i = np.zeros(ni)
                        fnlxc.pasdw_reduce_g(
                            tmp_i,
                            funcs_gi,
                            feat_g.T,
                            atom_slice.indset,
                        )
                        P += self.gd.dv * tmp_i.dot(coefs_bi[0])
                    for v in range(3):
                        dx_g = atom_slice.rad_g * atom_slice.rhat_gv[:,v]
                        fnlxc.pasdw_reduce_g(
                            stress_vv[v],
                            (dx_g * intb_vg).T,
                            feat_g.T,
                            atom_slice.indset,
                        )
                        if has_ofit:
                            fnlxc.pasdw_reduce_g(
                                stress_vv[v],
                                ft_xbg[v,:,0].T,
                                feat_g.T,
                                atom_slice.indset,
                            )
                else:
                    fnlxc.pasdw_reduce_g(
                        F_av[a],
                        intb_vg.T,
                        feat_g.T,
                        atom_slice.indset,
                    )
                    if has_ofit:
                        fnlxc.pasdw_reduce_g(
                            F_av[a],
                            ft_xbg[:,0].T,
                            feat_g.T,
                            atom_slice.indset,
                        )
        elif len(self.aalist) == len(self.setups): # For small systems w/o atom par
            for a in self.aalist:
                atom_slice = atom_slices[a]
                grads_vgi = atom_slice.get_grads()
                ni = grads_vgi.shape[-1]
                coefs_bi = coefs_abi[a]
                has_ofit = atom_slice.sinv_pf is not None
                if stress or has_ofit:
                    funcs_gi = atom_slice.get_funcs()
                if has_ofit:
                    coefs_bi = coefs_bi.dot(atom_slice.sinv_pf.T)
                    X_xii = atom_slice.get_ovlp_deriv(
                        funcs_gi, grads_vgi, stress=stress
                    )
                    ft_xbi = np.einsum('bi,...ji->...bj', coefs_bi, X_xii)
                    ft_xbg = np.einsum('...bi,gi->...bg', ft_xbi, funcs_gi)
                    ft_xbg *= self.gd.dv
                for ib, b in enumerate(self.alphas):
                    intb_vg = grads_vgi.dot(self.gd.dv * coefs_bi[ib])
                    if stress:
                        if aug:
                            tmp_i = np.zeros(ni)
                            fnlxc.pasdw_reduce_g(
                                tmp_i,
                                funcs_gi,
                                c_ag[b].T,
                                atom_slice.indset,
                            )
                            P += self.gd.dv * tmp_i.dot(coefs_bi[ib])
                        for v in range(3):
                            dx_g = atom_slice.rad_g * atom_slice.rhat_gv[:,v]
                            fnlxc.pasdw_reduce_g(
                                stress_vv[v],
                                (dx_g * intb_vg).T,
                                c_ag[b].T,
                                atom_slice.indset,
                            )
                            if has_ofit:
                                fnlxc.pasdw_reduce_g(
                                    stress_vv[v],
                                    ft_xbg[v,:,ib].T,
                                    c_ag[b].T,
                                    atom_slice.indset,
                                )
                    else:
                        fnlxc.pasdw_reduce_g(
                            F_av[a],
                            intb_vg.T,
                            c_ag[b].T,
                            atom_slice.indset,
                        )
                        if has_ofit:
                            fnlxc.pasdw_reduce_g(
                                F_av[a],
                                ft_xbg[:,ib].T,
                                c_ag[b].T,
                                atom_slice.indset,
                            )
                        #inds = tuple((atom_slice.indset-1).astype(int).tolist())
                        #print(intb_vg.shape, c_ag[b].shape, c_ag[b][inds].shape, atom_slice.indset.shape)
                        #F_av[a] += np.einsum('vg,g->v', intb_vg, c_ag[b][inds])
        else:
            raise RuntimeError('CIDER: Parallelization over atoms only supported for 1 alpha per core')

        if stress:
            for v in range(3):
                stress_vv[v,v] += P
            return 0.5 * (stress_vv + stress_vv.T)
        else:
            return F_av

    def add_forces(self, F_av):
        tmp_av = 0
        for s in range(self.nspin):
            tmp_av += self.construct_grad_terms(
                self.dedtheta_sag[s], self.c_sabi[s], aug=False, stress=False,
            )
            tmp_av += self.construct_grad_terms(
                self.Freal_sag[s], self.dedq_sabi[s], aug=True, stress=False,
            )
        if self.par_cider.comm is not None:
            self.par_cider.comm.sum(tmp_av, 0)
        if self.world.rank == 0:
            F_av[:] += tmp_av

    def calc_cider(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg,
                   tau_sg=None, dedtau_sg=None, compute_stress=False):
        feat = {}
        dfeat = {}
        p_iag = {}
        q_ag = {}
        dq_ag = {}

        self.timer.start('initialization')
        nspin = nt_sg.shape[0]
        nexp = self.nexp
        self.nspin = nspin
        if (self.setups is None) or (self.atom_slices is None):
            self.initialize_more_things()
            self.construct_cubic_splines()
        setups = self.setups
        D_asp = self.get_D_asp()
        if not (D_asp.partition.rank_a == self.atom_partition.rank_a).all():
            raise ValueError('rank_a mismatch')
        self.timer.stop()

        ascale, ascale_derivs = self.get_ascale_and_derivs(nt_sg, sigma_xg, tau_sg)
        cider_nt_sg = self.domain_world2cider(nt_sg)
        if cider_nt_sg is None:
            cider_nt_sg = {s:None for s in range(nspin)}

        self.timer.start('CIDER PAW')
        c_sabi, df_asbLg = self.paw_kernel.calculate_paw_feat_corrections(setups, D_asp)
        if len(c_sabi.keys()) == 0:
            c_sabi = {s: {} for s in range(nspin)}
        self.c_sabi = c_sabi
        self.timer.stop()

        self.timer.start('6d int')
        D_sabi = {}
        if compute_stress and nspin > 1:
            self.theta_sak_tmp = {s:{} for s in range(nspin)}
        for s in range(nspin):
            feat[s], dfeat[s], p_iag[s], q_ag[s], dq_ag[s], D_sabi[s] = \
                self.calculate_6d_integral_fwd(
                    cider_nt_sg[s], ascale[s], c_abi=c_sabi[s],
                )
            for a in self.alphas:
                self.Freal_sag[s][a][:] = self.rbuf_ag[a]
                if compute_stress and nspin > 1:
                    self.theta_sak_tmp[s][a] = self.theta_ak[a].copy()
        self.timer.stop()

        self.timer.start('Cider XC')
        feat_sig = np.stack([feat[s] for s in range(nspin)])
        del feat

        cond0 = feat_sig<0
        feat_sig[cond0] = 0.0

        if tau_sg is None: # GGA
            vfeat = self.cider_kernel.calculate(
                e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, feat_sig
            )
        else: # MGGA
            vfeat = self.cider_kernel.calculate(
                e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg,
                tau_sg, dedtau_sg, feat_sig
            )

        vfeat[cond0] = 0
        vexp = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        for s in range(nspin):
            vexp[s,:-1] = vfeat[s] * dfeat[s]
            vexp[s,-1] = 0.0
        self.timer.stop()

        self.timer.start('CIDER PAW BWD')
        v_sabi, deltaE, deltaV = self.paw_kernel.calculate_paw_feat_corrections(
            setups, D_asp, D_sabi=D_sabi, df_asbLg=df_asbLg
        )
        if len(v_sabi.keys()) == 0:
            v_sabi = {s: {} for s in range(nspin)}
        self.dedq_sabi = v_sabi
        self.timer.stop()

        self.timer.start('6d int')
        for s in range(nspin):
            if compute_stress and nspin > 1:
                for a in self.alphas:
                    self.theta_ak[a][:] = self.theta_sak_tmp[s][a]
            D_sabi[s] = self.calculate_6d_integral_bwd(
                cider_nt_sg[s], v_sg[s], vexp[s], vfeat[s],
                p_iag[s], q_ag[s], dq_ag[s],
                augv_abi=v_sabi[s], compute_stress=compute_stress)
            for a in self.alphas:
                self.dedtheta_sag[s][a][:] = self.rbuf_ag[a]
        self.timer.stop()

        self.add_scale_derivs(vexp, ascale_derivs, v_sg, dedsigma_xg, dedtau_sg)

        self.timer.start('dH_asp')
        dH_asp = self.get_dH_asp(setups, D_asp, D_sabi, deltaV)
        self.timer.stop()

        self.E_a_tmp = deltaE
        self.dH_asp_tmp = dH_asp


class CiderGGAPASDW(_CiderPASDW_MPRoutines, CiderGGA):
    """
    Class for evaluating nonlocal GGA functionals using the
    CIDER formalism. Initialize by calling from_mlfunc
    or from_joblib.
    """

    def __init__(self, cider_kernel, nexp, consts,
                 **kwargs):
        CiderGGA.__init__(
            self, cider_kernel, nexp, consts, **kwargs,
        )
        defaults_list = {
            'interpolate_dn1': False, # bool
            'encut_atom_min': 8000.0, # float
            'cut_xcgrid': False, # bool
            'pasdw_ovlp_fit': True,
            'pasdw_store_funcs': False,
        }
        settings = {}
        for k, v in defaults_list.items():
            settings[k] = kwargs.get(k)
            if settings[k] is None:
                settings[k] = defaults_list[k]
        self.__dict__.update(settings)
        self.k2_k = None
        self.setups = None

    def todict(self):
        d = super(CiderGGAPASDW, self).todict()
        d['_cider_type'] = 'CiderGGAPASDW'
        d['xc_params']['pasdw_store_funcs'] = self.pasdw_store_funcs
        d['xc_params']['pasdw_ovlp_fit'] = self.pasdw_ovlp_fit
        return d

    def add_forces(self, F_av):
        CiderGGA.add_forces(self, F_av)
        _CiderPASDW_MPRoutines.add_forces(self, F_av)

    def stress_tensor_contribution(self, n_sg):
        nspins = len(n_sg)
        stress_vv = super(CiderGGAPASDW, self).stress_tensor_contribution(n_sg)
        tmp_vv = np.zeros((3,3))
        for s in range(nspins):
            tmp_vv += self.construct_grad_terms(
                self.dedtheta_sag[s], self.c_sabi[s], aug=False, stress=True,
            )
            tmp_vv += self.construct_grad_terms(
                self.Freal_sag[s], self.dedq_sabi[s], aug=True, stress=True,
            )
        self.world.sum(tmp_vv)
        stress_vv[:] += tmp_vv
        self.gd.comm.broadcast(stress_vv, 0)
        return stress_vv

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac
        self.atom_slices = None

    @classmethod
    def from_mlfunc(cls, mlfunc, slx='GGA_X_PBE', slc='GGA_C_PBE',
                    Nalpha=None, encut=80, lambd=1.85, xmix=0.25,
                    debug=False, no_paw_atom_kernel=False,
                    **kwargs):

        if mlfunc.desc_version != 'd':
            raise ValueError("Only implemented for d version, found {}".format(mlfunc.desc_version))
        
        if debug:
            cider_kernel = CiderGGAHybridKernelPBEPot(
                mlfunc, xmix, slx, slc
            )
        else:
            cider_kernel = CiderGGAHybridKernel(
                mlfunc, xmix, slx, slc
            )

        consts = np.array([0.00, mlfunc.a0, mlfunc.fac_mul, mlfunc.amin])
        const_list = np.stack([
            0.5 * consts,
            1.0 * consts,
            2.0 * consts,
            consts * mlfunc.vvmul
        ])
        nexp = 4

        xc = cls(cider_kernel, nexp, const_list, Nalpha=Nalpha,
                 lambd=lambd, encut=encut, xmix=xmix, **kwargs)

        if no_paw_atom_kernel:
            xc.debug_kernel = CiderGGAHybridKernelPBEPot(
                mlfunc, xmix, slx, slc
            )

        return xc


class CiderMGGAPASDW(_CiderPASDW_MPRoutines, CiderMGGA):

    def __init__(self, cider_kernel, nexp, consts,
                 **kwargs):
        CiderMGGA.__init__(
            self, cider_kernel, nexp, consts, **kwargs,
        )
        defaults_list = {
            'interpolate_dn1': False, # bool
            'encut_atom_min': 8000.0, # float
            'cut_xcgrid': False, # bool
            'pasdw_ovlp_fit': True,
            'pasdw_store_funcs': False,
        }
        settings = {}
        for k, v in defaults_list.items():
            settings[k] = kwargs.get(k)
            if settings[k] is None:
                settings[k] = defaults_list[k]
        self.__dict__.update(settings)
        self.k2_k = None
        self.setups = None

    def todict(self):
        d = super(CiderMGGAPASDW, self).todict()
        d['_cider_type'] = 'CiderMGGAPASDW'
        d['xc_params']['pasdw_store_funcs'] = self.pasdw_store_funcs
        d['xc_params']['pasdw_ovlp_fit'] = self.pasdw_ovlp_fit
        return d

    def add_forces(self, F_av):
        MGGA.add_forces(self, F_av)
        _CiderPASDW_MPRoutines.add_forces(self, F_av)

    def stress_tensor_contribution(self, n_sg):
        nspins = len(n_sg)
        stress_vv = super(CiderMGGAPASDW, self).stress_tensor_contribution(n_sg)
        tmp_vv = np.zeros((3,3))
        for s in range(nspins):
            tmp_vv += self.construct_grad_terms(
                self.dedtheta_sag[s], self.c_sabi[s], aug=False, stress=True,
            )
            tmp_vv += self.construct_grad_terms(
                self.Freal_sag[s], self.dedq_sabi[s], aug=True, stress=True,
            )
        self.world.sum(tmp_vv)
        stress_vv[:] += tmp_vv
        self.gd.comm.broadcast(stress_vv, 0)
        return stress_vv

    def process_cider(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg):
        if self.fixed_ke:
            taut_sG = self._fixed_taut_sG
            if not self._taut_gradv_init:
                self._taut_gradv_init = True
                # ensure initialization for calculation potential
                self.wfs.calculate_kinetic_energy_density()
        else:
            taut_sG = self.wfs.calculate_kinetic_energy_density()

        if taut_sG is None:
            taut_sG = self.wfs.gd.zeros(len(nt_sg))

        taut_sg = np.empty_like(nt_sg)

        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)

        dedtaut_sg = np.zeros_like(nt_sg)
        self.calc_cider(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg,
                        taut_sg, dedtaut_sg)

        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        self.ekin = 0.0
        for s in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s], self.dedtaut_sG[s])
            self.ekin -= self.wfs.gd.integrate(
                self.dedtaut_sG[s] * (taut_sG[s] -
                                      self.tauct_G / self.wfs.nspins))

    def set_positions(self, spos_ac):
        MGGA.set_positions(self, spos_ac)
        self.spos_ac = spos_ac
        self.atom_slices = None

    def initialize(self, density, hamiltonian, wfs):
        MGGA.initialize(self, density, hamiltonian, wfs)
        self.dens = density
        self._hamiltonian = hamiltonian
        self.timer = wfs.timer
        self.world = wfs.world
        self.get_alphas()
        self.setups = None
        self.atom_slices = None
        self.gdfft = None
        self.pwfft = None
        self.rbuf_ag = None
        self.kbuf_ak = None
        self.theta_ak = None

    @classmethod
    def from_mlfunc(cls, mlfunc, slx='GGA_X_PBE', slc='GGA_C_PBE',
                    Nalpha=None, encut=300, lambd=1.8, xmix=0.25,
                    debug=False, no_paw_atom_kernel=False,
                    **kwargs):

        if mlfunc.desc_version != 'b':
            raise ValueError("Only implemented for d version, found {}".format(mlfunc.desc_version))

        if debug:
            raise NotImplementedError
        else:
            cider_kernel = CiderMGGAHybridKernel(
                mlfunc, xmix, slx, slc
            )

        consts = np.array([0.00, mlfunc.a0, mlfunc.fac_mul, mlfunc.amin])
        const_list = np.stack([
            0.5 * consts,
            1.0 * consts,
            2.0 * consts,
            consts * mlfunc.vvmul
        ])
        nexp = 4

        xc = cls(cider_kernel, nexp, const_list, Nalpha=Nalpha,
                 lambd=lambd, encut=encut, xmix=xmix, **kwargs)

        if no_paw_atom_kernel:
            raise NotImplementedError

        return xc


class SLCiderGGAHybridWrapper(CiderGGAHybridKernel):

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        super(SLCiderGGAHybridWrapper, self).calculate(
            e_g, n_sg, v_sg, sigma_xg, dedsigma_xg,
            np.zeros((n_sg.shape[0], 3,)+e_g.shape, dtype=e_g.dtype)
        )


class SLCiderMGGAHybridWrapper(CiderMGGAHybridKernel):

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg,
                  tau_sg, dedtau_sg):
        super(SLCiderMGGAHybridWrapper, self).calculate(
            e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg,
            np.zeros((n_sg.shape[0], 3,)+e_g.shape, dtype=e_g.dtype)
        )


class _SLCiderBase:

    def get_setup_name(self):
        return 'PBE'

    def todict(self):
        kparams = self.kernel.todict()
        return {
            'kernel_params': kparams,
        }

    def get_mlfunc_data(self):
        return yaml.dump(self.kernel.mlfunc.to_dict(),
                         Dumper=yaml.CDumper)

    @classmethod
    def from_joblib(cls, fname, **kwargs):
        mlfunc = joblib.load(fname)
        return cls.from_mlfunc(mlfunc, **kwargs)

    @staticmethod
    def from_mlfunc(mlfunc, xmix=0.25, slx='GGA_X_PBE', slc='GGA_C_PBE'):
        if mlfunc.desc_version == 'b':
            cider_kernel = SLCiderMGGAHybridWrapper(mlfunc, xmix, slx, slc)
            cls = SLCiderMGGA
        elif mlfunc.desc_version == 'd':
            cider_kernel = SLCiderGGAHybridWrapper(mlfunc, xmix, slx, slc)
            cls = SLCiderGGA
        else:
            raise ValueError("Only implemented for b and d version, found {}".format(mlfunc.desc_version))

        return cls(cider_kernel)


class SLCiderGGA(_SLCiderBase, DiffGGA):

    def todict(self):
        d = super(SLCiderGGA, self).todict()
        d['_cider_type'] = 'SLCiderGGA'
        return d


class SLCiderMGGA(_SLCiderBase, DiffMGGA):

    def todict(self):
        d = super(SLCiderMGGA, self).todict()
        d['_cider_type'] = 'SLCiderMGGA'
        return d
