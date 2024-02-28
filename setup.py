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

import os, sys
from setuptools import find_packages
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.core import Extension as NPExtension
from numpy.distutils.core import setup

fext_dir = 'ciderpress/lib/gpaw_utils_src'
fsources = ['gpaw_utils.f90', 'fast_sph_harm.f90']
fext = NPExtension(
    name='ciderpress.dft.futil',
    sources=[os.path.join(fext_dir, fsrc) for fsrc in fsources],
    f2py_options=['--quiet'],
)

class CMakeBuildExt(build_ext):
    def run(self):
        super(CMakeBuildExt, self).run()
        self.build_cmake(None)

    def build_cmake(self, extension):
        self.announce('Configuring extensions', level=3)
        src_dir = os.path.abspath(os.path.join(__file__, '..', 'ciderpress', 'lib'))
        cmd = ['cmake', f'-S{src_dir}', f'-B{self.build_temp}',
               '-DCMAKE_PREFIX_PATH={}'.format(sys.base_prefix),
               '-DBLA_VENDOR=Intel10_64lp_seq', '-DCMAKE_BUILD_TYPE=Release']
        configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')
        if configure_args:
            cmd.extend(configure_args.split(' '))
        self.spawn(cmd)

        self.announce('Building binaries', level=3)
        cmd = ['cmake', '--build', self.build_temp, '-j2']
        build_args = os.getenv('CMAKE_BUILD_ARGS')
        if build_args:
            cmd.extend(build_args.split(' '))
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)

    def get_ext_filename(self, ext_name):
        if 'ciderpress.lib' in ext_name:
            ext_path = ext_name.split('.')
            filename = build_ext.get_ext_filename(self, ext_name)
            name, ext_suffix = os.path.splitext(filename)
            return os.path.join(*ext_path) + ext_suffix
        else:
            return super(CMakeBuildExt, self).get_ext_filename(ext_name)

from numpy.distutils.command.build import build
build.sub_commands = ([c for c in build.sub_commands if c[0] == 'build_ext'] +
                      [c for c in build.sub_commands if c[0] != 'build_ext'])

# TODO: need to add gpaw>=22.8.1b1 to reqs at some point
with open('requirements.txt', 'r') as f:
    requirements = [l.strip() for l in f.readlines()]

description = """CiderPress is a package for running DFT calculations 
with CIDER functionals in the GPAW and PySCF codes."""

setup(
    name="ciderpress",
    description=description,
    version="0.0.10",
    packages=find_packages(exclude=['*test*', '*examples*']),
    ext_modules=[fext],
    cmdclass={'build_ext': CMakeBuildExt},
    setup_requires=['numpy'],
    include_package_data=True,  # include everything in source control
    install_requires=requirements,
    package_data={'': ['*.so', '*.dylib', '*.dll', '*.dat']},
)

