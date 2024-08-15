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

import os, subprocess


basedir = __file__
savedir = os.path.basename(os.path.join(basedir, '../functionals'))

os.makedirs(savedir, exist_ok=True)

os.chdir(savedir)
cmd = "wget 'https://zenodo.org/records/13323474/files/cider23_{}.zip?download=1' -O cider23_{}.zip"
unzip = "unzip cider23_{}.zip"
mv = "mv cider23_{}/* ."
rm = "rm -r cider23_{} cider23_{}.zip"
for fmt in ["functionals", "joblibs"]:
    subprocess.call(cmd.format(fmt, fmt), shell=True)
    subprocess.call(unzip.format(fmt), shell=True)
    subprocess.call(mv.format(fmt), shell=True)
    subprocess.call(rm.format(fmt, fmt), shell=True)

