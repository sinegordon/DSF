# -*- coding: utf-8 -*-
"""
@author: sinegordon@gmail.com
@author: sheafitz@udel.edu
"""

import MDAnalysis as ma
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter1d
import time
import pickle
import json
from numba import jit, prange
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--simultaneous_frames", type=int,
    default=1,
    help="Number of frames to load simultaneously",
)

args = parser.parse_args()

step = args.simultaneous_frames

print('it begins')
with open('./config.json', encoding='utf-8') as json_file:
    config = json.load(json_file)
bt = time.time()
# Full path to pdb file
PDB = config['pdb_file']
# Full path to trr file
TRR = config['trr_file']
print('u?')
u = ma.Universe(PDB, TRR, format="trr")  # , refresh_offsets=True)
print('u')
# et = time.time()
# print("Open timing is ", et-bt)
print("Open timing is ", time.time()-bt)
# Minumum and maximum wavevector in A^-1
kmin = config['kmin'] / (2.0 * np.pi)
kmax = config['kmax'] / (2.0 * np.pi)
# Step in reciprocal space in A^-1
kstep = config['kstep'] / (2.0 * np.pi)
kabsmas = np.linspace(kmin - 0.5 * kstep, kmax + 0.5 *
                      kstep, int((kmax - kmin + kstep) / kstep + 1))
box = u.dimensions
dkx = 2 * np.pi / box[0]
dky = 2 * np.pi / box[1]
frames_count = u.trajectory.n_frames

# Select indices of integestng atoms
select_string = config['select_string']
sel = u.select_atoms(select_string, periodic=True)
inds = sel.indices

# Count points in reciprocal space
k_count = config['k_count']


def random_k():
    testx = dkx * np.random.randint(-int(kmax / dkx), int(kmax / dkx))
    testy = dky * np.random.randint(-int(kmax / dky), int(kmax / dky))
    if kmin**2 < testx ** 2 + testy ** 2 < kmax**2:
        return(testx, testy)
    else:
        return(random_k())


kmas = np.zeros((k_count, 2), dtype='float32')
for i in range(len(kmas)):
    kmas[i, :] = random_k()
kmas = np.asfortranarray(kmas)
# kmas = np.reshape(list(zip(kx, ky)), (k_count, 2))
knorms1 = 1 / np.linalg.norm(kmas, axis=1)
kunits = np.float32(np.reshape(list(map(lambda i: np.dot(
    kmas[i], knorms1[i]), range(len(knorms1)))), (k_count, 2)))
kunits1 = np.zeros_like(kunits)
kunits1[:, 0] = kunits[:, 1]
kunits1[:, 1] = -kunits[:, 0]
# kunits1 = np.asfortranarray(kmas)
kunits1 = np.asfortranarray(kunits1)

# Return array [jl, jt] for every k in kmas for frame index = frame_ind


@jit(nopython=True, parallel=True)
def frame_processing_j(_v, _p):
    ret = np.zeros((len(_v), len(kunits), 2), dtype='complex64')
    for i in prange(len(_v)):
        v = np.asfortranarray(_v[i].T)
        p = np.asfortranarray(_p[i].T)

        exps = kmas @ p
        exps = np.exp(-1j*exps)

        ret[i, :, 0] = np.sum((kunits @ v) * exps, axis=1)
        ret[i, :, 1] = np.sum((kunits1 @ v) * exps, axis=1)

    return ret / np.sqrt(len(inds))


print("Frames count - ", frames_count)
jt = np.zeros((frames_count, k_count, 2), dtype='complex64')
# bt = time.time()
# @profile

_frame = u.trajectory[0]
_v = np.float32(_frame.velocities[inds])[:, 0:2]

v = np.asfortranarray(np.zeros((step, len(_v), 2), dtype='float32'))
p = np.asfortranarray(np.zeros_like(v))
print(frames_count % step, flush=True)


def main():
    for i in range(0, frames_count//step):
        print(i*step)
        for f in range(step):
            frame = u.trajectory[i+f]
            v[f] = np.float32(frame.velocities[inds])[:, 0:2]
            p[f] = np.float32(frame.positions[inds])[:, 0:2]
        jt[i:i+step, :, :] = frame_processing_j(v, p)
        print(i, (time.time()-bt), flush=True)
    for f in range(frames_count % step):
        frame = u.trajectory[f]
        v[f] = np.float32(frame.velocities[inds])[:, 0:2]
        p[f] = np.float32(frame.positions[inds])[:, 0:2]
    jt[-(frames_count % step):, :, :] = frame_processing_j(
        v[:frames_count % step], p[:frames_count % step])
    return


main()
# et = time.time()
# print("Solve timing is ", et-bt)

file_id = config['file_id']
with open(f'jt_{file_id}.pickle', 'wb') as f:
    pickle.dump(jt, f)
with open(f'kmas_{file_id}.pickle', 'wb') as f:
    pickle.dump(kmas, f)
with open(f'kabsmas_{file_id}.pickle', 'wb') as f:
    pickle.dump(kabsmas, f)
