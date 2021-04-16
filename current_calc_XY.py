# -*- coding: utf-8 -*-
"""
@author: sinegordon@gmail.com
"""

import MDAnalysis as ma
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import time
import pickle
import json

with open('./config.json', encoding='utf-8') as json_file:
    config = json.load(json_file)

bt = time.time()
# Full path to pdb file
PDB = config['pdb_file']
# Full path to trr file
TRR = config['trr_file']
u = ma.Universe(PDB, TRR, format="trr")
et = time.time()
print("Open timing is ", et-bt)
# Minumum and maximum wavevector in A^-1
kmin = config['kmin'] / (2.0 * np.pi)
kmax = config['kmax'] / (2.0 * np.pi)
# Step in reciprocal space in A^-1
kstep = config['kstep'] / (2.0 * np.pi)
kabsmas = np.linspace(kmin - 0.5 * kstep, kmax + 0.5 * kstep, int((kmax - kmin + kstep) / kstep + 1))
box = u.dimensions
dkx = 2 * np.pi / box[0]
dky = 2 * np.pi / box[1]
frames_count = u.trajectory.n_frames

# Select indices of integestng atoms
select_string = config['select_string']
sel = u.select_atoms(select_string, periodic = True)
inds = sel.indices

# Count points in reciprocal space
k_count = config['k_count']
kx = dkx * np.random.random_integers(-int(kmax / dkx), int(kmax / dkx), size = k_count)
ky = dky * np.random.random_integers(-int(kmax / dky), int(kmax / dky), size = k_count)
kmas = np.reshape(list(zip(kx, ky)), (k_count, 2))
knorms1 = 1 / np.linalg.norm(kmas, axis=1)
kunits = np.reshape(list(map(lambda i: np.dot(kmas[i], knorms1[i]), range(len(knorms1)))), (k_count, 2))
kunits1 = np.zeros_like(kunits)
kunits1[:, 0] =  kunits[:, 1]
kunits1[:, 1] = -kunits[:, 0]
    
# Return array [jx, jy] for every k in kmas for frame index = frame_ind
def frame_processing_j(frame_ind):
    bt = time.time()
    print("Processing frame # ", frame_ind)
    frame = u.trajectory[frame_ind]
    v = frame.velocities[inds]
    p = frame.positions[inds]
    prods = np.inner(kmas, p[:, 0:2])
    exps = np.exp(-1j*prods)
    qvl = np.inner(kunits, v[:, 0:2])
    qvt = np.inner(kunits1, v[:, 0:2])
    sprodsl = qvl * exps
    jl = np.sum(sprodsl, axis = 1)
    sprodst = qvt * exps
    jt = np.sum(sprodst, axis = 1)
    ret = np.zeros_like(kunits, dtype='complex')
    ret[:,0] = jl
    ret[:,1] = jt
    et = time.time()
    print("Frame #", frame_ind, " calculated for ", et-bt, "s.")
    return ret / np.sqrt(len(inds))

print("Frames count - ", frames_count)
jt = np.zeros((frames_count, k_count, 2), dtype = 'complex')    
bt = time.time()   
for i in range(frames_count):
    jt[i, :, :] = frame_processing_j(i)
et = time.time()
print("Solve timing is ", et-bt)

file_id = config['file_id']
with open(f'jt_{file_id}.pickle', 'wb') as f:
    pickle.dump(jt, f)
with open(f'kmas_{file_id}.pickle', 'wb') as f:
    pickle.dump(kmas, f)
with open(f'kabsmas_{file_id}.pickle', 'wb') as f:
    pickle.dump(kabsmas, f)
