"""
@author: sinegordon@gmail.com
@author: sheafitz@udel.edu
"""

import numpy as np
import time
import pickle
from scipy.fftpack import rfft, irfft, fftfreq
import json
from numba import jit, prange

with open('config.json', encoding='utf-8') as json_file:
    config = json.load(json_file)
file_id = config['file_id']
with open(f'jt_{file_id}.pickle', 'rb') as f:
    jt = pickle.load(f)
with open(f'kmas_{file_id}.pickle', 'rb') as f:
    kmas = pickle.load(f)
with open(f'kabsmas_{file_id}.pickle', 'rb') as f:
    kabsmas = pickle.load(f)
# Correlation window size
window = config['window']
jt_dim = np.shape(jt)
print(jt_dim)
frame_count = jt_dim[0]
# Delta time between dumps frame (in config in ps)
dt = config['dt'] * 10**-12
to_meV = 1.055/1.6*10**-12
# Array of average jx-correlator for every k in kmas
jlf = np.zeros((window, jt_dim[1]), dtype='complex')
# Array of average jx-correlator for every kabs in kabsmas
jlkt = np.zeros((window, len(kabsmas)-1), dtype='complex')
# Array of fourier abs of average jx-correlator for every kabs in kabsmas
jlkw = np.zeros((window, len(kabsmas)-1))
# Array of average jy-correlator for every k in kmas
jtf = np.zeros((window, jt_dim[1]), dtype='complex')
# Array of average jy-correlator for every kabs in kabsmas
jtkt = np.zeros((window, len(kabsmas)-1), dtype='complex')
# Array of fourier abs of average jy-correlator for every kabs in kabsmas
jtkw = np.zeros((window, len(kabsmas)-1))


@jit(nopython=True, parallel=True)
def multiplyer(mat, vec):
    out = np.zeros_like(mat)
    for i in prange(len(mat)):
        out[i, :] = mat[i, :] * np.conj(vec[:])
    return(out)

# Calculate j-correlator for every k in kmas


@jit(nopython=True, parallel=True)
def solver(jlf, jtf):
    for frame_index in prange(frame_count - window):
        if frame_index % 100 == 0:
            print("Solve ", (frame_index)/(frame_count - window)*100, " %")
        # jlf[:, :] += (jt[frame_index:frame_index+window, :, 0])*np.conj(jt[frame_index, :, 0])
        # jtf[:, :] += (jt[frame_index:frame_index+window, :, 1])*np.conj(jt[frame_index, :, 1])
        jlf[:, :] += multiplyer(jt[frame_index:frame_index +
                                window, :, 0], jt[frame_index, :, 0])
        jtf[:, :] += multiplyer(jt[frame_index:frame_index +
                                window, :, 1], jt[frame_index, :, 1])
    jlf /= frame_count - window
    jtf /= frame_count - window
    return(jlf, jtf)


jlf, jtf = solver(jlf, jtf)
# Calculate j-correlator for every kabs in kabsmas
for k in range(1, len(kabsmas)):
    inds = []
    for i in range(len(kmas)):
        kabs = np.linalg.norm(kmas[i])
        if kabs <= kabsmas[k] and kabs > kabsmas[k-1]:
            inds.append(i)
    print(len(inds))
    jlkt[:, k-1] = np.mean(jlf[:, inds], axis=1)
    jtkt[:, k-1] = np.mean(jtf[:, inds], axis=1)

# Calculate jlkw and jtkw for every kabs
for k in range(len(kabsmas)-1):
    jlkw[:, k] = np.abs(np.fft.fft(jlkt[:, k]))
    jtkw[:, k] = np.abs(np.fft.fft(jtkt[:, k]))
wmas = np.fft.fftfreq(window, dt) * to_meV
print("Solve done!")

with open(f'wmas_{file_id}.pickle', 'wb') as f:
    pickle.dump(wmas, f)
with open(f'jlkw_{file_id}.pickle', 'wb') as f:
    pickle.dump(jlkw, f)
with open(f'jtkw_{file_id}.pickle', 'wb') as f:
    pickle.dump(jtkw, f)
