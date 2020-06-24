# DSF
Python scripts to calculate current time correlations function of 2D systems.

To run, it is necessary that python version 3.6 or higher is installed on the computer. In addition, python-packages numpy, scipy and MDAnalysis are required. Starting in the following order
1. python current_calc_XY.py
2. python current_corr_XY.py

After that, the files kabsmas_ {fid}.pickle, wmas_ {fid}.pickle, jtkw_ {fid}.pickle and jlkw_ {fid}.pickle (where fid is the file_id parameter from the config.json) in which there will be arrays of wave vectors, frequencies, and correlators of transverse and longitudinal currents, respectively. Startup options are described in the config.json file. The following parameters must be set in it:

pdb_file - a pdb file describing the system;

trr_file - a file with a trajectory in gromacs format;

kmin - lower boundary of the wave vector in nanometers;

kmax - upper boundary of the wave vector in nanometers;

kstep - step along the wave vector in nanometers;

select_string - line of choice of atoms of interest in the MDAnalysis format;

file_id - startup identifier (any symbols);

window - the number of frames of the correlation window;

dt - time distance between adjacent frames in picoseconds;

These scripts were used in the following manuscripts:
- Molecular Picture of the Transient Nature of Lipid Rafts / Dima Bolmatov*, Dmytro Soloviov, Mikhail Zhernenkov, Dmitry Zav’yalov, Eugene Mamontov, Alexey Suvorov, Yong Q. Cai, and John Katsaras // Langmuir, 2020, 36, 18, 4887–4896. 
https://doi.org/10.1021/acs.langmuir.0c00125
- Anomalous Nanoscale Optoacoustic Phonon Mixing in Nematic Mesogens / D. Bolmatov, D. Soloviov, D. V. Zavyalov, L. Sharpnack, D. M. Agra-Kooijman, S. Kumar, Jiawei Zhang, M. Liu, J. Katsaras // Journal of Physical Chemistry Letters. - 2018. - Vol. 9, No. 10. - С. P. 2546-2553. https://pubs.acs.org/doi/full/10.1021/acs.jpclett.8b00926
- Crossover from picosecond collective to single particle dynamics defines the mechanism of lateral lipid diffusion / D. Bolmatov, Y. Q. Cai, D. V. Zavyalov, M. Zhernenkov // Biochimica et Biophysica Acta (BBA) – Biomembranes. - 2018. - Vol. 1860, Issue 11. - С. P. 2446-2455. – URL : https://doi.org/10.1016/j.bbamem.2018.07.004.
- Functional lipid pairs as building blocks of phase-separated membranes / D. Soloviov, Y. Q. Cai, D. Bolmatov, A. Suvorov, K. Zhernenkov, D. V. Zavyalov, A. Bosak, H. Uchiyama, M. Zhernenkov // Proceedings of the National Academy of Sciences of the United States of America. - 2020. - Vol. 117, Issue 9. – P. 4749-4757.
https://www.pnas.org/content/117/9/4749
