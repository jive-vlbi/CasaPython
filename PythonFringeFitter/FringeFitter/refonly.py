import numpy as np, scipy, itertools
import glob, lsqrs, ffd, fringer, utils
import gencal_cli

def bl_fun(p, Ts, Fs, data):
    dpsi, dtau, dr = p
    sz = utils.flatsize(data)
    dv = np.zeros((2*sz,), dtype=np.float) 
    dvc = lsqrs.gen_bl_model(dpsi, dr, dtau, Ts, Fs) - data
    dv[0:sz] = np.ndarray.flatten(dvc.real)
    dv[sz:] = np.ndarray.flatten(dvc.imag)
    return dv

def all_cals_to_ref(msname, polind, pol_id, timeq, solint=300):
    swids = utils.get_spectral_windows(msname)
    sm = utils.get_antenna_map(msname, 'NAME')
    antennas2 = utils.get_antennas(msname, timeq)
    ism = bli.invert_map(sm)
    ref_antenna = 0 # FIXME
    #
    cal_table_names = []
    for swid in range(1):
        anffd = ffd.FFData.make_FFD(msname, antennas2, swid, polind, pol_id, timeq, solint=solint)
        anffd.fft(32) # leave it fixed.
        all_phis = []
        all_dels = []
        stl = []
        for i, s in enumerate(antennas2):
            stl.append(ism[s])
            if s == ref_antenna:
                all_phis.append(0)
                all_dels.append(0)
                continue
            d = anffd.data[ref_antenna, i]
            args=(anffd.tgrid0, anffd.fgrid0, d)
            p_est = anffd.get_fringe_peak(ref_antenna, i)
            phi, delay, rate = scipy.optimize.leastsq(bl_fun, p_est, args=args, full_output=0)[0]
            all_phis.append(-180*phi/np.pi)
            all_dels.append(-delay*1.0e9)
        antennas = ",".join(stl)
        ph_name = 'phases_{}.G'.format(swid)
        del_name = 'dels_{}.G'.format(swid)
        #gencal(infile='afile', vis=msname, caltable=ph_name, caltype='ph',
        #       spw='{}'.format(swid), antenna=antennas,   pol='L',
        #       parameter = all_phis)
        #gencal(infile='afile', vis=msname, caltable=del_name, caltype='sbd',
        #       spw='{}'.format(swid), antenna=antennas,   pol='L',
        #       parameter = all_dels)
        #cal_table_names.append(ph_name)
        #cal_table_names.append(del_name)
    return all_phis, all_dels
        


ref_phis, ref_dels = all_cals_to_ref(msname, polind, pol_id, timeq, solint=300)
