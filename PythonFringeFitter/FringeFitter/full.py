## get_bl_data is effectively obsolete FFDData::make_FFD.
## sane time will probably come in handy, but I don't seem to be using it ATM.

def sane_time(t):
    # qa.time returns a list of one string.
    # no idea why.
    return qa.time(qa.quantity(t, 'd'), form='ymd')[0]

def get_bl_data(msname, antennas, ref_antenna, swid, pol_id, polind, timeq, startrow, fftpad=8):
    ref_s_ind = antennas.index(ref_antenna)
    n_antennas = len(antennas)
    e_baselines = bli.triangle_l(range(n_antennas))
    baselines = bli.triangle_l(antennas)
    ffd = get_some_data(msname, antenna_list, ant2, swid, pol_id, polind, timeq, startrow, nrow=40)
    first = True
    for (s0, s1), (i, j) in zip(baselines, e_baselines):
        print "(s0, s1), (i, j)", (s0, s1), (i, j)
        ffd = get_some_data(msname, s0, s1, swid, pol_id, polind, timeq, startrow=startrow)
        if first:
            shape = ffd.fgrid.shape
            data = np.zeros((n, n) + shape, np.complex)
            weights = np.ones((n, n) + shape, np.float)
            first = False
        data[i,j, :, :] = ffd.data
    Ts = ffd.tgrid0
    Fs = ffd.fgrid0
    params = []
    for p in antennas:
        if p==ref_antenna:
            params.extend([0.0, 0.0, 0.0])
        else:
            # find the index for ref_antenna-to-p baseline
            if (ref_antenna, p) in baselines:
                ind = baselines.index((ref_antenna, p))
                sgn = 1
            elif (p, ref_antenna) in baselines:
                ind = baselines.index((p, ref_antenna))
                sgn = -1
            else:
                raise ValueError, "No baseline {}--{}".format(ref_antenna, p)
            i,j = e_baselines[ind]
            psi, delay, rate = fringer.fringe_plot_data(data[i,j], fftpad, 0.5, 'none', ffd)
            params.extend([sgn*psi, sgn*rate, sgn*delay])
    return params, data, ffd, weights

if __name__ == '__main__':
    # f.ex.:
    msname = "EY015D.ms"
    swid = 0 # "spectral window" = what we normally call a subband.
    pol_id, polind = 0, 0 
    timeq = 'TIME > MJD("2012-10-24-21:30:00") and TIME < MJD("2012-10-24-21:31:20")'
    ant1, ant2  = 0, 1
    ffd = full.get_some_data(msname, 0, 6, swid, pol_id, polind, timeq)
