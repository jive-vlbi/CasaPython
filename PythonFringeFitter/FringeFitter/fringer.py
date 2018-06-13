from taskinit import casalog
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np, scipy
from numpy import fft
import math, sys, logging, inspect, itertools
import ffd, lsqrs, param, utils, bli

def index_of_max(a):
    imax = np.argmax(a, axis=None)
    return np.unravel_index(imax, a.shape)

def get_tb(taskname):
    # a record on the stack is (frame object, filename, ...)
    for rec in inspect.stack():
        if rec[1].find('ipython console')>-1:
            g = rec[0].f_globals
            break
    else:
        raise RuntimeError("Can't find the ipython console level")
    if taskname != None:
        g['__last_task']=taskname
        g['taskname']=taskname
    return g['tb']

def fringe_plot_data(data, pad=8, keyhole=None, plot=None, ffd=None, title="", peak=None):
    if keyhole==None:
        keyhole = pad
    padded_shape = tuple(map(lambda n: pad*n, data.shape))
    b = fft.fft2(data, s=padded_shape)
    c = fft.fftshift(b) 
    ni, nj = c.shape
    # casalog.post("c.shape {}".format(c.shape), "DEBUG")
    if ffd==None:
        yaxis0, xaxis0 = np.meshgrid(
            np.arange(-nj/2, nj/2),
            np.arange(-ni/2, ni/2),
            indexing='ij')
    else: 
        # as set up, x is frequency, y is time.
        (nt,) = ffd.times.shape
        (nf,) = ffd.freqs.shape
        bw = nf*ffd.df
        T = nt*ffd.dt
        rs = np.linspace(-nt/(2*T), nt/(2*T), nt*pad, endpoint=False)
        taus = np.linspace(-nf/(2*bw), nf/(2*bw), nf*pad, endpoint=False)
        # x axis of fft space is delay, y axis is fringe rate.
        xaxis0, yaxis0 = np.meshgrid(taus, rs, indexing='ij')
    scutout = map(lambda n:
                  slice(max(0, int(pad*(n)/2-keyhole*n)),
                        min(n*pad, int(pad*(n)/2+keyhole*n+1))),
                  data.shape)
    ind = index_of_max(abs(c))
    fringe_rate = yaxis0[ind]
    delay = xaxis0[ind]
    xind, yind = ind
    if peak==None:
        xpeak = xaxis0[ind]
        ypeak = yaxis0[ind]
    else:
        xpeak, ypeak = peak
    psi = np.angle(np.sum(c[xind-pad/2:xind+pad/2, yind-pad/2:yind+pad/2]))
    # casalog.post("Max value {}; Fringe rate: {}; Delay: {}".format(c[ind], fringe_rate, delay), "DEBUG")
    # casalog.post("Index {} shape {}".format(ind , scutout), "DEBUG")
    # casalog.post("Fringe rate grid size {}".format(yaxis0[0,1]-yaxis0[0,0]), "DEBUG")
    # casalog.post("Delay grid size {}".format(xaxis0[1,0]-xaxis0[0,0]), "DEBUG")
    if plot=='3d':
        fig = plt.figure()
        if title != '':  plt.title(title, y=1.2)
        # casalog.post("Actual data shape: {}".format(d.shape), "DEBUG")
        # casalog.post("xaxis0.shape {}".format(xaxis0.shape), "DEBUG")
        # casalog.post("data.shape {} ""c.shape {} ""scutout {}".format(data.shape, c.shape, scutout), "DEBUG")
        ax = plt.axes(projection='3d')
        d = np.abs(c[scutout]), 
        ax.plot_surface(xaxis0[scutout], yaxis0[scutout],
                        d, cmap=plt.cm.jet, rstride=1, cstride=1)
        ax.set_xlabel("delay (s)")
        ax.set_ylabel("fringe-rate (rad/s)")
        plt.show()
    elif plot=='2d':
        fig = plt.figure()
        if title != '':
            plt.title(title, y=1.2)
        cutout = map(lambda n:
                     [int(-keyhole*n), int(+keyhole*n)],
                     data.shape)
        fcutout = cutout[0] + cutout[1] # these are lists! this is appending!
        xmin = xaxis0[scutout][0,0]
        xmax = xaxis0[scutout][-1,0]
        ymin = yaxis0[0, 0]
        ymax = yaxis0[0, -1]
        extent = [xmin, xmax, ymin, ymax]
        casalog.post("Extent".format(extent), "DEBUG")
        plt.imshow(d.T,
                   interpolation='nearest',
                   #extent=fcutout,
                   extent=extent,
                   aspect='auto',
                   origin='lower',
                   vmin=d.min(), vmax=d.max()
                   )
        plt.colorbar()
        plt.show()
    elif plot=='1d':
        fig = plt.figure()
        plt.subplot(4, 1, 1)
        xslice = slice(nf*pad/2 - nlumps*pad, nf*pad/2 + nlumps*pad)
        yslice = slice(nt*pad/2 - nlumps*pad, nt*pad/2 + nlumps*pad)
        plt.plot(xaxis0[xslice, yind], np.abs(c[xslice, yind])) 
        plt.ylabel('Fringe height')
        plt.xlabel('Group delay (s)')
        plt.axvline(x=xpeak, linestyle='--', color='r')
        # 
        plt.subplot(4, 1, 2)
        plt.plot(xaxis0[xslice, yind], np.angle(c[xslice, yind]), 'r')
        plt.ylim(-np.pi, np.pi)
        plt.ylabel('Fringe phase')
        plt.xlabel('Group delay (s)')
        # 
        plt.subplot(4, 1, 3)
        plt.plot(yaxis0[xind, yslice], np.abs(c[xind, yslice]))
        plt.ylabel('Fringe height')
        plt.xlabel('Delay rate (s/s)')
        plt.axvline(x=ypeak, linestyle='--', color='r')
        # 
        plt.subplot(4, 1, 4)
        plt.plot(yaxis0[xind, yslice], np.angle(c[xind, yslice]), 'r')
        plt.ylim(-np.pi, np.pi)
        plt.ylabel('Fringe phase')
        plt.xlabel('Group delay (s)')
        #
        if title != '':
            plt.title(title, y=4.75)
        plt.show()
    else:
        pass
    return (psi, delay, fringe_rate)




def ms_basics(ms_name):
    d = {}
    tb.open(ms_name + '::DATA_DESCRIPTION')
    for k in ['SPECTRAL_WINDOW_ID', 'POLARIZATION_ID']:
        d[k] = tb.getcol(k)
    tb.open(ms_name + '::SPECTRAL_WINDOW')
    d['CHAN_FREQ'] = tb.getcol('CHAN_FREQ')
    tb.open(ms_name + '::ANTENNA')
    d['ANTENNA_NAME'] = list(tb.getcol('NAME'))
    tb.open(ms_name + '::FIELD')
    d['SOURCE_NAME'] = list(tb.getcol('NAME'))
    return d

def get_big_chan(tb, ant1=0, ant2=1, pol=0):
    ds = [tb.query("ANTENNA1={} and ANTENNA2={} and DATA_DESC_ID={}".
                   format(ant1, ant2, i))
          .getcol('DATA', nrow=40)[pol]
          for i in range(8)]
    all_d = np.concatenate(ds, axis=0)
    return all_d

##
## 2015-05-06: Stacking the baselines
##

def get_stackable_baseline(tb, q, ant1, ant2, pol=0, nrow=40, startrow=0):
    if ant1 < ant2:
        sgn = 1
    else:
        sgn = -1
        (ant1, ant2) = (ant2, ant1)
    try:
        t2 = tb.query(q.format(ant1, ant2))
        data0 = unitize(t2.getcol('DATA', nrow=nrow, startrow=startrow)[pol])
        data = data0 if sgn == 1 else 1/data0
    except RuntimeError, e:
        raise RuntimeError, "Fail with {} {}".format(ant1, ant2)
    return data

def sum_stack2(tb, q, ref, k, pol=0, nrow=40, startrow=0):
    ants = set(tb.getcol('ANTENNA1'))
    ant_triples = [(ref, i, k) for i in ants if i!=ref and i!=k]
    l = []
    for (i, j, k) in ant_triples:
        try:
            bl1 = get_stackable_baseline(tb, q, i, j, pol, nrow, startrow)
            bl2 = get_stackable_baseline(tb, q, j, k, pol, nrow, startrow)
            l.append(bl1 * bl2)
        except RuntimeError, e:
            print >>sys.stderr, "Fail with {}, {}, {}".format(i,j,k)
            continue
    stack = sum(l)
    return stack

def sum_stack3(tb, q, ref, k, pol=0, nrow=40, startrow=0):
    ants = set(tb.getcol('ANTENNA1'))
    ant_quads = [(ref, i, j, k)
                 for i in ants if i!=ref and i!=k
                 for j in ants if j!=ref and j!=k and j!=i]
    res = []
    for (i, j, k, l) in ant_quads:
        try:
            bl1 = get_stackable_baseline(tb, q, i, j, pol, nrow, startrow)
            bl2 = get_stackable_baseline(tb, q, j, k, pol, nrow, startrow)
            bl3 = get_stackable_baseline(tb, q, k, l, pol, nrow, startrow)
            res.append(bl1 * bl2 * bl3)
        except RuntimeError, e:
            print >>sys.stderr, "Fail with {}".format((i,j,k,l))
            continue
    stack = sum(res)
    return stack
    
def rms_window(d, maxind, shape):
    ni, nj = d.shape
    maxi, maxj = maxind
    wi, wj = shape
    count = 0
    total = 0.0
    for i in range(ni):
        for j in range(nj):
            if ((maxi - wi) <= i and i <= (maxi + wi) and
                (maxj - wj) <= j and j <= (maxj + wj)):
                continue
            else:
                count += 1
                total += abs(d[i,j])**2
    return math.sqrt(total/count)

def centre_fft(d):
    return fft.fftshift(fft.fft2(d))

def unitize(d):
    return d/np.absolute(d)

def snr(f, hole_shape=(3,3)):
    absf = abs(f)
    ind = index_of_max(absf)
    peak = absf[ind]
    rms = rms_window(absf, ind, hole_shape)
    return peak/rms


def fit_fringe_lm(msname, antennas2, ref_antenna, swid, polind, pol_id, timeq,
                  datacol="DATA", solint=None, threshold=0.0, snr_threshold=0.0,
                  threshold_method=None, ref_weights=True, pad=8):
    anffd = ffd.FFData.make_FFD(msname, antennas2, swid, polind, pol_id, timeq,
                                solint=solint, datacol=datacol)
    return fit_fringe_ffd(anffd, ref_antenna, antennas2, pad=pad,
                          threshold_method=threshold_method,
                          threshold=threshold, snr_threshold=snr_threshold)

def fit_fringe_ffd(anffd, ref_antenna, antennas2, pad=8,
                   threshold_method=None, threshold=0.0, snr_threshold=0.0,
                   snr_threshold_method=None):
    anffd.fft(pad=pad)
    (nf,) = anffd.freqs.shape
    n_antennas = len(antennas2)
    F_midpoint = anffd.fgrid0[nf//2][0]
    # get_params uses p-antennas
    params2 = anffd.get_params(ref_antenna=ref_antenna)
    anffd.params2 = params2
    e_ref_antenna = anffd.get_antenna_index(ref_antenna) # FIXME: really?
    # everything else uses e-antennas
    ref_params = param.get_antenna_parameters(params2, e_ref_antenna)
    anffd.report_fringes(e_ref_antenna)
    anffd.report_snrs(e_ref_antenna)
    ## FIXME: I don't know why we're doing this here, or at all, but
    ## SNRs need the other sensible kind of weights.
    anffd.make_weighted_weights()
    if threshold_method == 'raw':
        e_antennas_to_remove0 = anffd.get_e_antennas_below_threshold(e_ref_antenna, threshold)
    elif threshold_method == 'snr':
        e_antennas_to_remove0 = anffd.report_snrs(e_ref_antenna, snr_threshold)
    else:
        e_antennas_to_remove0 = []
    e_antennas_to_remove = sorted(e_antennas_to_remove0)
    # Here we need to remove also antennas from anffd.data and anffd.weights
    p_antennas_to_remove = [anffd.get_p_antenna_index(e) for e in e_antennas_to_remove]
    p_antennas_to_keep = [anffd.get_p_antenna_index(e) for e in range(n_antennas)
                          if e not in  e_antennas_to_remove]
    if len(e_antennas_to_remove) > 0:
        casalog.post("{} antennas have fringes below thresholds and will be removed".format(
            len(e_antennas_to_remove), "INFO"))
        casalog.post("Masking antennas " + ", ".join("{}".format(p) for p in p_antennas_to_remove),
                     "INFO")
    data, weights = trim_data(anffd, e_antennas_to_remove)
    n_actual_e_antennas = len(antennas2) - len(e_antennas_to_remove)
    # # FIXME: can this be folded into new reduced_antenna_map paradigm?
    params3, removed_params = param.remove_antennas(params2, e_antennas_to_remove)
    new_e_ref_antenna = e_ref_antenna - sum([int(a < e_ref_antenna) for a in e_antennas_to_remove])
    params4, ref_params = param.remove_ref(params3, new_e_ref_antenna)
    casalog.post("params4: {}".format(params4))
    args=(anffd.tgrid0, anffd.fgrid0, n_actual_e_antennas, data, weights, new_e_ref_antenna)
    casalog.post("Starting least-squares solver", "INFO")

    # Caution! New!
    eps = 1e-10
    sweight = np.sum(weights*np.logical_not(data.mask))
    ftol = eps*sweight
    casalog.post(
        "Sum of weights {}; error per phasor {}; ftol={}."
        "".format(sweight, eps, ftol))
    res_t = scipy.optimize.leastsq(lsqrs.vector_s3_test, params4,
                                   full_output=1, args=args,
                                   maxfev=20,
                                   col_deriv=True,
                                   Dfun=lsqrs.matrix_j_s3,
                                   ftol=ftol)
    casalog.post("Least-squares solver finished", "INFO")
    sol_out = list(res_t[0])
    casalog.post("Least-squares solver finished after {} iterations".format(res_t[2]['nfev']), "DEBUG")
    # 1 t/m 4 good; others bad.
    casalog.post("Solver status {}, message: {}".format(res_t[4], res_t[3]), "DEBUG")
    # casalog.post("sol_out".format(sol_out), "DEBUG")
    sigma_p = param.get_phases(sol_out) # FIXME
    pout2d0 = param.restore_ref(sol_out, ref_params, new_e_ref_antenna)
    dels0 = param.get_delays(pout2d0)
    casalog.post("dels0 {}".format(dels0))
    flags, pout2d = param.restore_antennas(pout2d0, removed_params, e_antennas_to_remove)
    #
    dels = param.get_delays(pout2d)
    r0s = param.get_rates(pout2d)
    rs = r0s/anffd.ref_freq
    phs0 = param.get_phases(pout2d)
    DT = anffd.tgrid0[0, -1]
    # phs = +np.array(phs0) + np.pi*DT*np.array(r0s)
    # We do NOT correct anything here; leave that to the writer.
    phs = np.array(phs0) 
    # + 2*np.pi*F_midpoint*np.array(dels)
    casalog.post("dels {}".format(dels))
    return flags, dels, phs, rs, sigma_p

def trim_data(anffd, e_antennas_to_remove):
    n_antennas = anffd.data.shape[0]
    n_actual_e_antennas = n_antennas - len(e_antennas_to_remove)
    new_shape = (n_actual_e_antennas, n_actual_e_antennas) + anffd.data.shape[2:]
    weights = np.zeros(new_shape, np.float)
    data = np.ma.array(data = np.zeros(new_shape, np.complex),
                       fill_value=0+0j)
    reduced_antenna_map = make_reduced_antenna_map(n_antennas, e_antennas_to_remove)
    # casalog.post("reduced_antenna_map".format(reduced_antenna_map), "DEBUG")
    for i1, j1 in bli.upper_triangle(n_actual_e_antennas):
        i0 = reduced_antenna_map[i1]
        j0 = reduced_antenna_map[j1]
        weights[i1, j1] = anffd.weights[i0, j0]
        weights[j1, i1] = anffd.weights[j0, i0]
        data[i1, j1] = anffd.data[i0, j0]
        data[j1, i1] = anffd.data[j0, i0]
    return data, weights

def make_reduced_antenna_map(n_antenna, e_antennas_to_remove):
    j = 0
    m = {}
    for i in range(n_antenna):
        if i in e_antennas_to_remove:
            continue
        else:
            m[j] = i
            j += 1
    return m
