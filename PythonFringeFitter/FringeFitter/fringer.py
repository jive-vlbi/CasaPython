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
