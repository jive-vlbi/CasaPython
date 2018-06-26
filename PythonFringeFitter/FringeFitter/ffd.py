from taskinit import tb
from taskinit import casalog
import operator
import itertools
import math
import numpy as np
import numpy.ma as ma
from numpy import fft
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import scipy
# mine:
import fringer, lsqrs, utils
import param
import bli
import sys


def mkdatetime(date, time):
    return date + "-" + time

def mktimeqs(date, times):
    datetimes = [(mkdatetime(date, start), mkdatetime(date, end))
                 for start, end in times]
    timeqs = ['TIME > MJD("{}") and TIME < MJD("{}")'.format(start, end)
              for start, end in datetimes]
    return timeqs

def actual_timerangeq(msname, timeq):
    tb.open(msname)
    table2 = tb.taql(
        "select TIME, ANTENNA1, ANTENNA2 from {} where {}"
        .format(msname, timeq))
    ant_time_map = dict()
    for (t, a) in zip(table2.getcol('TIME'), table2.getcol('ANTENNA1')):
        ant_time_map.setdefault(a, set()).add(t)
    for (t, a) in zip(table2.getcol('TIME'), table2.getcol('ANTENNA2')):
        ant_time_map.setdefault(a, set()).add(t)
    casalog.post("Antennas available {}".format(
        utils.list_of_numbers_to_s(ant_time_map.keys())), "INFO")
    times_for_all_antennas = reduce(operator.and_, ant_time_map.values())
    # casalog.post("[ffd.actual_timerangeq] Times: {} {}".format(
    #     len(times_for_all_antennas), sorted(times_for_all_antennas)), "DEBUG")
    query = "TIME > {} and TIME < {}".format(min(times_for_all_antennas),
                                             max(times_for_all_antennas))
    # print >>sys.stderr, "Adjusting time query from {} to {}".format(timeq, query)
    return query

def actual_min_max_times(msname, timeq):
    tb.open(msname)
    table2 = tb.taql(
        "select TIME, TIME+INTERVAL as T1, ANTENNA1, ANTENNA2 from {} where {}"
        .format(msname, timeq))
    ant_time_map = dict()
    for (t, a) in zip(table2.getcol('TIME'), table2.getcol('ANTENNA1')):
        ant_time_map.setdefault(a, set()).add(t)
    for (t, a) in zip(table2.getcol('TIME'), table2.getcol('ANTENNA2')):
        ant_time_map.setdefault(a, set()).add(t)
    casalog.post("Antennas available {}".format(
        utils.list_of_numbers_to_s(ant_time_map.keys())), "INFO")
    times_for_all_antennas = reduce(operator.and_, ant_time_map.values())
    return min(times_for_all_antennas), max(times_for_all_antennas)

def another_min_max_time(msname, timeq):
    tb.open(msname)
    t = tb.taql("select TIME as t0, TIME+INTERVAL as t1 from {} where {}".format(
        msname, timeq))
    tmin = min(t.getcol('t0'))
    tmax = max(t.getcol('t1'))
    return tmin, tmax
    
def divide_up_timerange(msname, timeq, solint, solmin=0, solsub=1, dofloat=True):
    print >>sys.stderr, "Subdividing {}".format(timeq)
    floatEps = 1e-3
    solint = float(solint) # no silly integers for us
    tmin, tmax = another_min_max_time(msname, timeq)
    Dt = tmax - tmin
    solintChange = False
    if dofloat:
        if solint > 0.75*Dt:
            solint = Dt
            solintChange = True
        else:
            # FIXME: round solint
            n0, nerr = divmod(Dt, solint)
            print "nerr", nerr
            if nerr < floatEps:
                pass
            elif nerr <= 0.5:
                solint = Dt / n0
                solintChange = True
            else: # nerr > 0.5
                solint = Dt / (n0+1)
                solintChange = True
    if solintChange:
        casalog.post("Using solint of {}".format(solint))
    # Irregardlesswisely, we continue.
    shift = solint / solsub
    t = tmin
    times = []
    while t < tmax-floatEps:
        t1 = min(tmax, t+solint)
        times.append((t, t1))
        t += shift
    if solmin > 0:
        tpairs = filter(lambda (t0, t1): (t1-t0) > solmin, times)
    else:
        tpairs = times
    qs = map(lambda (t0, t1): "TIME > {} and TIME <= {}".format(t0, t1),
             tpairs)
    print >>sys.stderr, "Solution interval queries:", "\n".join(qs)
    return qs
    

def actual_antennas(msname, timeq):
    tb.open(msname)
    table2 = tb.taql("select TIME, ANTENNA1, ANTENNA2"
                     " from {} where ".format(msname) + timeq)
    d = {}
    for a in table2.getcol('ANTENNA1'):
        d.__setitem__(a, d.get(a, 0)+1)
    for a in table2.getcol('ANTENNA2'):
        d.__setitem__(a, d.get(a, 0)+1)
    return d

def distinct_thing(msname, query, colname):
    tb.open(msname)
    q = "select distinct {} from {} where {}".format(
        colname, msname, query)
    r = list(tb.taql(q).getcol(colname))
    try:
        assert len(r) == 1
    except:
        raise AssertionError("Query {} gives non-unique result ({})".format(q, r))
    return r[0]

def actual_source(msname):
    return distinct_thing('FIELD_ID', msname, "True")


class FFData(object):
    @staticmethod
    def checked_interval(msname, qrest):
        tb.open(msname)
        query2 = 'SELECT DISTINCT INTERVAL FROM {} WHERE {}'.format(
            msname, qrest)
        dts = tb.taql(query2).getcol('INTERVAL')
        if False:
            casalog.post("{}".format(dts), "DEBUG")
        # Check the list dts has only a single element: the
        # following will raise a ValueError otherwise.
        [dt] = dts 
        return dt
    @staticmethod
    def get_global_times(msname, qrest, startrow):
        """:param msname: The measurement set
:param qrest: A TaQL query to select the data
:param solint: the solution interval (in seconds)
:param startrow: the row of the time array to start at.
"""
        dt = FFData.checked_interval(msname, qrest)
        q1 = 'SELECT DISTINCT TIME FROM {} WHERE {}'.format(msname, qrest)
        times_t = tb.taql(q1)
        casalog.post("Query {}".format(q1), "DEBUG")
        times = times_t.getcol('TIME')[startrow:]
        (n_times,) = times.shape
        casalog.post("Distinct times {}".format(n_times), "DEBUG")
        return times, dt
    @staticmethod
    def get_freqs(msname, swid):
        tb.open(msname+"::SPECTRAL_WINDOW")
        freqs = tb.getcol('CHAN_FREQ')[:, swid]
        ref_freq = tb.getcol('REF_FREQUENCY')[swid]
        (n_freqs,) = freqs.shape
        dfreqs = freqs - freqs[0]
        # Patch to reproduce off by 1:
        # freqs = freqs[0] + dfreqs * (n_freqs+1)/n_freqs
        # Ends here
        dfs = tb.getcol('CHAN_WIDTH')[:, swid]
        dfreq = dfs[0]
        assert all(np.equal(dfs, dfreq))
        return freqs, n_freqs, dfreq, ref_freq
    @staticmethod
    def get_all_freqs(msname):
        tb.open(msname+"::SPECTRAL_WINDOW")
        chanfreqs = tb.getcol('CHAN_FREQ')
        n_chanfreqs, n_swids = chanfreqs.shape
        ref_freq = tb.getcol('REF_FREQUENCY')[0]
        n_freqs = n_swids*n_chanfreqs
        dfs = tb.getcol('CHAN_WIDTH')
        assert utils.allequal(dfs)
        df = dfs[0, 0]
        min_chan_freq = min(chanfreqs[0])
        # This is quite opinionated. It may be that *within* an SPW all
        # channels have the same width but still not the case that
        # consecutive SPWs match up.
        freqs = min_chan_freq + df*np.arange(n_freqs)
        offsets = ((chanfreqs[0] - min_chan_freq)/df).astype(int)
        return freqs, n_chanfreqs, n_freqs, df, offsets, ref_freq
    @staticmethod
    def get_data_desc_id(msname, swid, pol_id):
        tb.open(msname + "::DATA_DESCRIPTION")
        # FIXME! The polarization ID is not the polarization itself, it
        # is a pointer into the POLARIZATION table where the
        # "CORR_TYPE" column includes hard-coded integers such as 5 for
        # 'LL' and 8 for 'RR'. Harro has code on eee in a
        # mstoolutil.py::PolarizationMap class which can help.
        t = tb.query('SPECTRAL_WINDOW_ID={} AND POLARIZATION_ID={}'.format(
            swid, pol_id))
        [ddid] = list(t.rownumbers())
        return ddid
    @staticmethod
    def get_data_desc_map(msname, pol_id):
        tb.open(msname + "::DATA_DESCRIPTION")
        t = tb.query('POLARIZATION_ID={}'.format(pol_id))
        sw_to_dd = dict(zip(t.getcol('SPECTRAL_WINDOW_ID'), t.rownumbers()))
        return sw_to_dd
    # Eventually and belatedly I had a thought that
    # (1) it's reasonable to have a data-class for the data itself,
    # which ffd is,
    # (2) we could derive subclasses for filling them or even
    # (3) have separate filler classes for them but
    # (4) having the construction here with nowhere to put state is not
    # really helping.
    #
    # Sadly, though, it currently is how it is.
    @staticmethod
    def make_FFD(msname, antenna_list, swid, pol_id, polind, qrest, 
                 datacol="DATA", solint=120):
        """
:param msname: the Measurement Set
:param antenna_list: the list of antenna IDs
:param swid: the spectral window ID
:param polind: the polarization index
:param qrest: *part* of a TaQL query, suitable for inserting after a WHERE clause.
"""
        # FIXME! We don't actually *use* solint!
        # FIXME! We don't set startrow!
        # Were these meeant to be related?
        startrow = 0
        freqs, n_freqs, df, ref_freq = FFData.get_freqs(msname, swid)
        ddid = FFData.get_data_desc_id(msname, swid, pol_id)
        times, dt = FFData.get_global_times(
            msname, qrest, startrow)
        n_times = len(times)
        nrow = n_times
        bi = bli.BaselineIterator(antenna_list)
        # Make arrays:
        dshape = (bi.n_antennas, bi.n_antennas, n_freqs, n_times)
        raw_data = np.zeros(dshape, np.complex)
        data = np.zeros(dshape, np.complex)
        flags = np.ones(dshape, np.bool)
        weights = np.zeros(dshape, np.float)
        #
        tb.open(msname)
        casalog.post("Loading data", "INFO")
        # We have Measurement Set antenna ids (s0, s1)
        # and array indices (i, j) running in parallel.
        verbose = False
        for (s0, s1), (i, j) in bi.iter():
            if verbose:
                casalog.post("Loading data for base-line ({}, {}).".format(
                    s0, s1), "DEBUG")
                casalog.post("(s0, s1)={}; (i, j)={}".format(
                    (s0, s1), (i, j)), "DEBUG")
                casalog.post("ddid {}".format(ddid), "DEBUG")
            query = ('DATA_DESC_ID = {} AND ANTENNA1={} AND ANTENNA2={}'
                     ' AND {}'.format(ddid, s0, s1, qrest))
            if verbose:
                casalog.post("Query {}".format(query), "DEBUG")
            t = tb.query(query)
            casalog.post("Nrows for {}-{} = {}".format(
                s0, s1, t.nrows()), "DEBUG")
            actual_times = t.getcol("TIME", startrow=startrow, nrow=n_times)
            # NB: the timerange gets preprocessed by
            # utils.actual_timerangeq so this is safer than it looks.
            time_ints = ((actual_times-actual_times[0])/dt + 0.5).astype(np.int)
            if verbose:
                casalog.post(
                    "len(time_ints) {} {}\nn_times {}\ndt {}".format(
                        len(time_ints), time_ints[-10:], n_times, dt),
                    "DEBUG")
            d = t.getcol(datacol, startrow=startrow, nrow=nrow)[polind]
            if verbose: casalog.post("nrows: {}".format(t.nrows()), "DEBUG")
            f = t.getcol("FLAG", startrow=startrow, nrow=nrow)[polind]
            fr = t.getcol("FLAG_ROW", startrow=startrow, nrow=nrow)
            weights1d = t.getcol('WEIGHT', startrow=startrow, nrow=nrow)[polind]
            fl = operator.or_(f, fr)
            # note: can't wait till the end to unitize, because
            # unitizer dislikes zeros.
            for ai, l in enumerate(time_ints):
                if verbose: casalog.post("ai, l {} {}".format(ai, l), "DEBUG")
                raw_data[i, j, :, l] = d[:, ai]
                data[i, j, :, l] = fringer.unitize(d[:, ai])
                weights[i, j, :, l] = weights1d[ai]
                flags[i, j, :, l] = fl[:, ai]
        try:
            firstind = list(np.logical_or.reduce(
                np.isnan(data), axis=(0, 1, 2))).index(False)
        except ValueError:
            firstind = 0
        trimmed_data = data[:, :, :, firstind:]
        trimmed_weights = weights[:, :, :, firstind:]
        trimmed_flags = flags[:, :, :, firstind:]
        trimmed_raw = raw_data[:, :, :, firstind:]
        masked = ma.masked_invalid(
            ma.masked_array(trimmed_data, mask=trimmed_flags))
        masked.fill_value = (0.0+0.0j)
        ffd = FFData(masked, times[firstind:], freqs, dt, df, trimmed_weights)
        # And these are extra:
        ffd.ref_freq = ref_freq
        ffd.flags = trimmed_flags
        ffd.raw_data = trimmed_raw
        ffd.bi = bi
        # FIXME some more! We need to record the set-up data for use in applying the results
        ffd.msname = msname
        ffd.antenna_list = antenna_list
        ffd.swid = swid
        ffd.pol_id = pol_id
        ffd.polind = polind
        ffd.qrest = qrest
        ffd.datacol = datacol
        return ffd
    @staticmethod
    def make_FFD_multiband(msname, antenna_list, pol_id, polind, qrest, datacol="DATA", solint=120):
        # FIXME: We don't use solint
        bi = BaselineIterator(antenna_list)
        (times, dt) = FFData.get_global_times(msname, qrest)
        n_times = len(times)
        sw_to_dd = FFData.get_data_desc_map(msname, pol_id)
        (freqs, n_chanfreqs, n_freqs,
         df, offsets, ref_freq) = FFData.get_all_freqs(msname)
        #
        dshape = (bi.n_antennas, bi.n_antennas, n_freqs, n_times)
        raw_data = np.zeros(dshape, np.complex)
        data = np.zeros(dshape, np.complex)
        flags = np.ones(dshape, np.bool)
        weights = np.zeros(dshape, np.float)
        #
        tb.open(msname)
        for (sw, offset), ((s0, s1), (i, j)) in itertools.product(
                enumerate(offsets), bi.iter()):
            ddid = sw_to_dd[sw]
            # casalog.post("(s0, s1)={}; (i, j)={}".format(
            #    (s0, s1), (i, j)), "DEBUG")
            # casalog.post( "nrow {} ddid {} sw {} offset".format(
            #     nrow, ddid, sw, offset), "DEBUG")
            query = ('DATA_DESC_ID = {} AND ANTENNA1={} '
                     'AND ANTENNA2={} AND '.format(
                         ddid, s0, s1)+qrest)
            t = tb.query(query)
            actualrows = t.nrows()
            if actualrows==0:
                casalog.post("No data for baseline {}-{}, skipping".format(s0, s1))
                continue
            actual_times = t.getcol("TIME")
            if len(actual_times) > n_times:
                raise RuntimeError(
                    "Too many times on baseline {}-{}: {} > {}".format(
                        s0, s1, len(actual_times), n_times))
            time_ints = (actual_times-actual_times[0]/dt + 0.5).astype(np.int)
            d = t.getcol(datacol)[polind]
            # casalog.post("Query {}".format(q), "DEBUG")
            f = t.getcol("FLAG")[polind]
            fr = t.getcol("FLAG_ROW")
            weights1d = t.getcol('WEIGHT')[polind]
            fl = operator.or_(f, fr)
            # note: can't wait till the end to unitize, because
            # unitizer dislikes zeros.

            # casalog.post("Number of distinct times {} "
            #              "Number of times found {}".format(
            #        n_times, len(time_ints)), "DEBUG")
            for ai in range(len(time_ints)):
                # third index is frequency
                fslice = slice(offset, offset+n_chanfreqs)
                try:
                    raw_data[i, j, fslice, ai] = d[:, ai]
                except ValueError:
                    print >>sys.stderr, ai, fslice, offset, d.shape, raw_data.shape
                data[i, j, fslice, ai] = fringer.unitize(d[:, ai])
                weights[i, j, fslice, ai] = weights1d[ai]
                flags[i, j, fslice, ai] = fl[:, ai]
        try:
            firstind = list(np.logical_or.reduce(
                np.isnan(data), axis=(0, 1, 2))).index(False)
        except ValueError:
            firstind = 0
        tb.close()
        casalog.post("firstind {}".format(firstind), "DEBUG")
        trimmed_data = data[:, :, :, firstind:]
        trimmed_raw = raw_data[:, :, :, firstind:]
        trimmed_weights = weights[:, :, :, firstind:]
        trimmed_flags = flags[:, :, :, firstind:]
        masked = ma.masked_invalid(
            ma.masked_array(trimmed_data, mask=trimmed_flags))

        masked.fill_value = (0.0+0.0j)

        ffd = FFData(masked, times[firstind:], freqs, dt, df, trimmed_weights)
        #
        ffd.ref_freq = ref_freq
        ffd.flags = flags[:, :, :, firstind:]
        ffd.raw_data = trimmed_raw
        ffd.weights = trimmed_weights
        ffd.antenna_list = antenna_list
        ffd.bi = bi
        ffd.msname = msname
        ffd.antenna_list = antenna_list
        ffd.swid = None
        ffd.pol_id = pol_id
        ffd.polind = polind
        ffd.qrest = qrest
        ffd.datacol = datacol
        return ffd
    def __init__(self, data, times, freqs, dt, df, weights):
        self.data = data
        self.times = times
        self.freqs = freqs
        self.dt = dt
        self.df = df
        self.tgrid, self.fgrid = np.meshgrid(times, freqs)
        self.tgrid0, self.fgrid0 = np.meshgrid(times-times[0], freqs-freqs[0])
        self.weights = weights
        # FIXME: Make an extra copy of weights for SNR routine.
        self.snr_weights = weights
    def get_interval(self):
        return self.times[-1] - self.times[0]
    def get_ref_time(self):
        (n_times,) = self.times.shape
        return self.times[n_times//2]
    def get_min_time(self):
        return self.times[0]        
    def get_mid_freq(self):
        (n_freqs,) = self.freqs.shape
        return self.freqs[n_freqs//2]
    def get_antenna_index(self, i):
        return self.bi.ant_ind_map[i]
    def get_p_antenna_index(self, i):
        return self.bi.ant_inv_map[i]
    def get_baseline(self, i, j):
        if (i, j) in self.bi.e_baselines:
            res = self.data[i, j].copy()
        else:
            res = self.data[j, i].conjugate()
        return res
    def get_e_stacked_baseline(self, i, j, stack=3):
        nantennas = self.data.shape[0]
        bld = self.get_baseline(i, j)
        if stack > 1:
            ks = [k for k in range(nantennas)
                  if k != i and k != j]
            for k in ks:
                bld += self.get_baseline(i, k) * self.get_baseline(k, j)
                if stack > 2:
                    ls = [l for l in range(nantennas)
                          if l != i and l != j and l != k]
                    for l in ls:
                        bld += (self.get_baseline(i, k) *
                                self.get_baseline(k, l) *
                                self.get_baseline(l, j))
                else:
                    pass
        return fringer.unitize(bld)
    def fft(self, pad=8):
        casalog.post("Starting Fourier transform with "
                     "oversampling of {}.".format(pad))
        self.pad = pad
        # Note that we make ffts of _all_ baselines here.  We're only
        # really supposed to be doing baselines with a ref antenna.
        # That's an expensive over-reach!
        padded_shape = tuple(pad*n for n in self.data.shape[2:])
        data = (self.data * self.snr_weights).filled(fill_value=0.0)
        # data = (self.data * 1).filled(fill_value=0.0)
        b = fft.fft2(data, s=padded_shape) # does last 2 axes by default!
        # casalog.post("Shoulda transformed")
        self.c_all = fft.fftshift(b, axes=[2, 3]) # needs to be told which axes!
        (n_times,) = self.times.shape
        (n_freqs,) = self.freqs.shape
        bw = n_freqs*self.df
        T = n_times*self.dt
        self.rs = np.linspace(-n_times/(2*T), n_times/(2*T), n_times*pad,
                              endpoint=False)
        self.taus = np.linspace(-n_freqs/(2*bw), n_freqs/(2*bw), n_freqs*pad,
                                endpoint=False)
        # x axis of fft space is delay, y axis is fringe rate.
        self.xaxis0, self.yaxis0 = np.meshgrid(self.taus, self.rs,
                                               indexing='ij')
        casalog.post("Fourier transform finished.")
    def get_n_e_antennas(self):
        return self.bi.n_antennas
    def get_fringe_peak_array(self):
        shape = self.data.shape[:2]
        result = np.zeros(shape)
        for i, j in utils.itershape(shape):
            result[i, j] = self.get_fringe_peak_size(i, j)
        return result + result.transpose()
    def report_fringes(self, ref_e_antenna):
        for j in self.bi.iterate_e_baselines_to(ref_e_antenna):
            ind = self.bi.sort_e_index(ref_e_antenna, j)
            peak = self.get_fringe_peak_size(*ind)
            peak_ind = self.get_fringe_peak_ind(*ind)
            casalog.post("On baseline ({}, {}), fringe peak at {}, index {}".format(
                ref_e_antenna, j, peak, peak_ind), "DEBUG")
        return None
    def get_e_antennas_below_snr_threshold(self, ref_e_antenna, snr_threshold):
        result = []
        for j in self.bi.iterate_e_baselines_to(ref_e_antenna):
            ind = self.bi.sort_e_index(ref_e_antenna, j)
            snr = self.calc_e_snr(*ind)
            if snr < snr_threshold:
                result.append(j)
        return result
    def get_e_antennas_below_raw_threshold(self, ref_e_antenna, raw_threshold):
        result = []
        for j in self.bi.iterate_e_baselines_to(ref_e_antenna):
            ind = self.bi.sort_e_index(ref_e_antenna, j)
            peak = self.get_fringe_peak_size(*ind)
            if peak < raw_threshold:
                result.append(j)
        return result
    def get_p_antennas_below_snr_threshold(self, ref_p_antenna, snr_threshold):
        ref_e_antenna = self.bi.p_to_e(ref_p_antenna)
        e_antennas = self.get_e_antennas_below_snr_threshold(ref_e_antenna, snr_threshold)
        p_antennas = [self.bi.e_to_p(e) for e in e_antennas]
        return p_antennas
    def get_p_antennas_below_raw_threshold(self, ref_p_antenna, raw_threshold):
        ref_e_antenna = self.bi.p_to_e(ref_p_antenna)
        e_antennas = self.get_e_antennas_below_raw_threshold(ref_e_antenna, raw_threshold)
        p_antennas = [self.bi.e_to_p(e) for e in e_antennas]
        return p_antennas
    def calc_e_snr(self, i, j):
        """Calculate baseline signal to noise ration using the formula from
AIPS FRING.FOR lines 4162 t/m 4163.

I have literally no idea where the tangents and powers come from, sorry."""
        d = self.data[i, j]
        w = self.snr_weights[i, j]
        d.count(axis=0)
        xcount = d.count()
        sumw = np.sum(w)
        sumww = np.sum(w*w)
        work1 = self.get_fringe_peak_size(i, j)
        # casalog.post("Baseline {}-{}".format(i,j), "DEBUG")
        # casalog.post("work1 {}\nxcount {}\nsumw {}\nsumww {}".format(
        #     work1, xcount, sumw, sumww), "DEBUG")
        if xcount == 0:
            casalog.post(
                "Baseline (matrix indices) {}-{} has no data".format(i, j))
            cwt = 0
        else:
            cwt = ((math.tan(math.pi/2*work1/sumw)**1.163) *
                   math.sqrt(sumw/math.sqrt(sumww/xcount)))
        return cwt
    def report_snrs(self, ref_e_antenna, snr_threshold=0.0):
        bad_antennas = []
        for j in self.bi.iterate_e_baselines_to(ref_e_antenna):
            ind = self.bi.sort_e_index(ref_e_antenna, j)
            snr = self.calc_e_snr(*ind)
            if snr < snr_threshold:
                bad_antennas.append(j)
            casalog.post(
                "On baseline ({}, {}), SNR of {}".format(
                    ref_e_antenna, j, snr), "DEBUG")
        return bad_antennas
    def mask_fringes_below_threshold(self, threshold):
        """This masks FFT fringes on any baseline where the FFT peak is below a threshold.

This shouldn't be necessary, since we actually want to remove *antennas* with
inadequate SNR to the reference antenna."""
        baselines = []
        shape = self.data.shape[:2]
        for i, j in utils.itershape(shape):
            peak = self.get_fringe_peak_size(i, j)
            if peak < threshold:
                self.data.mask[i, j] = True
                baselines.append((i, j))
        return baselines
    def get_e_antennas_below_threshold(self, ref_e_antenna, threshold):
        """Select e_indices of antennas whose FFT fringe peaks to ref_e_antenna are below a threshold.

These antennas should be removed from the data set before the least-squares algorithm is applied."""
        n_e_antennas = self.get_n_e_antennas()
        l = []
        for j in range(n_e_antennas):
            if j == ref_e_antenna:
                continue
            else:
                peak = self.get_fringe_peak_size(ref_e_antenna, j)
                if peak < threshold:
                    l.append(j)
        return l
    def get_fringe_peak(self, i, j):
        c = self.c_all[i, j]
        ind = self.get_fringe_peak_ind(i, j)
        fringe_rate = self.yaxis0[ind]
        delay = self.xaxis0[ind]
        # FIXME: I still don't understand why this is this way round and
        # not -this.
        # psi = np.angle(np.sum(c[xind-self.pad/2:xind+self.pad/2,
        #                         yind-self.pad/2:yind+self.pad/2]))
        psi = np.angle(c[ind])
        return psi, delay, fringe_rate
    def get_fringe_peak_size(self, i, j):
        c = self.c_all[i, j]
        ind = self.get_fringe_peak_ind(i, j)
        peak = abs(c)[ind]
        return peak
    def get_fringe_peak_ind(self, i, j):
        c = self.c_all[i, j]
        ind = fringer.index_of_max(abs(c))
        return ind
    def get_params(self, ref_antenna=None):
        params = []
        if ref_antenna == None:
            ref_antenna = self.bi.antenna_list[0]
        for k in self.bi.antenna_list:
            if k == ref_antenna:
                params.extend([0.0, 0.0, 0.0])
            else:
                # find the index for ref_antenna-to-p baseline
                sgn, (i, j) = self.bi.get_sign_and_indices(ref_antenna, k)
                psi, delay, rate = self.get_fringe_peak(i, j)
                params.extend([sgn*psi, sgn*rate, sgn*delay])
        return params
    def make_ref_weights(self, ref_e_antenna):
        # Not currently used
        n_antennas = self.bi.n_antennas
        weights = np.zeros((n_antennas, n_antennas, 1, 1))
        for j in self.bi.iterate_e_baselines_to(ref_e_antenna):
            weights[j, ref_e_antenna] = 1.0
            weights[ref_e_antenna, j] = 1.0
        self.weights = weights
    def make_weighted_weights(self):
        sh = self.data.shape[:2] + (1, 1)
        weights = np.zeros(sh)
        w = np.sum(np.abs(self.raw_data)*self.weights, axis=(2, 3))
        w += w.transpose()
        weights[:, :, 0, 0] = w
        self.weights = weights
    def fit_fringe(self, ref_antenna, antennas2, pad=8,
                       threshold_method=None, threshold=0.0, snr_threshold=0.0,
                       snr_threshold_method=None):
        self.fft(pad=pad)
        (nf,) = self.freqs.shape
        n_antennas = len(antennas2)
        F_midpoint = self.fgrid0[nf//2][0]
        # get_params uses p-antennas
        params2 = self.get_params(ref_antenna=ref_antenna)
        self.params2 = params2
        e_ref_antenna = self.get_antenna_index(ref_antenna) # FIXME: really?
        # everything else uses e-antennas
        ref_params = param.get_antenna_parameters(params2, e_ref_antenna)
        self.report_fringes(e_ref_antenna)
        self.report_snrs(e_ref_antenna)
        ## FIXME: I don't know why we're doing this here, or at all, but
        ## SNRs need the other sensible kind of weights.
        self.make_weighted_weights()
        if threshold_method == 'raw':
            e_antennas_to_remove0 = self.get_e_antennas_below_threshold(e_ref_antenna, threshold)
        elif threshold_method == 'snr':
            e_antennas_to_remove0 = self.report_snrs(e_ref_antenna, snr_threshold)
        else:
            e_antennas_to_remove0 = []
        e_antennas_to_remove = sorted(e_antennas_to_remove0)
        # Here we need to remove also antennas from self.data and self.weights
        p_antennas_to_remove = [self.get_p_antenna_index(e) for e in e_antennas_to_remove]
        p_antennas_to_keep = [self.get_p_antenna_index(e) for e in range(n_antennas)
                              if e not in  e_antennas_to_remove]
        if len(e_antennas_to_remove) > 0:
            casalog.post("{} antennas have fringes below thresholds and will be removed".format(
                len(e_antennas_to_remove), "INFO"))
            casalog.post("Masking antennas " + ", ".join("{}".format(p) for p in p_antennas_to_remove),
                         "INFO")
        self.trim_data(e_antennas_to_remove)
        n_actual_e_antennas = len(antennas2) - len(e_antennas_to_remove)
        # # FIXME: can this be folded into new reduced_antenna_map paradigm?
        params3, removed_params = param.remove_antennas(params2, e_antennas_to_remove)
        new_e_ref_antenna = e_ref_antenna - sum([int(a < e_ref_antenna) for a in e_antennas_to_remove])
        params3b = params3[:]
        disps = 1e-11
        for i in range(len(params3b)/3):
            d = 0 if i==0 else disps
            params3b.insert(4*(i+1)-1, d)
        assert (len(params3b) % 4) == 0
        params4, ref_params = param.remove_ref(params3b, new_e_ref_antenna)
        casalog.post("params3b: {}".format(params3b))
        casalog.post("params4: {}".format(params4))
        args=(self.tgrid0, self.fgrid0, self.freqs, n_actual_e_antennas, self.data, self.weights, new_e_ref_antenna)
        casalog.post("Starting least-squares solver", "INFO")
        # Caution! New!
        eps = 1e-10
        sweight = np.sum(self.weights*np.logical_not(self.data.mask))
        ftol = eps*sweight
        casalog.post(
            "Sum of weights {}; error per phasor {}; ftol={}."
            "".format(sweight, eps, ftol))
        res_t = scipy.optimize.leastsq(lsqrs.vector_s3_test, params4,
                                       full_output=1, args=args,
                                       maxfev=100,
                                       # col_deriv=True,
                                       # Dfun=lsqrs.matrix_j_s3,
                                       ftol=ftol,
                                       diag = (n_actual_e_antennas-1)*[1,  1e-2, 1e-7, 1e-2], 
        )
        casalog.post("Least-squares solver finished", "INFO")
        sol_out = list(res_t[0])
        casalog.post("Least-squares solver finished after {} iterations".format(res_t[2]['nfev']), "INFO")
        # 1 t/m 4 good; others bad.
        casalog.post("Solver status {}, message: {}".format(res_t[4], res_t[3]), "INFO")
        # casalog.post("sol_out".format(sol_out), "DEBUG")
        pout2d0 = param.restore_ref(sol_out, ref_params, new_e_ref_antenna)
        dels0 = param.get_delays(sol_out)
        flags, pout2d = param.restore_antennas(pout2d0, removed_params, e_antennas_to_remove)
        #
        dels = param.get_delays(pout2d)
        r0s = param.get_rates(pout2d)
        rs = r0s/self.ref_freq
        phs0 = param.get_phases(pout2d)
        disps = param.get_disps(pout2d)
        sigma_p = param.get_phases(pout2d) 
        casalog.post("All terms: {}".format(pout2d), "INFO")
        casalog.post("Delay terms: {}".format(dels), "INFO")
        casalog.post("Rate terms: {}".format(rs), "INFO")
        casalog.post("Dispersive terms: {}".format(disps), "INFO")
        DT = self.tgrid0[0, -1]
        # phs = +np.array(phs0) + np.pi*DT*np.array(r0s)
        # We do NOT correct anything here; leave that to the writer.
        phs = np.array(phs0) 
        # + 2*np.pi*F_midpoint*np.array(dels)
        self.flags = flags
        self.dels = dels
        self.phs = phs
        self.rs = rs
        self.disps = disps
        self.sigma_p = sigma_p
    def trim_data(self, e_antennas_to_remove):
        n_antennas = self.data.shape[0]
        n_actual_e_antennas = n_antennas - len(e_antennas_to_remove)
        new_shape = (n_actual_e_antennas, n_actual_e_antennas) + self.data.shape[2:]
        weights = np.zeros(new_shape, np.float)
        data = np.ma.array(data = np.zeros(new_shape, np.complex),
                           fill_value=0+0j)
        reduced_antenna_map = fringer.make_reduced_antenna_map(n_antennas, e_antennas_to_remove)
        # casalog.post("reduced_antenna_map".format(reduced_antenna_map), "DEBUG")
        for i1, j1 in bli.upper_triangle(n_actual_e_antennas):
            i0 = reduced_antenna_map[i1]
            j0 = reduced_antenna_map[j1]
            self.weights[i1, j1] = self.weights[i0, j0]
            self.weights[j1, i1] = self.weights[j0, i0]
            self.data[i1, j1] = self.data[i0, j0]
            self.data[j1, i1] = self.data[j0, i0]


class FFDPlotter(object):
    def __init__(self, ffd):
        self.ffd = ffd
    ## New: we plot all the baselines straight out of the ffd.
    ## Remark: This should presumably also be a
    def plot_fringes(self):
        n_cols = len(self.ffd.baselines)
        for l, (i, j) in enumerate(self.ffd.e_baselines):
            self.plot_baseline_fringe_p(l, n_cols, i, j)
    def plot_baseline_fringe(self, i, j):
        self.plot_baseline_fringe_p(0, 1, i, j)
    def plot_baseline_fringe_p(self, l, n_cols, i, j):
        nlumps = 5 # number of peaks in sincified data to show
        (n_times,) = self.ffd.times.shape
        (n_freqs,) = self.ffd.freqs.shape

        c = self.ffd.c_all[i, j]
        peak_ind = self.ffd.get_fringe_peak_ind(i, j)
        xpeak_ind, ypeak_ind = peak_ind
        xpeak = self.ffd.xaxis0[peak_ind]
        ypeak = self.ffd.yaxis0[peak_ind]
        casalog.post("Max value {}; Fringe rate: {}; Delay: {}".format(
            c[peak_ind], ypeak, xpeak), "DEBUG")
        casalog.post("Index {}".format(peak_ind), "DEBUG")
        #
        pad = self.ffd.pad
        xslice = slice(n_freqs*pad/2 - nlumps*pad, n_freqs*pad/2 + nlumps*pad)
        yslice = slice(n_times*pad/2 - nlumps*pad, n_times*pad/2 + nlumps*pad)
        #
        plt.subplot(2, n_cols, l+1)
        plt.plot(self.ffd.xaxis0[xslice, ypeak_ind],
                 np.abs(c[xslice, ypeak_ind]))
        plt.ylabel('Fringe height')
        plt.xlabel('Group delay (s) ({}-{})'.format(i, j))
        plt.axvline(x=xpeak, linestyle='--', color='r')
        #
        plt.subplot(2, n_cols, n_cols + l+1)
        plt.plot(self.ffd.yaxis0[xpeak_ind, yslice],
                 np.abs(c[xpeak_ind, yslice]))
        plt.ylabel('Fringe height')
        plt.xlabel('Delay rate (s/s) ({}-{})'.format(i, j))
        plt.axvline(x=ypeak, linestyle='--', color='r')
        #
        title = "Baseline {}-{}".format(i, j)
        plt.title(title, y=2.75)
        plt.show()
    def plot_baseline_fringe3d(self, i, j, keyhole0=0.5, title=''):
        keyhole = keyhole0/self.ffd.pad
        c = self.ffd.c_all[i, j]
        peak = self.ffd.get_fringe_peak_ind(i, j)
        scutout = [slice(max(0, k-keyhole*n), min(n, k+keyhole*n+1))
                   for (k, n) in zip(peak, c.shape)]
        d = np.abs(c[scutout])
        xaxis0, yaxis0 = np.meshgrid(self.ffd.taus, self.ffd.rs, indexing='ij')
        if title != '':
            plt.title(title)
        ax = plt.axes(projection='3d')
        #
        ax.plot_surface(xaxis0[scutout], yaxis0[scutout],
                        d, cmap=plt.cm.jet, rstride=1, cstride=1)
        ax.set_xlabel("delay (s)")
        ax.set_ylabel("fringe-rate (rad/s)")
        plt.show()
    def plot_baseline_fringe2d(self, i, j, keyhole0=0.5, title=''):
        keyhole = keyhole0/self.ffd.pad
        c = self.ffd.c_all[i, j]
        peak = self.ffd.get_fringe_peak_ind(i, j)
        scutout = [slice(max(0, int(p-keyhole*n)), min(n, p+keyhole*n+1))
                   for (p, n) in
                   zip(peak, c.shape)]
        d = np.abs(c[scutout])
        xmin = self.ffd.taus[scutout[0].start]
        xmax = self.ffd.taus[scutout[0].stop]
        ymin = self.ffd.rs[scutout[1].start]
        ymax = self.ffd.rs[scutout[1].stop]
        extent = [xmin, xmax, ymin, ymax]
        if title == '':
            title = "FFT fringe plot for {}-{}".format(i, j)
        plt.title(title)
        plt.imshow(d.T,
                   interpolation='nearest',
                   extent=extent,
                   aspect='auto',
                   origin='lower',
                   vmin=d.min(), vmax=d.max())
        plt.xlabel('delay (s)')
        plt.ylabel('delay-rate (rad/s)')
        plt.colorbar()
        plt.show()
    def plot_phase(self, i, j, title=''):
        phis = np.angle(np.ndarray.mean(self.ffd.data[i, j], axis=1))
        plt.figure()
        if title == '':
            title = "Phase across the band"
        plt.title(title)
        plt.plot(range(len(phis)), phis)
        plt.ylabel('Phase angle (radians)')
        plt.xlabel('Channel {}'.format((i, j)))

