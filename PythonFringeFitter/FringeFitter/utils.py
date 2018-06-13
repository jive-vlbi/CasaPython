import operator, itertools
from casac import casac
import numpy as np, math
import lsqrs
import tasks

tb = casac.table()
qa = casac.quanta()

def list_of_numbers_to_s(l):
    return ", ".join([str(x) for x in l])

def get_antenna_map(msname, name_col='NAME'):
    tb.open("{}::ANTENNA".format(msname))
    # name_col = 'STATION'
    return dict((s, i) for (i,s) in enumerate(tb.getcol(name_col)))

def get_nchans(vis):
    tb.open(vis)
    q = 'select distinct NUM_CHAN from {}::SPECTRAL_WINDOW'.format(vis)
    (nchans,) = tuple(tb.taql(q).getcol('NUM_CHAN'))
    return nchans

# Flagging command:
def flag_edges(vis, width=2):
    nchans = get_nchans(vis)
    tasks.flagdata(vis, spw='*:{}~{};{}~{}'.format(0, width, nchans-width-1, nchans-1, 
                                                  mode='manual', flagbackup=True))
    
def print_res(antenna_map, antennas2, phases, delays, rates, sigs=None):
    # Note: prints here are allowed
    inv_sm = dict((v, k) for k,v in antenna_map.iteritems())
    fmts = " ".join(["{:3>}"] + 6*["{:>20}"])
    print '#',fmts.format('', "Phase1", "Phase2", "Delay1", "Delay2", "Rate1", "Rate2")
    for i, s in enumerate(antennas2):
        st = inv_sm[s]
        p1, d1, r1 = phases[0][i], delays[0][i], rates[0][i]
        p2, d2, r2 = phases[1][i], delays[1][i], rates[1][i]
        print fmts.format(st, p1, p2, d1, d2, r1, r2),
        if sigs:
            print "{:>20} {:>20}".format(sigs[0][3*i+2], sigs[1][3*i+2])
        else:
            print


def flatsize(a): return reduce(operator.mul, a.shape)


def itershape(shape):
    return itertools.product(*map(range, shape))

def allequal(a):
    i0 = tuple(0 for d in a.shape)
    return np.all(np.equal(a, a[i0]))

def size_bl(msname, i, j, constraint='True'):
    # eg. constraint = 'DATA_DESC_ID=0 and SCAN_NUMBER=2' 
    tb.open(msname)
    q = ('select gsum(sumsquare(abs(DATA))) from {} '
         'where ANTENNA1={} and ANTENNA2={} and {}'.format(msname, i, j, constraint))
    return tb.taql(q).getcol('Col_1')

#FIXME: needs a sensible name
def total_perspective_vortex(msname, antennas, constraint='True'):
    n = len(antennas)
    w = np.zeros((n, n), np.float)
    index_thing = zip(bli.triangle(n), bli.triangle_l(antennas))
    for ((ai, aj), (i, j)) in index_thing:
        si, sj = bli.ij(i, j)
        w[ai, aj]  =  np.float(size_bl(msname, si, sj, constraint))
    w /= np.sum(w)
    w += np.transpose(w)
    return w

def get_scan_length(msname, scan_number):
    tb.open(msname)
    ts = tb.query("SCAN_NUMBER={}".format(scan_number)).getcol("TIME")
    DT = int(math.ceil(max(ts) - min(ts)))
    return DT
    

def get_source_times(msname, source_name):
    tb.open(msname)
    t = tb.taql("SELECT DISTINCT TIME FROM {} "
                "WHERE FIELD_ID IN [SELECT rowid() "
                "FROM ::FIELD where NAME ~ p/{}/]".format(msname, source_Name))
    times = t.getcol("TIME")
    return times

def pairwise(l):
    return zip(l[:-1], l[1:])

def get_source_time_ranges(msname, source_name, solint=None):
    tb.open(msname)
    t = tb.taql("SELECT DISTINCT TIME FROM {} WHERE FIELD_ID IN "
                "[SELECT rowid() FROM ::FIELD where NAME ~ p/{}/]".format(
                    msname, source_Name))
    times = t.getcol("TIME")
    [dt] = tb.query("TRUE", columns="DISTINCT INTERVAL").getcol("INTERVAL")
    epsilon = 1.0e-05
    # break_Inds marks the _end_ of an interval
    bIs = np.where((times[1:]-times[:-1])>dt+epsilon)[0]
    l = [0] + list(np.ndarray.flatten(np.array(list(zip(bIs, bIs + 1))))) + [-1]
    timeranges = [(times[l[i]], times[l[i+1]])
                  for i in itertools.islice(range(len(l)), None, None, 2)]
    if solint != None:
        timeranges = reduce(operator.add,
                            [(list(np.arange(t0, t1, solint))+ [t1]) for (t0, t1) in timeranges])
    return timeranges

def get_polarization_ids(msname):
    tb.open(msname)
    t = tb.taql("select distinct POLARIZATION_ID from {}::DATA_DESCRIPTION".format(msname))
    return list(t.getcol("POLARIZATION_ID"))

def get_n_polarizations(msname, pol_id):
    t = tb.taql("select NUM_CORR from {}::POLARIZATION".format(msname))
    return t.getcol("NUM_CORR")[pol_id]

def get_sources(msname):
    tb.open(msname)
    t = tb.taql('SELECT DISTINCT rowid(), NAME FROM {}::FIELD'.format(msname))
    return zip(t.getcol('Col_1'), t.getcol('NAME'))

def get_spectral_windows(msname):
    tb.open(msname)
    t = tb.taql('SELECT rowid() FROM {}::SPECTRAL_WINDOW'.format(msname))
    return list(t.getcol('Col_1'))

def get_antennas(msname, timeq):
    tb.open(msname)
    t2 = tb.taql("select ANTENNA1, ANTENNA2 from {} where ".format(msname) + timeq)
    s = set(t2.getcol('ANTENNA1')) | set(t2.getcol('ANTENNA2'))
    return sorted(s)

def antenna_by_name(msname, s):
    tb.open(msname)
    [r] = tb.taql('SELECT rowid() FROM {}::ANTENNA where NAME ~ p/{}/'.format(msname, 'EF')).getcol('Col_1')
    return r

def get_min_freqs(msname):
    tb.open(msname)
    freqs = tb.taql("select DISTINCT min(CHAN_FREQ) as f from {}".format(msname+"::SPECTRAL_WINDOW")).getcol('f')
    return freqs

def get_mid_freqs(msname):
    tb.open(msname)
    [nf] = list(tb.taql("select DISTINCT NUM_CHAN as n from {}".format(msname+"::SPECTRAL_WINDOW")).getcol('n'))
    freqs = tb.taql("select DISTINCT CHAN_FREQ[{}] as f from {}".format(nf//2, msname+"::SPECTRAL_WINDOW")).getcol('f')
    return freqs

def get_reference_freqs(msname):
    tb.open(msname)
    t = tb.taql("select REF_FREQUENCY from {}::SPECTRAL_WINDOW".format(msname))
    return list(t.getcol("REF_FREQUENCY"))


def get_ref_freq_dif(msname):
    tb.open("{}::SPECTRAL_WINDOW".format(msname))
    rfs = tb.getcol("REF_FREQUENCY")
    fs = tb.getcol("CHAN_FREQ")
    return rfs-fs[0]

def flattenificate(a):
    n_sw, n_pol, n_ant = a.shape
    l = []
    for isw in range(n_sw):
        for iant in range(n_ant):
            for ipol in range(n_pol):
                l.append(a[isw, ipol, iant])
    return l

def render_time(t):
    tu = qa.quantity(t, 's') # => {'unit': 'd', 'value': 57575.579976851855}
    return qa.time(tu,form="fits")[0] 

def turns_to_radians(t):
    """Convert turns of phase into radians from -pi to pi.

You'd think it would be hard to get this one wrong, but I established
the hard way that it is at least possible."""
    return (((2*t + 1) % 2.0) - 1)*np.pi

