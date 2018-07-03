"""Apply calibration

Come on pylint this is a docstring."""

import numpy as np
import casac 
import ffd
import bli

class Applier(object):
    """Apply calibration"""
    def __init__(self, msname, antenna_list, swid, pol_id,
                 polind, qrest, ref_time,
                 phs, dels, rs, disps, datacol="CORRECTED"):
        self.msname = msname
        self.antenna_list = antenna_list
        self.swid = swid
        self.pol_id = pol_id
        self.polind = polind
        self.qrest = qrest
        self.ref_time = ref_time
        self.datacol = datacol
        self.phs = phs
        self.dels = dels
        self.rs = rs
        self.disps = disps
        #
        (self.freqs, n_freqs, dfreq, self.ref_freq) = \
            ffd.FFData.get_freqs(self.msname, self.swid)
    def applyCalsToMS(self):
        """Actually rotate visibilities"""
        # Required members:
        # msname
        # ref_time, polind, pol_id, swid
        # antenna_list
        # dels, phs, rs, ref_time, freqs
        ddid = ffd.FFData.get_data_desc_id(self.msname, self.swid, self.pol_id)
        bi = bli.BaselineIterator(self.antenna_list)
        f0 = np.min(self.freqs)
        fm = np.max(self.freqs)
        fs = self.freqs[:, np.newaxis]
        dfs = fs - self.f0
        tb = casac.casac.table()
        tb.open(self.msname)
        # We have Measurement Set antenna ids (s0, s1)
        # and array indices (i, j) running in parallel.
        for (s0, s1), (i, j) in bi.iter():
            dtau = self.dels[j] - self.dels[i]
            dpsi = self.phs[j] - self.phs[i]
            dr = self.rs[j] - self.rs[i]
            ddisp = self.disps[j] - self.disps[i]
            print s0, s1, dpsi, dtau, dr, ddisp
            # We can calculate the corrections due to delay and
            # constant immediately; the rate term requires time.
            phasor_tau = np.exp(-1j*2*np.pi*dtau*dfs) # Complex 1-dim array
            phasor_psi = np.exp(-1j*2*np.pi*dpsi) # Complex scalar
            ang_disp = ddisp/fs + (ddisp/f0/fm)*dfs - ddisp/f0
            print "Max dispersive angle (deg)", 180*np.max(np.abs(ang_disp))
            phasor_disp = np.exp(-1j*2*np.pi*ang_disp)
            query = ('DATA_DESC_ID = {} AND ANTENNA1={} AND ANTENNA2={}'
                     ' AND {}'.format(ddid, s0, s1, self.qrest))
            tbl = tb.query(query)
            dts = tbl.getcol('TIME') - self.ref_time
            phasor_r = np.exp(-1j*2*np.pi*dr*dts)
            vis = tbl.getcol(self.datacol)
            notvis = np.ones(vis[self.polind:].shape, np.complex64)
            # vis2 = phasor_disp*notvis
            # vis2 = phasor_psi*phasor_tau*vis
            # vis2 = phasor_psi*phasor_tau*phasor_r*phasor_disp*notvis
            vis[self.polind, :] *= phasor_psi*phasor_tau*phasor_r*phasor_disp
            tbl.putcol('CORRECTED_DATA', vis)
            tbl.flush()
            tbl.close()
        tb.flush()
        tb.close()
        return None


# Oh come ON! This is just stupidly stupid!
class DispApplier(object):
    """Apply calibration"""
    def __init__(self, msname, antenna_list, swid, pol_id,
                 polind, qrest, ref_time,
                 phs, disps, datacol="CORRECTED"):
        self.msname = msname
        self.antenna_list = antenna_list
        self.swid = swid
        self.pol_id = pol_id
        self.polind = polind
        self.qrest = qrest
        self.ref_time = ref_time
        self.datacol = datacol
        self.phis = phs
        self.disps = disps
        #
        (self.freqs, n_freqs, dfreq, self.ref_freq) = \
            ffd.FFData.get_freqs(self.msname, self.swid)
    def applyDispToMS(self):
        """Actually rotate visibilities"""
        ddid = ffd.FFData.get_data_desc_id(self.msname, self.swid, self.pol_id)
        bi = bli.BaselineIterator(self.antenna_list)
        f0 = np.min(self.freqs)
        fm = np.max(self.freqs)
        fs = self.freqs[:, np.newaxis]
        dfs = fs - f0
        tb = casac.casac.table()
        tb.open(self.msname)
        # We have Measurement Set antenna ids (s0, s1)
        # and array indices (i, j) running in parallel.
        for (s0, s1), (i, j) in bi.iter():
            dphi = self.phis[j] - self.phis[i]
            ddisp = self.disps[j] - self.disps[i]
            ang_disp = ddisp/fs + (ddisp/f0/fm)*dfs - ddisp/f0
            print "Max dispersive angle (deg)", 180*np.max(np.abs(ang_disp))
            phasor_disp = np.exp(-1j*dphi)*np.exp(-1j*2*np.pi*ang_disp)
            query = ('DATA_DESC_ID = {} AND ANTENNA1={} AND ANTENNA2={}'
                     ' AND {}'.format(ddid, s0, s1, self.qrest))
            tbl = tb.query(query)
            vis = tbl.getcol(self.datacol)
            notvis = np.ones(vis[self.polind:].shape, np.complex64)
            vis[self.polind, :] *= phasor_disp
            tbl.putcol('CORRECTED_DATA', vis)
            tbl.flush()
            tbl.close()
        tb.flush()
        tb.close()
        return None

# if False:
#     tbl = tb.open('n14p1.ms')
#     tbl = tb.query('DATA_DESC_ID=0 and ANTENNA1=0 and ANTENNA2=1 and SCAN_NUMBER=1')
#     v = tbl.getcol('CORRECTED_DATA')
#     v.shape
#     # => (4, 64, 120)
