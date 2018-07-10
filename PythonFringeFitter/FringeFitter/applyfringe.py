"""Apply calibration

Come on pylint this is a docstring."""

import numpy as np
import casac 
import ffd
import bli

class Phasor(object):
    def __init__(self, freqs):
        self.f0 = np.min(freqs)
        self.fm = np.max(freqs)
        self.fs = freqs[:, np.newaxis]
        self.dfs = self.fs - self.f0
    def getPhasorDelay(self, dtau):
        phasor_tau = np.exp(-1j*2*np.pi*dtau*self.dfs) # Complex n-by-1 matrix
        return phasor_tau
    def getPhasorConst(self, dpsi):
        phasor_psi = np.exp(-1j*2*np.pi*dpsi) # Complex scalar
        return phasor_psi
    def getPhasorDisp(self, ddisp):
        ang_disp = ddisp/self.fs + (ddisp/self.f0/self.fm)*self.dfs - ddisp/self.f0
        return np.exp(-1j*2*np.pi*ang_disp)
    def getCorrectionPhasor(self, dpsi, ddisp):
        return self.getPhasorConst(dpsi)*self.getPhasorDisp(ddisp)
    
class TimePhasor(Phasor):
    def __init__(self, freqs, dts):
        Phasor.__init__(self, freqs)
        self.dts = dts
    def getPhasorRate(self, dr):
        phasor_rates = np.exp(-1j*2*np.pi*dr*self.dts) # Complex vector
        return phasor_rates
    def getCorrectionPhasor(self, dpsi, dtau, dr, ddisp):
        phasor_psi = self.getPhasorConst(dpsi)
        phasor_tau = self.getPhasorDelay(dtau)
        phasor_rate = self.getPhasorRate(dr)
        phasor_disp = self.getPhasorDisp(ddisp)
        return phasor_psi*phasor_tau*phasor_rate*phasor_disp

class Applier(object):
    def __init__(self, msname, antenna_list, swid, pol_id,
                 polind, qrest, ref_time, datacol="CORRECTED"):
        self.msname = msname
        self.antenna_list = antenna_list
        self.swid = swid
        self.pol_id = pol_id
        self.polind = polind
        self.qrest = qrest
        self.ref_time = ref_time
        self.datacol = datacol
        (self.freqs, n_freqs, dfreq, self.ref_freq) = \
            ffd.FFData.get_freqs(self.msname, self.swid)
        self.ddid = ffd.FFData.get_data_desc_id(self.msname, self.swid, self.pol_id)
    
    
class AllApplier(Applier):
    """Apply calibration"""
    def __init__(self, msname, antenna_list, swid, pol_id,
                 polind, qrest, ref_time,
                 phs, dels, rs, disps, datacol="CORRECTED"):
        Applier.__init__(self, msname, antenna_list, swid, pol_id,
                         polind, qrest, ref_time,
                         datacol)
        #
        self.phs = phs
        self.dels = dels
        self.rs = rs
        self.disps = disps
    def getPhasor(self, freqs, dts):
        return TimePhasor(freqs, dts)
    def getParams(self, i, j):
        dpsi = self.phs[j] - self.phs[i]
        dtau = self.dels[j] - self.dels[i]
        dr = self.rs[j] - self.rs[i]
        ddisp = self.disps[j] - self.disps[i]
        return [dpsi, dtau, dr, ddisp]
    def applyCalsToMS(self):
        """Actually rotate visibilities"""
        bi = bli.BaselineIterator(self.antenna_list)
        tb = casac.casac.table()
        tb.open(self.msname)
        # We have Measurement Set antenna ids (s0, s1)
        # and array indices (i, j) running in parallel.
        for (s0, s1), (i, j) in bi.iter():
            # Get times and a data table. This is upsetting because the
            # time dimension is baseline-dependent and i had intended
            # my phasor class to be baseline agnostic.
            query = ('DATA_DESC_ID = {} AND ANTENNA1={} AND ANTENNA2={}'
                     ' AND {}'.format(self.ddid, s0, s1, self.qrest))
            tbl = tb.query(query)
            dts = tbl.getcol('TIME') - self.ref_time
            phasor = self.getPhasor(self.freqs, dts)
            # We can calculate the corrections due to delay and
            # constant immediately; the rate term requires time.
            params = self.getParams(i, j)
            correction = phasor.getCorrectionPhasor(*params)
            # The variable notvis is used when we want to hack and see
            # the correction *term* instead of the corrected data
            vis = tbl.getcol(self.datacol)
            notvis = np.ones(vis[self.polind:].shape, np.complex64)
            vis[self.polind, :] *= correction
            tbl.putcol('CORRECTED_DATA', vis)
            tbl.flush()
            tbl.close()
        tb.flush()
        tb.close()
        return None


# Oh come ON! This is just stupidly stupid!
class DispApplier(Applier):
    """Apply calibration"""
    def __init__(self, msname, antenna_list, swid, pol_id,
                 polind, qrest, ref_time,
                 phs, disps, datacol="CORRECTED"):
        Applier.__init__(self, msname, antenna_list, swid, pol_id,
                         polind, qrest, ref_time,
                         datacol="CORRECTED")
        self.phis = phs
        self.disps = disps
    def getParams(self, i, j):
        dphi = self.phis[j] - self.phis[i]
        ddisp = self.disps[j] - self.disps[i]
        return [dphi, ddisp]
    def getPhasor(self, freqs, dts):
        phasor = Phasor(self.freqs)

# if False:
#     tbl = tb.open('n14p1.ms')
#     tbl = tb.query('DATA_DESC_ID=0 and ANTENNA1=0 and ANTENNA2=1 and SCAN_NUMBER=1')
#     v = tbl.getcol('CORRECTED_DATA')
#     v.shape
#     # => (4, 64, 120)
