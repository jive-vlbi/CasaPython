import math, numpy as np, numpy.ma as ma
import operator
import ffd, utils

def orderize(i, j):
    return (i, j) if i<j else (j, i)

def signed_orderize(i, j):
    return (1, (i, j)) if i<j else (-1, (j, i))


class ClosureTool(object):
    def __init__(self, msname, scan, pol_id, polind=0):
        self.msname = msname
        self.scan = scan
        self.pol_id = pol_id
        self.polind = polind
        self.timeq = "SCAN_NUMBER={}".format(self.scan)
        self.antennas = sorted(ffd.actual_antennas(self.msname, self.timeq).keys())
        self.antenna_map = utils.get_antenna_map(msname)
        self.anffd = ffd.FFData.make_FFD(msname, self.antennas,
                                         self.pol_id, self.polind, self.timeq,
                                         datacol="CORRECTED_DATA", solint=3000)
    def closures(self, n1, n2, n3):
        p1, p2, p3 = map(self.antenna_map.get, [n1, n2, n3])
        return self.p_closures(p1, p2, p3)
    def p_closures(self, p1, p2, p3):
        i1, i2, i3 = map(self.anffd.bi.p_to_e, [p1, p2, p3])
        return self.i_closures(i1, i2, i3)
    def i_closures(self, i1, i2, i3):
        d = self.anffd.data
        assert i1 != i2
        assert i2 != i3
        assert i3 != i1
        # Could write this as a reduce.
        flags = d[orderize(i1, i2)].mask | d[orderize(i2, i3)].mask | d[orderize(i3, i1)].mask

        s1, t1 = signed_orderize(i1, i2)
        closure = s1*np.angle(d[t1])
        s2, t2 = signed_orderize(i2, i3)
        closure += s2*np.angle(d[t2])
        s3, t3 = signed_orderize(i3, i1)
        closure += s3*np.angle(d[t3])
        closure = ma.array(closure, mask=flags)
        return 180/np.pi*np.mean(closure)
    def baseline(self, n1, n2):
        d = self.anffd.data
        p1, p2 = map(self.antenna_map.get, [n1, n2])
        i1, i2 = map(self.anffd.bi.p_to_e, [p1, p2])
        s1, t1 = signed_orderize(i1, i2)
        bl = s1*np.angle(d[t1])
        return 180/np.pi*np.mean(bl)
    def all_triangles(self, n1, n2):
        result = {}
        for k in self.antenna_map.keys():
            if k == n1  or k == n2:
                continue
            try:
                angle = self.closures(n1, n2, k)
                result[k] = angle
            except KeyError:
                continue
        return result
    
if __name__=='__main__':
    msname = "n14c2.ms"
    scan = 5
    pol_id = 0 # RR
    c = closure.ClosureTool(msname, scan, pol_id)
    # Answers are in *degrees*
    c.closures("EF", "ON", "BD")
    # Can also check per baseline to make sure.
    c.baseline("ZC", "EF")
