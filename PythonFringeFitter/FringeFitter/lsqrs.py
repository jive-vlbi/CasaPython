import operator
import itertools
import numpy as np
import utils, bli

def add_ref_zeroes(v0, ref_ant):
    return add_n_ref_zeroes(v0, ref_ant, 4)

def add_n_ref_zeroes(v0, ref_ant, n):
    l = list(v0)
    for i in range(n):
        l.insert(4*ref_ant+i, 0.0)
    v = np.array(l)
    return v


class BaselinePhasorAll(object):
    def __init__(self, dts, dfs, freqs):
        self.f0 = np.min(freqs)
        self.fm = np.max(freqs)
        # dfs and dts are the time and frequency grids for the ffd data
        # set.  They necessarily have the same two-dimensional shape
        self.dfs = dfs
        self.dts = dts
    def get_grid_shape(self):
        return self.dfs.shape
    def get_grid_size(self):
        return utils.flatsize(self.dfs)
    def get_nparam(self):
        return 4
    def make_model(self, dpsi, dr, dtau, ddisp):
        ph_disp = (ddisp/(self.f0+self.dfs) +
                   (ddisp/self.f0/self.fm)*self.dfs -
                   ddisp/self.f0)
        
        result = (np.exp(1j*dpsi) *
                  np.exp(1j*2*np.pi*dr*self.dts) *
                  np.exp(1j*2*np.pi*dtau*self.dfs) *
                  np.exp(1j*2*np.pi*ph_disp))
        return result
    # FIXME: not used or tested, yet.
    def make_jacobian(self, dpsi, dr, dtau, ddisp):
        jac_shape = (4,) + self.dfs.shape
        jac = np.zeros(jac_shape, np.complex)
        # We *don't* handle weights in this class at all.
        D = 1j*(self.make_model(dpsi, dr, dtau, ddisp))
        jac[0, :, :] = D            # psi derivative
        jac[1, :, :] = 2*np.pi*self.dts*D # r derivative
        jac[2, :, :] = 2*np.pi*self.dfs*D # tau derivative
        jac_disp = (1/(self.f0+self.dfs) +
                    1/self.f0/self.fm*self.dfs -
                    1/self.f0)
        jac[3, :, :] = 2*np.pi*jac_disp*D
        return jac


## Factoring out a baseline model generator into a class means we can
## use the same framework for generating delay and rate models or
## dispersive models by using an instance of a different BaselinePhasor
## class as our generator.

class BaselinePhasorDisp(BaselinePhasorAll):
    def __init__(self, dts, dfs, freqs):
        self.dts = dts
        self.dfs = dfs
        self.freqs = freqs
        self.f0 = np.min(freqs)
        self.fm = np.max(freqs)
        self.fs = self.f0 + self.dfs
    def get_nparam(self):
        return 2
    def make_model(self, dpsi, ddisp):
        # We add a final argument for the sky-frequencies.
        ph_disp = ddisp/(self.fs) + (ddisp/self.f0/self.fm)*self.dfs - ddisp/self.f0
        result = (np.exp(1j*dpsi) *
                  np.exp(1j*2*np.pi*ph_disp))
        return result
    def make_jacobian(self, dpsi, ddisp):
        jac_shape = (2,) + self.dfs.shape
        jac = np.zeros(jac_shape)
        # We *don't* handle weights in this class at all.
        D = 1j*(self.make_model(dpsi, dr, dtau, ddisp))
        jac[0, :, :] = D            # psi derivative
        jac_disp = (1/(self.f0+self.dfs) +
                    1/self.f0/self.fm*self.dfs -
                    1/self.f0)
        jac[1, :, :] = 2*np.pi*jac_disp*D

# currently the official function
def vector_s3_test(param_vec0, bl_model, n_ant, data, weights, ref_ant=None):
    nparam = bl_model.get_nparam()
    param_vec = add_n_ref_zeroes(param_vec0, ref_ant, nparam)
    param_matrix = np.reshape(param_vec, (n_ant, nparam))
    data_shape = (n_ant, n_ant) + bl_model.get_grid_shape()
    model = np.zeros(data_shape, dtype=np.complex)
    # We loop over upper triangular baseline pairs and calculate the model for each
    for i, j in bli.triangle(n_ant):
        dparam = param_matrix[j]-param_matrix[i]
        model[i, j, :, :] = bl_model.make_model(*dparam)
    dvc = weights*(model - data.filled(0+0j))
    # xs
    sz = bl_model.get_grid_size()
    # we make a single real array to hold the vector the library
    # routines use to calculate xi-squared.
    dv = np.zeros((2*n_ant*n_ant*sz,), dtype=np.float)
    # Then we put stuff into a single big array.
    for l, (i, j) in enumerate(itertools.product(range(n_ant),
                                                 range(n_ant))):
        sr = slice((2*l+0)*sz, (2*l+1)*sz)
        si = slice((2*l+1)*sz, (2*l+2)*sz)
        d = np.ndarray.flatten(dvc[i, j])
        dv[sr] = d.real
        dv[si] = d.imag
    return dv

# the official jacobian function
def matrix_j_s3(param_vec0, bl_model, n_ant, data, weights, ref_ant):
    nparam = bl_model.get_nparam()
    param_vec = add_n_ref_zeroes(param_vec0, ref_ant, nparam)
    param_matrix = np.reshape(param_vec, (n_ant, nparam))
    # The jacobian is real and has one row for each parameter for each antenna
    jac = np.zeros((nparam*(n_ant-1), 2*utils.flatsize(data)), dtype=np.float)
    # skip ref_ant
    loop_range = range(0, ref_ant) + range(ref_ant+1, n_ant)
    for p_antenna, p in enumerate(loop_range):
        for l, (i, j) in enumerate(itertools.product(range(n_ant),
                                                     range(n_ant))):
            # p must be equal to exactly one of the antennas i and j
            # for there to be non-zero jacobian entries.
            if (p == i and p == j) or (p != i and p != j):
                continue
            else:
                sz = bl_model.get_grid_size()
                sr = slice((2*l+0)*sz, (2*l+1)*sz)
                si = slice((2*l+1)*sz, (2*l+2)*sz)
                dparam = param_matrix[j]-param_matrix[i]
                w = weights[i, j]
                # Calculate the jacobian for baseline (i,j)
                jac_p = bl_model.make_jacobian(*dparam)
                # And then squish it into the appropriate  part of the
                # big jacobian array
                sgn = -1 if i == p else 1 # j == p
                for k in range(nparam):
                    flat_jac_k = np.ndarray.flatten(sgn*w*jac_p[k])
                    # print "Shape ", jac_p.shape, jac.shape, data.shape
                    jac[nparam*p_antenna+k, sr] = flat_jac_k.real
                    jac[nparam*p_antenna+k, si] = flat_jac_k.imag
    return jac

#### End of dispersion-only hacking

def sigma_p(param_vec, bl_model, n_ant, data, weights, ref_ant=None):
    delta = vector_s3_test(param_vec, bl_model, n_ant, data, weights, ref_ant)
    jt = matrix_j_s3(param_vec, bl_model, n_ant, data, weights, ref_ant)
    n = len(param_vec)
    ngridpoints = reduce(operator.mul, data.shape[-2:])
    m = n_ant * (n_ant-1)/2 * ngridpoints
    inv_w_squared = 1/float(m-n+1) * np.linalg.norm(delta)**2
    cov = np.dot(jt, np.transpose(jt))
    covinv = np.linalg.inv(cov)
    d = np.sqrt(inv_w_squared*np.diag(covinv))
    return d
