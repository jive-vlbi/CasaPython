import operator, itertools, logging
import scipy.optimize, numpy as np
import utils, bli

def add_ref_zeroes(l, ref_ant):
    l.insert(4*ref_ant, 0.0)
    l.insert(4*ref_ant+1, 0.0)
    l.insert(4*ref_ant+2, 0.0)
    l.insert(4*ref_ant+4, 0.0)
    v = np.array(l)
    return v

def make_some_s3_noise(grid_shape, n_antennas, cycles=1):
    shp = (n_antennas, n_antennas) + grid_shape 
    res = np.zeros(shp, dtype=np.complex)
    for i,j in bli.triangle(n_antennas):
        randoms = np.random.random(grid_shape)
        noise = np.exp(2*np.pi*1j*cycles*(randoms-1))
        res[i,j] = noise
    return res

def gen_bl_model(dpsi, dr, dtau, ddisp, Ts, Fs, freqs):
    # We add a final argument for the sky-frequencies.
    Fave = np.mean(freqs)
    F0 = np.min(freqs)
    ph_disp = ddisp*(Fs+F0) + (ddisp/Fave**2)*Fs - ddisp/F0
    result = (np.exp(1j*dpsi) *
              np.exp(1j*2*np.pi*dr*Ts) *
              np.exp(1j*2*np.pi*dtau*Fs)
              # * np.exp(1j*2*np.pi*ph_disp)
    ) + 1e200*(ddisp)
    return result

def gen_model_s3(v0, Ts, Fs, freqs, n, ref_ant=None):
    # casalog.post("gen_model_s3".format(v0), "DEBUG")
    if ref_ant != None:
        v = add_ref_zeroes(list(v0), ref_ant)
    else:
        v = v0
    data_shape = (n, n) + Ts.shape
    param = np.reshape(v, (n, 4))
    res = np.zeros(data_shape, dtype=np.complex)
    for i, j in bli.triangle(n):
        dpsi, dr, dtau, ddisp = param[j]-param[i]
        res[i, j, :, :] = gen_bl_model(dpsi, dr, dtau, ddisp, Ts, Fs, freqs)
    return res

def fun_s3(v, Ts, Fs, n, data, weights, ref_ant=None):
    dv = (gen_model_s3(v, Ts, Fs, n, ref_ant) - data)
    s = np.ndarray.flatten(np.abs(weights*dv)**2)
    return np.sum(s)

# currently the official function
def vector_s3_test(v, Ts, Fs, freqs, nantennas, data, weights, ref_ant=None):
    sz = utils.flatsize(Ts)
    dv = np.zeros((2*nantennas*nantennas*sz,), dtype=np.float) 
    dvc = weights*(gen_model_s3(v, Ts, Fs, freqs, nantennas, ref_ant) - data.filled(0+0j))
    for l, (i, j) in enumerate(itertools.product(range(nantennas),
                                                 range(nantennas))):
        sr = slice(2*l*sz, (2*l+1)*sz)
        si = slice((2*l+1)*sz, (2*l+2)*sz)
        d = np.ndarray.flatten(dvc[i, j])
        dv[sr] = d.real
        dv[si] = d.imag
    return dv

# FIXME! Jacobian must also have a dispersive parameter!
def matrix_j_s3(v, Ts, Fs, nantennas, data, weights, ref_ant=None):
    if ref_ant != None:
        jac = np.zeros((4*(nantennas-1), 2*utils.flatsize(data)), dtype=np.float)
        loop_range = range(0, ref_ant) + range(ref_ant+1, nantennas) 
        v = add_ref_zeroes(list(v), ref_ant)
    else:
        loop_range = range(nantennas)
        jac = np.zeros((4*(nantennas), 2*utils.flatsize(data)), dtype=np.float)        
    param = np.reshape(v, (nantennas, 4))
    # skip ref_ant
    # Note that this fails if there isn't one!
    for pind, p in enumerate(loop_range):
        for l, (i, j) in enumerate(itertools.product(range(nantennas),
                                                     range(nantennas))):
            if (p == i and p == j) or (p != i and p != j):
                continue
            else:
                sz = utils.flatsize(Ts)
                sr = slice(2*l*sz, (2*l+1)*sz)
                si = slice((2*l+1)*sz, (2*l+2)*sz)
                dpsi, dr, dtau, ddisp = param[j]-param[i]
                sgn = -1 if i == p else 1 # j == p
                D = weights[i,j]*1j*(gen_bl_model(dpsi, dr, dtau, Ts, Fs))
                psic = np.ndarray.flatten(sgn*D)
                jac[4*pind, sr] = psic.real
                jac[4*pind, si] = psic.imag
                rc = np.ndarray.flatten(sgn*2*np.pi*Ts*D)        # r_val 
                jac[4*pind+1, sr] = rc.real
                jac[4*pind+1, si] = rc.imag
                tauc = np.ndarray.flatten(sgn*2*np.pi*Fs*D)      # tau_val
                jac[4*pind+2, sr] = tauc.real
                jac[4*pind+2, si] = tauc.imag
    return jac

def sigma_p(v, Ts, Fs, nantennas, data, weights, ref_ant=None):
    delta = vector_s3(v, Ts, Fs, nantennas, data, weights, ref_ant)
    jt = matrix_j_s3(v, Ts, Fs, nantennas, data, weights, ref_ant)
    n = len(v)
    ngridpoints = reduce(operator.mul, data.shape[-2:])
    m = nantennas * (nantennas-1)/2 * ngridpoints
    inv_w_squared = 1/float(m-n+1) * np.linalg.norm(delta)**2
    casalog.post("inv_w_squared {} m {}".format(inv_w_squared,  m), "DEBUG")
    cov = np.dot(jt, np.transpose(jt))
    covinv = np.linalg.inv(cov)
    d = np.sqrt(inv_w_squared*np.diag(covinv))
    return d

def minimise_wrapper(fn, v, **kvargs):
    ref_ant = kvargs['args'][-1]
    v_dash, refs = param.remove_ref(v, ref_ant)
    casalog.post("[optimise_wrapper] refs {}".format(refs), "DEBUG")
    casalog.post("[minimise_wrapper] v_dash {}".format(v_dash), "DEBUG")
    res_t = scipy.optimize.minimize(fn, v_dash, **kvargs)
    casalog.post("[minimise_wrapper] res_t".format(res_t), "DEBUG")
    sol_arr = res_t.x
    casalog.post("[minimise_wrapper] sol_arr".format(sol_arr), "DEBUG")
    sol_out = list(sol_arr)
    sol = param.restore_ref(sol_out, refs, ref_ant)
    casalog.post("[minimise_wrapper] sol_out".format(sol_out), "DEBUG")
    casalog.post("[minimise_wrapper] sol".format(sol), "DEBUG")
    return np.array(sol), res_t

def minimise_wrapper_lm(fn, v, **kvargs):
    ref_ant = kvargs['args'][-1]
    v_dash, refs = param.remove_ref(v, ref_ant)
    res_t = scipy.optimize.leastsq(
        lsqrs.vector_s3_test, v_dash,
        full_output=1,
        args=(anffd.tgrid0/r_scale, anffd.fgrid0/tau_scale,
              len(antennas2), anffd.data, anffd.weights, ref_antenna),
        col_deriv = True,
        Dfun=lsqrs.matrix_j_s3,
        ftol = 1e-1,
        maxfev=5
    )
    sol_arr = res_t[0]
    sol_out = list(sol_arr)
    sol = restore_ref(sol_out, refs, ref_ant)
    return np.array(sol), res_t


if __name__ == '__main__':
    import numpy as np
    fs = np.linspace(0, 16e6, 32, False)
    ts = np.arange(0.0, 60, 2)
    Ts, Fs = np.meshgrid(ts, fs)
    params = [2.0, 2e-4, 1.1e-09,
              1.0, 3e-4, 2.1e-09,
              0.0, 4e-4, 3.1e-09]
    data2 = lsqrs.gen_model_s3(params, Ts, Fs, 3)
    weights = np.ones(data2.shape)
    lsqrs.minimise_wrapper(lsqrs.fun_s3,
                           [2.5, 2.0e-4, 1.1e-09,
                            1.5, 3.5e-4, 3.5e-09,
                            0.0, 4.0e-4, 3.1e-09],
                           args=(Ts, Fs, 3, data2*noise_s3, weights, 2), method='TNC', jac=lsqrs.jac_s3,
                           bounds=[(-10, +10), (-1e-3, 1e-3), (-1e-8, 1e-8),
                                   (-10, +10), (-1e-3,1e-3), (-1e-8, 1e-8)],
                           options={'ftol':1e-20})
    
