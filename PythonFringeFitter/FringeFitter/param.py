def scale_params(params, scales):
    # if we have an array we can't use its type as a constructor;
    # for lists, tuple and other iterables we can.
    t0 = type(params)
    t = {np.ndarray: np.array}.get(t0, t0)
    params2 = list(params)[:]
    for i in range(4):
        params2[i::4] = [x*scales[i] for x in params2[i::4]]
    return t(params2)

def get_phases(params): return params[0::4]
def get_rates(params):  return params[1::4]
def get_delays(params): return params[2::4]
def get_disps(params):  return params[3::4]

def remove_ref(v, ref_ant):
    n = len(v)/4
    v_dash = v[:] # make a copy to mutate
    disp_ref = v_dash.pop(4*ref_ant+3)
    tau_ref =  v_dash.pop(4*ref_ant+2)
    r_ref =    v_dash.pop(4*ref_ant+1)
    psi_ref =  v_dash.pop(4*ref_ant+0)
    for i in range(n-1):
        v_dash[4*i+0] -= psi_ref
        v_dash[4*i+1] -= r_ref
        v_dash[4*i+2] -= tau_ref
        v_dash[4*i+3] -= disp_ref
    return v_dash, (psi_ref, r_ref, tau_ref, disp_ref)

def restore_ref(params0, refs, ref_ant):
    n = len(params0)/4+1
    # Make a copy since it might be array rather than list
    sol = params0[:] 
    (psi_ref, r_ref, tau_ref, disp_ref) = refs
    sol.insert(4*ref_ant+0, 0.0)
    sol.insert(4*ref_ant+1, 0.0)
    sol.insert(4*ref_ant+2, 0.0)
    sol.insert(4*ref_ant+3, 0.0)
    for i in range(n):
        sol[4*i+0] += psi_ref
        sol[4*i+1] += r_ref
        sol[4*i+2] += tau_ref
        sol[4*i+3] += disp_ref
    return sol

def remove_non_ref(v, ant):
    v_dash = v[:] # make a copy to mutate
    disp = v_dash.pop(4*ant+3)
    tau =  v_dash.pop(4*ant+2)
    r =    v_dash.pop(4*ant+1)
    psi =  v_dash.pop(4*ant+0)
    return v_dash, (psi, r, tau, disp)

def restore_non_ref(params0, stored_params, ant):
    # We put values for phase, delay and rate for a given antenna
    # contiguously in an array.
    (psi, r, tau, disp) = stored_params
    # Make a copy since it might be array rather than list
    # also out of politeness
    params = params0[:] 
    params.insert(4*ant+0, psi)
    params.insert(4*ant+1, r)
    params.insert(4*ant+2, tau)
    params.insert(4*ant+3, disp)
    return params

def get_antenna_parameters(params, i):
    return tuple(params[4*i:4*(i+1)])

def add_ref_parameters(params0, rp, ref_ant):
    params = params0[:]
    n = len(params)/4
    (psi_ref, r_ref, tau_ref, disp_ref) = rp
    for i in range(n):
        if i == ref_ant:
            continue
        else:
            params[4*i+0] += psi_ref
            params[4*i+1] += r_ref
            params[4*i+2] += tau_ref
            params[4*i+3] += disp_ref
    return params

def remove_antennas(v, ants0):
    ants = reversed(sorted(ants0))
    vals = []
    for a in ants:
        v, p = remove_non_ref(v, a)
        # we list the results backwards because we reversed the list.
        vals.insert(0, p)
    return v, vals

def restore_antennas(params, frozen_params, ants0):
    n_ants = len(ants0) + len(params)/4
    flags = [False for i in range(n_ants)]
    ants = sorted(ants0)
    for a, p in zip(ants, frozen_params):
        flags[a] = True
        params = restore_non_ref(params, p, a)
    return flags, params

