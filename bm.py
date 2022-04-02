from time import time
import numpy as np
from numba import jit

np.set_printoptions(precision=8)

def clamped_statistics(states):
    nstates = states.shape[0]
    s_i = np.mean(states, axis=0)
    s_ij = (1 / nstates) * (states.T @ states)
    return s_i, s_ij

@jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@jit(nopython=True)  # TODO test whether nogil=True can break +=
def compute_statistics(states, weights, theta, Z):
    nstates, nneurons = states.shape
    s_i = np.zeros(nneurons)
    s_ij = np.zeros((nneurons, nneurons))
    for i in range(nstates):
        s = states[i]
        p = p_boltzmann(np.expand_dims(s, axis=0), weights, theta, Z)
        s_i += s * p
        s_ij += np.dot(np.expand_dims(s, axis=1), np.expand_dims(s, axis=0)) * p
    return s_i, s_ij

@jit(nopython=True)
def compute_stochastic_statistics(states):
    nstates, nneurons = states.shape
    s_i = np.zeros(nneurons)
    s_ij = np.zeros((nneurons, nneurons))
    for i in range(nstates):
        s = states[i]
        s_i += s
        s_ij += np.expand_dims(s, axis=1) @ np.expand_dims(s, axis=0)
    return s_i / nstates, s_ij / nstates



@jit(nopython=True)
def p_boltzmann(state, weights, theta, Z):
    assert state.shape[0] == 1, "state must have shape (1,n)"
    assert theta.shape[1] == 1, "theta must have shape (n,1)"
    first_term = np.dot(state, weights) @ state.T
    second_term = np.dot(state, theta)
    p = (1/Z) * np.exp(0.5 * first_term + second_term)
    return p[0,0]  # unpack array


@jit(nopython=True)
def sequential_dynamics(weights, theta, nstates=1000, burnin=2000):
    nneurons = theta.shape[0]
    nstates += burnin
    states = np.zeros((nstates, nneurons))
    state = np.random.random(nneurons)
    state[state>.5] = 1
    state[state<=.5] = -1
    coins = np.random.random(nstates)
    idx = np.random.randint(low=0, high=nneurons, size=nstates)
    # generate new states
    for i in np.arange(nstates):
        s = np.expand_dims(state, axis=1)
        probs = sigmoid((weights @ s) + theta)  # probabilities that s_i is 1 in t+1
        p = probs[idx[i]]
        prob = p[0]  # force casting to tf.float, otherwise numba will complain about data types
        if coins[i] < prob:
            state[idx[i]] = 1
            states[i] = state
        else:
            state[idx[i]] = -1
            states[i] = state
    return states[burnin:]


@jit(nopython=True)
def mean_field(m, weights, theta, precision=1e-13):
    '''
    compute mean field activation of neurons by fixed point iteration
    pass m vector under the assumption that mean activations change slowly between
    iterations of the learning iteration. Passing the previous mean field should
    therefore speed up convergence as opposed to random initialization every time
    '''
    assert m.shape[1] == 1, "mean field must have shape (n,1)"
    delta_m = 1
    m_new = np.zeros(m.shape)
    while delta_m > precision:
        m_new = np.tanh((weights @ m) + theta)
        delta_m = np.max(np.abs(m_new - m))
        m = m_new
    return m
        

@jit(nopython=True)
def calc_Z(states, weights, theta):
    nstates = states.shape[0]
    probs = np.zeros(nstates)
    nstates = states.shape[0]

    for i in range(nstates):
        s = np.expand_dims(states[i], axis=0)
        probs[i] = p_boltzmann(s, weights, theta, 1)
    return np.sum(probs)

@jit(nopython=True)
def log_likelihood(states, weights, theta, Z):
    ndata = states.shape[0]
    log_probs = np.zeros(ndata)
    for i in range(ndata):
        s = np.expand_dims(states[i], axis=0)
        log_probs[i] = np.log(p_boltzmann(s, weights, theta, Z))
    return np.sum(log_probs)


@jit(nopython=True)
def update_params(weights, theta, s_i_clamp, s_ij_clamp, s_i, s_ij, eta):
    dw = (s_ij_clamp - s_ij)
    # dw = (dw + dw.T) / 2
    dw -= np.diag(np.diag(dw))
    weights += eta * dw

    dt = (s_i_clamp - np.expand_dims(s_i, axis=1)) 
    theta += eta * dt
    return weights, theta, dw, dt

@jit(nopython=True)
def calc_F(weights, theta, m):
    assert theta.shape[1] == 1, "invalid theta in calc_F"
    assert m.shape[1] == 1, "invalid m in calc_F"

    term_1 = -0.5 * ( (m.T @ weights) @ m )
    term_2 = - (theta.T @ m)
    term_3 = 0.5 * np.sum((1 + m)*np.log(.5*(1 + m)) + (1 - m) * np.log(.5*(1 - m)))
    F = term_1 + term_2 + term_3
    return F


# @jit(nopython=True)
def fit_bm_deterministic(data, allstates, s_i_clamp, s_ij_clamp, eta_init, schedule=15000, init_scale=1):
    # init boltzmann machine
    nneurons = s_i_clamp.shape[0]
    theta = np.random.normal(size=(nneurons, 1)) * init_scale
    weights = np.random.normal(size=(nneurons,nneurons)) * init_scale
    weights = (weights + weights.T) *0.5  # symmetry
    weights -= np.diag(np.diag(weights))  # zero diagonal

    # compute statistics
    
    # bookkeping
    lls = list()

    # main loop
    max_weight_change = 1
    i = 0
    while max_weight_change > 1e-8:
        eta = eta_init * np.exp(-(i/schedule))
        Z = calc_Z(allstates, weights, theta)
        
        old_weights, old_theta = weights.copy(), theta.copy()

        s_i, s_ij = compute_statistics(data, weights, theta, Z)
        new_weights, new_theta, _, _ = update_params(weights, theta, s_i_clamp, s_ij_clamp, s_i, s_ij, eta)
        max_weight_change = np.max(np.abs(weights - old_weights))
        max_theta_change = np.max(np.abs(theta - old_theta))
        weights, theta = new_weights, new_theta
        
        lls.append(log_likelihood(data, weights, theta, Z))
        print(f" {i}: theta max {max_theta_change:.6e}, weights max {max_weight_change:.6e}, ll: {lls[-1]:.2f}, eta: {eta:.6e}", end='\r')
        i += 1

    return weights, theta, lls


# @jit(nopython=True)
def fit_bm_stochastic(data, allstates, s_i_clamp, s_ij_clamp, eta_init, nglauber=10000, schedule=10000, init_scale=1, compare_explicit=False, explicit_Z=False):
    '''
    fit boltzmann machine using stochastic sequential dynamics
    note that Z is computed explicitly (intractable for large nneurons) but is only used
    to assess convergence. We could approximate Z by sampling from the Boltzmann Distribution
    (not necessary for fitting the distribution)
    '''
    # init boltzmann machine
    nneurons = s_i_clamp.shape[0]
    theta = np.random.normal(size=(nneurons, 1)) * init_scale
    weights = np.random.normal(size=(nneurons,nneurons)) * init_scale
    weights = (weights + weights.T) *0.5  # symmetry
    weights -= np.diag(np.diag(weights))  # zero diagonal

    # compute statistics
    
    # bookkeping
    lls = list()

    if compare_explicit:
        max_weight_diffs = list()

    # main loop
    max_weight_change = 1
    i = 0
    while max_weight_change > 1e-8:
        eta = eta_init * np.exp(-(i/schedule))
        if explicit_Z:
            Z = calc_Z(allstates, weights, theta)
        else:
            m = np.random.normal(size=(nneurons,1)) * 0.01
            m = mean_field(m, weights, theta)
            F = calc_F(weights, theta, m)
            Z = np.exp(-F)
        
        states = sequential_dynamics(weights, theta, nstates=nglauber)
        old_weights, old_theta = weights.copy(), theta.copy()
        
        s_i, s_ij = compute_stochastic_statistics(states)
        if compare_explicit:
            new_weights, new_theta, dw_stoc, dt_stoc = update_params(weights, theta, s_i_clamp, s_ij_clamp, s_i, s_ij, eta)
            s_i_explicit, s_ij_explicit = compute_statistics(allstates, weights, theta, Z)
            _,_, dw_true, dt_true = update_params(weights, theta, s_i_clamp, s_ij_clamp, s_i_explicit, s_ij_explicit, eta)
            max_weight_diff = np.max(np.abs(dw_true - dw_stoc))
            max_weight_diffs.append(max_weight_diff)
        else:
            new_weights, new_theta, _, _ = update_params(weights, theta, s_i_clamp, s_ij_clamp, s_i, s_ij, eta)

        max_weight_change = np.max(np.abs(weights - old_weights))
        max_theta_change = np.max(np.abs(theta - old_theta))
        weights, theta = new_weights, new_theta
        lls.append(log_likelihood(data, weights, theta, Z))
        print(f" {i}: theta max {max_theta_change:.6e}, weights max {max_weight_change:.6e}, ll: {lls[-1]:.2f}, eta: {eta:.6e}", end='\r')
        i += 1
    if compare_explicit:
        return weights, theta, lls, max_weight_diffs
    else:
        return weights, theta, lls


# @jit(nopython=True)
def fit_bm_mean_field(data, allstates, s_i_clamp, s_ij_clamp, eta_init, schedule=1000, init_scale=0.1, compare_explicit=False, explicit_Z=True):

    '''
    fit boltzmann machine using stochastic sequential dynamics
    note that Z is computed explicitly (intractable for large nneurons) but is only used
    to assess convergence. We could approximate Z by sampling from the Boltzmann Distribution
    (not necessary for fitting the distribution)
    '''
    # init boltzmann machine
    nneurons = s_i_clamp.shape[0]
    theta = np.random.normal(size=(nneurons, 1)) * init_scale
    weights = np.random.normal(size=(nneurons,nneurons)) * init_scale
    weights = (weights + weights.T) *0.5  # symmetry
    weights -= np.diag(np.diag(weights))  # zero diagonal
    m = np.random.normal(size=(nneurons, 1)) * 0.1

    # bookkeping
    lls = list()

    if compare_explicit:
        max_weight_diffs = list()

    # main loop
    max_weight_change = 1
    i = 0
    while max_weight_change > 1e-8:
        eta = eta_init * np.exp(-(i/schedule))
        old_weights, old_theta = weights.copy(), theta.copy()
        
        m = mean_field(m, weights, theta)

        msqueeze = np.squeeze(m)
        chi = np.linalg.inv(np.diag(1 / (1-msqueeze**2)) - weights)
        s_ij = chi + (m @ m.T)
        
        # compute partition sum
        if explicit_Z:
            Z = calc_Z(allstates, weights, theta)
        else:
            F = calc_F(weights, theta, m)
            Z = np.exp(-F)

        if compare_explicit:
            new_weights, new_theta, dw_stoc, dt_stoc = update_params(weights, theta, s_i_clamp, s_ij_clamp, msqueeze, s_ij, eta)
            s_i_explicit, s_ij_explicit = compute_statistics(allstates, weights, theta, Z)
            _,_, dw_true, dt_true = update_params(weights, theta, s_i_clamp, s_ij_clamp, s_i_explicit, s_ij_explicit, eta)
            max_weight_diff = np.max(np.abs(dw_true - dw_stoc))
            max_weight_diffs.append(max_weight_diff)
        else:
            new_weights, new_theta, _, _ = update_params(weights, theta, s_i_clamp, s_ij_clamp, msqueeze, s_ij, eta)


        max_weight_change = np.max(np.abs(weights - old_weights))
        max_theta_change = np.max(np.abs(theta - old_theta))
        weights, theta = new_weights, new_theta
        lls.append(log_likelihood(data, weights, theta, Z))
        print(f" {i}: theta max {max_theta_change:.6e}, weights max {max_weight_change:.6e}, ll: {lls[-1]:.2f}, eta: {eta:.6e}", end='\r')
        i += 1
    if compare_explicit:
        return weights, theta, lls, max_weight_diffs
    else:
        return weights, theta, lls

def log_p(state, weights, theta, log_Z):
    assert state.shape[0] == 1, "state must have shape (1,n)"
    assert theta.shape[1] == 1, "theta must have shape (n,1)"
    first_term = np.dot(state, weights) @ state.T
    second_term = np.dot(state, theta)
    log_p =  -log_Z + 0.5 * first_term + second_term
    return log_p[0,0]  # unpack array



def fit_bm_mean_field_direct(data, s_i_clamp, s_ij_clamp, epsilon=1e-5):
    print(epsilon)
    nneurons = s_i_clamp.shape[0]
    m = s_i_clamp + 0.0000001
    C = s_ij_clamp - (s_i_clamp @ s_i_clamp.T)
    C_inv = np.linalg.inv(C + (np.eye(nneurons) * epsilon))
    weights = np.diag(1/(1-(m.squeeze()**2)))
    weights = weights - C_inv
    theta = np.arctanh(m) - (weights @ m)
    F = calc_F(weights, theta, m)
    print(f"F: {F}")
    log_Z = -F
    ndata = data.shape[0]
    log_probs = np.zeros(ndata)
    for i in range(ndata):
        s = np.expand_dims(data[i], axis=0)
        log_probs[i] = log_p(s, weights, theta, log_Z)
    ll = np.sum(log_probs)
    return weights, theta, ll
