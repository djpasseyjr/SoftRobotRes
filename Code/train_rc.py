from rescomp import DrivenResComp
import numpy as np
from math import floor
from scipy import io
from scipy.interpolate import CubicSpline
import pickle

TRAIN_PER = .9
TOL = 1

def train_rc(*args, save_file="train_rc.pkl"):
    n, mean_degree, sigma, gamma, delta, ridge_alpha, window, overlap, ntrials, matlab_files = args
    results = results_dict(*args)
    # Make data
    ts = []
    us = []
    u_drives = []
    for mf in matlab_files:
        t, u, u_drive = interp_data(mf)
        ts.append(t)
        us.append(u)
        u_drives.append(u_drive)
    # Test data is taken from the last file
    tr_idx = floor(TRAIN_PER * len(t))
    ts[-1] = t[:tr_idx]
    test_t = t[tr_idx:]
    # Batch train rc
    for i in range(ntrials):
        rc = DrivenResComp(
            # Variable parameters
            res_sz=n,
            connect_p=mean_degree / n,
            sigma=sigma,
            gamma=gamma,
            delta=delta,
            ridge_alpha=ridge_alpha,
            # Fixed parameters
            sparse_res=True,
            uniform_weights=True,
            spect_rad=1.0,
            signal_dim=18,
            drive_dim=6
        )
        err = rc.fit_batch(ts, us, u_drives, window, overlap=overlap)
        results[i]["err"] = err
        results[i]["rc"] = rc
        # Predict
        pre = rc.predict(test_t, u_drive)
        results[i]["acc_duration"] = how_long_accurate(us[-1](test_t).T, pre, tol=TOL)
        pickle.dump(results, open(save_file, 'wb'))


def results_dict(*args):
    n, mean_degree, sigma, gamma, delta, ridge_alpha, window, overlap, ntrials, matlab_files = args
    return { i : {"n" : n,
            "mean_degree" : mean_degree,
            "sigma" : sigma,
            "gamma" : gamma,
            "delta" : delta,
            "ridge_alpha" : ridge_alpha,
            "window" : window,
            "overlap" : overlap,
            "matlab_files" : matlab_files
            } for i in range(ntrials)}

def make_u(t, sig_data):
    return CubicSpline(t, sig_data.T)

def interp_data(fname):
    data = io.loadmat(fname)
    q = data['q'].T
    pref = data["pref"].T
    pest = data['pest'].T
    t = data['t'][0]
    qd = data['qd'].T
    # Stack together pressure, position, and velocity
    signal = np.vstack((pest, q, qd))
    u = make_u(t, signal)
    u_drive = make_u(t, np.reshape(pref, (6,len(t))))
    return t, u, u_drive

def how_long_accurate(u, pre, tol=1):
    """ Find the first i such that ||u_i - pre_i||_2 > tol """
    for i in range(u.shape[1]):
        dist = np.sum((u[:,i] - pre[:,i])**2)**.5
        if dist > tol:
            return i
    return u.shape[1]

def test_train_rc():
    n = 500
    mean_degree = 30
    sigma = .1
    gamma = 5
    delta = .1
    ridge_alpha = .001
    window = 1
    overlap = .9
    ntrials = 2
    matlab_files = ["bellows_arm_whitened.mat"]
    train_rc(n, mean_degree, sigma, gamma, delta, ridge_alpha, window, overlap, ntrials, matlab_files)
    
    d = pickle.load(open("train_rc.pkl", "rb"))
    assert d[0]["err"] != d[1]["err"]
    assert d[0]["sigma"] == sigma
