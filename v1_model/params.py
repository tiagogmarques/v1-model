
import numpy as np
from .utils import sample_dist
import scipy.stats as stats


def generate_gabor_param(n_sc, n_cc, seed=0, rand_flag=False, sf_corr=0, sf_max=12, sf_min=0, fixed_ssi=0):
    features = n_sc + n_cc

    # Generates random sample
    np.random.seed(seed)

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    if rand_flag:
        print('Uniform gabor parameters')
        ori_bins = np.array([0, 180])
        ori_dist = np.array([1])

        nx_bins = np.array([0.1, 10**0.2])
        nx_dist = np.array([1])

        ny_bins = np.array([0.1, 10**0.2])
        ny_dist = np.array([1])

        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8, 11.2])
        sf_s_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1, 1])
        sf_c_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1, 1])

        ssi_bins = np.array([0, 1.0])
        ssi_dist = np.array([1])

    else:
        print('Neuronal distributions gabor parameters')
        # DeValois 1982a
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        # ori_dist = np.array([77, 49, 66, 54])
        ori_dist = np.array([110, 83, 100, 92])
        ori_dist = ori_dist / ori_dist.sum()

        # Ringach 2002b
        nx_bins = np.logspace(-1, 0.2, 6, base=10)
        ny_bins = np.logspace(-1, 0.2, 6, base=10)
        n_joint_dist = np.array([[2.,  0.,  1.,  0.,  0.],
                                 [8.,  9.,  4.,  1.,  0.],
                                 [1.,  2., 19., 17.,  3.],
                                 [0.,  0.,  1.,  7.,  4.],
                                 [0.,  0.,  0.,  0.,  0.]])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_dist_marg = n_joint_dist / n_joint_dist.sum(axis=1, keepdims=True)

        # DeValois 1982b
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8, 11.2])
        # sf_s_dist = np.array([4, 4, 8, 25, 33, 26, 28, 12, 8])
        # sf_c_dist = np.array([0, 0, 9, 9, 7, 10, 23, 12, 14])
        sf_s_dist = np.array([8, 14, 20, 43, 40, 44, 31, 16, 8])
        sf_c_dist = np.array([2, 1, 11, 14, 22, 23, 32, 15, 16])

        # Cavanaugh 2002
        ssi_bins = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ssi_dist = np.array([66, 44, 38, 23, 16])
        ssi_dist = ssi_dist / ssi_dist.sum()

    phase = sample_dist(phase_dist, phase_bins, features)
    ori = sample_dist(ori_dist, ori_bins, features)
    # ori[ori < 0] = ori[ori < 0] + 180
    if fixed_ssi == 1:
        ssi = np.ones_like(ori) * 0.5
    else:
        ssi = sample_dist(ssi_dist, ssi_bins, features)

    sfmax_ind = np.where(sf_bins <= sf_max)[0][-1]
    sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

    sf_bins = sf_bins[sfmin_ind:sfmax_ind+1]
    sf_s_dist = sf_s_dist[sfmin_ind:sfmax_ind]
    sf_c_dist = sf_c_dist[sfmin_ind:sfmax_ind]

    sf_s_dist = sf_s_dist / sf_s_dist.sum()
    sf_c_dist = sf_c_dist / sf_c_dist.sum()

    if rand_flag:
        if n_sc > 0:
            sf_s = sample_dist(sf_s_dist, sf_bins, n_sc, scale='log2')
        else:
            sf_s = np.array([])
        if n_cc > 0:
            sf_c = sample_dist(sf_c_dist, sf_bins, n_cc, scale='log2')
        else:
            sf_c = np.array([])
        sf = np.concatenate((sf_s, sf_c))

        nx = sample_dist(nx_dist, nx_bins, features, scale='log10')
        ny = sample_dist(ny_dist, ny_bins, features, scale='log10')
    else:
        cov_mat = np.array([[1, sf_corr], [sf_corr, 1]])

        if n_sc > 0:
            samps = np.random.multivariate_normal([0, 0], cov_mat, n_sc)
            samps_cdf = stats.norm.cdf(samps)

            nx_s = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
            nx_s = 10**nx_s

            ny_samp = np.random.rand(n_sc)
            ny_s = np.zeros(n_sc)
            for samp_ind, nx_samp in enumerate(nx_s):
                bin_id = np.argwhere(nx_bins < nx_samp)[-1]
                ny_s[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                                 np.log10(ny_bins))
            ny_s = 10**ny_s

            sf_s = np.interp(samps_cdf[:,1], np.hstack(([0], sf_s_dist.cumsum())), np.log2(sf_bins))
            sf_s = 2**sf_s
        else:
            nx_s = np.array([])
            ny_s = np.array([])
            sf_s = np.array([])

        if n_cc > 0:
            samps = np.random.multivariate_normal([0, 0], cov_mat, n_cc)
            samps_cdf = stats.norm.cdf(samps)

            nx_c = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
            nx_c = 10**nx_c

            ny_samp = np.random.rand(n_cc)
            ny_c = np.zeros(n_cc)
            for samp_ind, nx_samp in enumerate(nx_c):
                bin_id = np.argwhere(nx_bins < nx_samp)[-1]
                ny_c[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                                 np.log10(ny_bins))
            ny_c = 10**ny_c

            sf_c = np.interp(samps_cdf[:,1], np.hstack(([0], sf_c_dist.cumsum())), np.log2(sf_bins))
            sf_c = 2**sf_c
        else:
            nx_c = np.array([])
            ny_c = np.array([])
            sf_c = np.array([])

        nx = np.concatenate((nx_s, nx_c))
        ny = np.concatenate((ny_s, ny_c))
        sf = np.concatenate((sf_s, sf_c))

    return sf, ori, phase, nx, ny, ssi


def cadena_gabor_param():

    phase = np.array([0, 90, 0])  # 2 simple cells and complex cell
    ori = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # 8 orientations
    nx = np.array([0.25, 0.5, 0.75])  # 3 different sizes

    sf = []
    for n in nx:
        sf.append(n/(0.17/4))
        sf.append(n/(0.31/4))
        sf.append(n/(0.60/4))
    sf = np.array(sf)  # 9 sf, 3 per size

    phase = np.repeat(phase, 3 * 3 * 8)  # repeated to account for 3 sizes, 3 sf, and 8 orientations

    nx = np.repeat(nx, 3 * 8)  # repeated to account for 3 sf and 8 orientations

    sf = np.repeat(sf, 8)  # repeated to account for 8 orientations

    ori = np.tile(ori, 3 * 3 * 3)  # tiled to account for 3 sf, 3 sizes, and 3 phases
    sf = np.tile(sf, 3)  # tiled to account for 3 phases
    nx = np.tile(nx, 3)  # tiled to account for 3 phases
    ny = nx

    ssi = np.zeros_like(ori)

    return sf, ori, phase, nx, ny, ssi
