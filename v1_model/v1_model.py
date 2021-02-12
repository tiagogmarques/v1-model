
from collections import OrderedDict
from .modules import V1_Model
from .params import generate_gabor_param, cadena_gabor_param
import numpy as np
import functools


def v1_model(sf_corr=0.75, sf_max=9, sf_min=0, fixed_ssi=0, fixed_sd=0, rand_param=False, gabor_seed=0,
             simple_channels=128, complex_channels=128, cadena=0,
             o_d=1, k_inh_mult=0, sig_inh_mult=1,
             noise_mode=None, noise_scale=0.35, noise_level=0.07, k_exc=25, cs_ratio=1,
             image_size=224, fov=8, ksize=25, stride=4):

    if k_inh_mult == 0:
        ksize_div = 1
    else:
        ksize_div = 41

    out_channels = simple_channels + complex_channels

    if cadena == 1:
        print('Generating Cadena GFB')
        sf, theta, phase, nx, ny, ssi = cadena_gabor_param()
    else:
        sf, theta, phase, nx, ny, ssi = generate_gabor_param(simple_channels, complex_channels, gabor_seed, rand_param,
                                                             sf_corr, sf_max, sf_min, fixed_ssi)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'cs_ratio': cs_ratio, 'ksize': ksize, 'stride': stride}
    div_norm_params = {'o_d': o_d, 'k_inh_mult': k_inh_mult, 'sig_inh_mult': sig_inh_mult}


    if cadena == 1:
        model_suffix = 'cadena'+'_seed'+str(gabor_seed)+'_sc'+str(simple_channels)+'_cc'+str(complex_channels)+\
                       'cs_ratio'+str(cs_ratio)+'_FoV-'+str(fov)
    else:
        model_suffix = 'seed'+str(gabor_seed)+'_sfcorr'+str(sf_corr)+'_sc'+str(simple_channels)+\
                       '_cc'+str(complex_channels)+'_fssi'+str(fixed_ssi)+'_kim'+str(k_inh_mult)+\
                       '_fsi'+str(fixed_sd)+'_sim'+str(sig_inh_mult)+'_cs_ratio'+str(cs_ratio)+'_FoV-'+str(fov)

    # Conversions
    ppd = image_size / fov
    ppd_div = ppd / stride

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta / 180 * np.pi
    phase = phase / 180 * np.pi
    if fixed_sd == 1:
        sig_inh = np.ones_like(sigx) * k_inh_mult / ppd_div
    else:
        sig_inh = (sigx+sigy)/2 / stride * sig_inh_mult
    k_inh = (1 / (1 - ssi) - 1) * k_inh_mult

    model = V1_Model(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase, k_inh=k_inh, sig_inh=sig_inh,
                     simple_channels=simple_channels, complex_channels=complex_channels,
                     k_exc=k_exc, cs_ratio=cs_ratio, o_d=o_d,
                     noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                     ksize=ksize, ksize_div=ksize_div, stride=stride, input_size=image_size)

    model.identifier = 'gfb_' + model_suffix


    model.params = OrderedDict()
    model.params.fov = fov
    model.params.simple_channels = simple_channels
    model.params.complex_channels = complex_channels
    model.params.rand_param = rand_param
    model.params.sf_corr = sf_corr
    model.params.seed = gabor_seed
    model.params.k_exc = k_exc
    model.params.k_inh_mult = k_inh_mult
    model.params.sig_inh_mult = sig_inh_mult

    model.params.cs_ratio = cs_ratio
    model.params.o_d = o_d
    model.params.image_size = image_size
    model.params.ksize = ksize
    model.params.ksize_div = ksize_div
    model.params.stride = stride

    model.gabor_params = gabor_params
    model.arch_params = arch_params
    model.div_norm_params = div_norm_params

    model.eval()

    return model


def activations_wrapper(model):
    from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
    image_size = model.params.image_size
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size,
                                      normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    wrapper = PytorchWrapper(identifier=model.identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper


