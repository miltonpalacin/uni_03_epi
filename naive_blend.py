import numpy as np

def naive_array(source_ini, mask_ini, target_ini, dim_ini):
    source = source_ini.copy()
    mask = mask_ini.copy()
    target = target_ini.copy()
    dim = dim_ini.copy()
    fusion = target[dim[0]:dim[1], dim[2]:dim[3], :] * (1 - mask) + source * mask
    target[dim[0]:dim[1], dim[2]:dim[3], :] = fusion
    return fusion, target
