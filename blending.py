import poisson_blending as pbnd
import scipy.sparse.linalg as linalg
import numpy as np


# naive resolution
def naive_array(source_ini, mask_ini, target_ini, dim_ini):
    source = source_ini.copy()
    mask = mask_ini.copy()
    target = target_ini.copy()
    dim = dim_ini.copy()
    fusion = target[dim[0]:dim[1], dim[2]:dim[3], :] * (1 - mask) + source * mask
    target[dim[0]:dim[1], dim[2]:dim[3], :] = fusion
    return fusion, target


# linear least squares solver
def linlsq_solver(A, b, dim):
    x = linalg.spsolve(A.tocsc(), b)
    return np.reshape(x, (dim[0], dim[1]))


# stitches poisson equation solution with target
def stitch_images(source, target, dim):
    target[dim[0]:dim[1], dim[2]:dim[3], :] = source
    return target


# performs poisson blending
def blend_image(source_ini, mask_ini, target_ini, dim_ini, mix):
    source = source_ini.copy()
    mask = mask_ini.copy()
    target = target_ini.copy()
    dim = dim_ini.copy()
    equation_param = []
    ch_data = {}

    # construct poisson equation
    for ch in range(3):
        ch_data['source'] = source[:, :, ch]
        ch_data['mask'] = mask[:, :, ch]
        ch_data['target'] = target[:, :, ch]
        equation_param.append(pbnd.poisson_blending(ch_data, dim, mix))

    # solve poisson equation
    image_solution = np.empty_like(source)
    for i in range(3):
        image_solution[:, :, i] = linlsq_solver(equation_param[i][0], equation_param[i][1], source.shape)

    image_solution = stitch_images(image_solution, target, dim)

    return image_solution
