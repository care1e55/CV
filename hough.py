import sys, os.path, cv2, numpy as np
from typing import Tuple
from numpy import unravel_index


def gradient_img(img: np.ndarray) -> np.ndarray:
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)
    return magnitude


def hough_transform(
        img: np.ndarray, 
        theta: float, 
        rho: float
) -> Tuple[np.ndarray, list, list]:
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta))
    max_dist = np.ceil(np.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1])).astype(np.int)
    rhos = np.linspace(-max_dist, max_dist, 2*max_dist)
    accumulator = np.zeros((2*max_dist, thetas.shape[0]), dtype=np.int)
    y_idxs, x_idxs = np.nonzero(img)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for i in range(x_idxs.shape[0]):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(thetas.shape[0]):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + max_dist)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def get_lines(
        ht_map: np.ndarray, 
        n_lines: int,
        thetas: list, 
        rhos: list,
        min_delta_rho: float, 
        min_delta_theta: float
) -> list:
    lines_idx = np.argsort(ht_map.flatten())[::-1]
    result_lines = []
    lines_ = {"rhos": [], "thetas": []}
    for i in lines_idx:
        in_range = False
        rho_idx, theta_idx = unravel_index(i, ht_map.shape)
        for j in lines_["rhos"]:
            if (j-min_delta_rho < rhos[rho_idx] < j+min_delta_rho):
                in_range = True
                break
        if not in_range:
            for j in lines_["thetas"]:
                if (j-min_delta_theta < thetas[theta_idx] < j+min_delta_theta):
                    in_range = True
                    break
        if not in_range:
            lines_["rhos"].append(rhos[rho_idx]),
            lines_["thetas"].append(thetas[theta_idx])
        if len(lines_["rhos"]) == n_lines:
            break
    
    for i in zip(lines_["rhos"], lines_["thetas"]):
        k = - np.cos(i[1])/np.sin(i[1])
        b = i[0]/np.sin(i[1])
        result_lines.append((k, b))

    return result_lines


def main():
    assert len(sys.argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho, \
        n_lines, min_delta_rho, min_delta_theta = sys.argv[1:]

    theta = float(theta)
    assert theta > 0.0

    rho = float(rho)
    assert rho > 0.0

    n_lines = int(n_lines)
    assert n_lines > 0

    min_delta_rho = float(min_delta_rho)
    assert min_delta_rho > 0.0

    min_delta_theta = float(min_delta_theta)
    assert min_delta_theta > 0.0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    gradient = gradient_img(img.astype(float))

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(
        ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta
    )

    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write(f'{line[0]:.3f}, {line[1]:.3f}\n')


if __name__ == '__main__':
    main()
