import sys, os.path, json, numpy as np
from numpy import random
from math import *
from sklearn.metrics import mean_squared_error
from scipy.stats.distributions import chi2
from sklearn.linear_model import LinearRegression


def generate_data(
        img_size: tuple, 
        line_params: tuple,
        n_points: int, 
        sigma: float, 
        inlier_ratio: float
) -> np.ndarray:
    a, b, c = line_params
    line_points_number = round(n_points*inlier_ratio)
    outline_points_number = n_points - line_points_number
    line_noise_x = random.normal(scale=np.sqrt(sigma),size=(line_points_number,1))
    line_noise_y = random.normal(scale=np.sqrt(sigma),size=(line_points_number,1))
    x = np.linspace(-img_size[0]//2,img_size[0]//2,line_points_number)[:, np.newaxis]
    x = x + line_noise_x
    y = -(a*x+c)/b + line_noise_y
    line_y = y[y<img_size[0]//2][:, np.newaxis]
    line_x = x[y<img_size[0]//2][:, np.newaxis]
    outline_noise = random.uniform(-1,1,(outline_points_number, 2))
    outline_noise_x = outline_noise[:,0][:, np.newaxis]*(img_size[0]//2)
    outline_noise_y = outline_noise[:,1][:, np.newaxis]*(img_size[1]//2)
    X = np.concatenate((line_x, outline_noise_x))
    Y = np.concatenate((line_y, outline_noise_y))
    data = np.hstack((X,Y))
    return data

def compute_ransac_threshold(
        alpha: float, 
        sigma: float
) -> float:
    return np.sqrt(chi2.ppf(alpha, df=2)*sigma)

def compute_ransac_iter_count(
        conv_prob: float, 
        inlier_ratio: float
) -> int:
    m = 2
    return int(round(log(1-conv_prob)/log(1 - inlier_ratio**m)))

def compute_line_ransac(
        data: np.ndarray, 
        threshold: float, 
        iter_count: int
) -> tuple:
    iterations = 0
    best_fit = None
    best_error = 1e6
    n = 2
    best_inliers = 0
    while iterations < iter_count:
        perm = np.random.permutation(data.shape[0])
        maybe_inliers = data[perm[:n]]
        not_maybe_inliers = data[perm[n:]]
        x = maybe_inliers[:,0] 
        y = maybe_inliers[:,1]
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        also_inliers = []
        for point in not_maybe_inliers:
            y_pred = model.predict(point[0].reshape(-1, 1))
            error = mean_squared_error(point[1].reshape(-1, 1), [y_pred])
            if error < threshold**2:
                also_inliers.append(point)
        if len(also_inliers) < 2:
            continue
        concat_data = np.concatenate((also_inliers, maybe_inliers))
        better_model = LinearRegression().fit(concat_data[:,0].reshape(-1, 1), concat_data[:,1].reshape(-1, 1))
        y_pred_concat = better_model.predict(concat_data[:,0].reshape(-1, 1))
        current_error = mean_squared_error(concat_data[:,0].reshape(-1, 1), y_pred_concat)

        if len(concat_data) > best_inliers or (
            len(concat_data) == best_inliers 
            and current_error < best_error
        ):
            best_inliers = len(concat_data)
            best_fit = model
            best_error = current_error
        iterations += 1
    # return [params["b"] * best_fit.coef_[0], params["b"], params["b"] * best_fit.intercept_]
    return [best_fit.coef_[0], -1, best_fit.intercept_]


def detect_line(params: dict) -> tuple:
    data = generate_data(
        (params['w'], params['h']),
        (params['a'], params['b'], params['c']),
        params['n_points'], params['sigma'], params['inlier_ratio']
    )
    threshold = compute_ransac_threshold(
        params['alpha'], params['sigma']
    )
    iter_count = compute_ransac_iter_count(
        params['conv_prob'], params['inlier_ratio']
    )
    detected_line = compute_line_ransac(data, threshold, iter_count)
    return detected_line


def main():
    assert len(sys.argv) == 2
    params_path = sys.argv[1]
    assert os.path.exists(params_path)
    with open(params_path) as fin:
        params = json.load(fin)
    assert params is not None

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used

    sigma - Gaussian noise
    alpha - probability of point is an inlier

    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    detected_line = detect_line(params)
    print(detected_line)


if __name__ == '__main__':
    main()
