import sys, os.path, cv2, numpy as np
import warnings
warnings.filterwarnings("ignore")


def otsu(img: np.ndarray) -> np.ndarray:
    histogram = np.histogram(img, 256)[0]
    variances = []
    for t in range(0,256):
        p0 = histogram[:t]
        p1 = histogram[t:]
        w0 = np.sum(p0)
        w1 = np.sum(p1)
        i0 = np.array([j for j in range(t)])
        i1 = np.array([j for j in range(t, 256)])
        m0 = np.sum(p0*i0)/w0
        m1 = np.sum(p1*i1)/w1
        v0 = np.sum(((i0-m0)**2)*p0)/w0
        v1 = np.sum(((i1-m1)**2)*p1)/w1
        cur_variance = v0*w0 + v1*w1
        if np.isnan(cur_variance):
            continue
        else:
            variances.append(cur_variance)
    threshold = np.argmin(variances)
    img_new = np.where(img > threshold, img, 0)
    img_new = np.where(img_new < threshold, img_new, 255)
    return img_new


def main():
    assert len(sys.argv) == 3
    src_path, dst_path = sys.argv[1], sys.argv[2]

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = otsu(img)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
