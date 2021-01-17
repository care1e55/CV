import sys, os.path, cv2, numpy as np
from tqdm import tqdm

# linear
def box_filter(img: np.ndarray, w: int, h: int) -> np.ndarray:
    div_h, div_w = float(h), float(w)
    shift_y = h // 2
    shift_x = w // 2
    img_copy = np.copy(img).astype(float)
    img_new = np.copy(img).astype(float)

    # Pad or ignore or bring complicated logic on edges. We choose ignore
    for y in tqdm(range(img.shape[0])):
        for x in range(img.shape[1]):
            if (x<shift_x) or (y<shift_y) or ((x+shift_x)>=img.shape[1]) or ((y+shift_y)>=img.shape[0]):
                continue
            elif (x == shift_x):
                hor_block = img_copy[y, x-shift_x:x+shift_x+1]
                img_new[y,x] = np.sum(hor_block)/div_w
            else:
                img_new[y,x] = img_new[y,x-1] + img_copy[y,x+shift_x]/div_w - img_copy[y,x-shift_x-1]/div_w
    img_copy = np.copy(img_new)
    for y in tqdm(range(img.shape[0])):
        for x in range(img.shape[1]):
            if (x<shift_x) or (y<shift_y) or ((x+shift_x)>=img.shape[1]) or ((y+shift_y)>=img.shape[0]):
                continue
            elif (y == shift_y):
                ver_block = img_new[y-shift_y:y+shift_y+1, x]
                img_new[y,x] = np.sum(ver_block)/div_h
            else:
                img_new[y,x] = img_new[y-1,x] + img_copy[y+shift_y,x]/div_h - img_copy[y-shift_y-1,x]/div_h 
    img_new = np.round(img_new).astype(int)
    return img_new


def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    w, h = int(sys.argv[3]), int(sys.argv[4])
    assert w > 0
    assert h > 0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = box_filter(img, w, h)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
