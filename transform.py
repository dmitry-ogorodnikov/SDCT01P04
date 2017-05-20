import cv2
import numpy as np
import glob
from calibration import cam_param


def warping(img, inverse=False, show=False):
    src = np.float32([[190, img.shape[0]], [600, 445], [680, 445], [img.shape[1] - 160, img.shape[0]]])
    offset = 200
    dst = np.float32([[src[0, 0] + offset, src[0, 1]], [src[0, 0] + offset, 0],
                      [src[3, 0] - offset, 0], [src[3, 0] - offset, src[3, 1]]])

    if inverse:
        mat = cv2.getPerspectiveTransform(dst, src)
    else:
        mat = cv2.getPerspectiveTransform(src, dst)

    result = cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    if show:
        temp = img.copy()
        cv2.line(temp, tuple(src[0]), tuple(src[1]), color=[0, 0, 255])
        cv2.line(temp, tuple(src[1]), tuple(src[2]), color=[0, 0, 255])
        cv2.line(temp, tuple(src[2]), tuple(src[3]), color=[0, 0, 255])
        cv2.line(temp, tuple(src[3]), tuple(src[0]), color=[0, 0, 255])

        warped = cv2.warpPerspective(temp, mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        cv2.imshow("Region", temp)
        cv2.imshow("Warped", warped)
        cv2.waitKey()

    return result


def main():
    img_files = glob.glob('test_images/*.jpg')
    mtx, dist = cam_param('calib.p')
    for file in img_files:
        img = cv2.imread(file)
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        warping(undist_img, show=True)
    return None


if __name__ == '__main__':
    main()
