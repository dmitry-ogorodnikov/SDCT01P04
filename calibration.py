import cv2
import numpy as np
import pickle
import glob


def extract_pattern_corners(images, n_corners, show=True):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...
    obj_points = np.zeros((n_corners[0] * n_corners[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:n_corners[0], 0:n_corners[1]].T.reshape(-1, 2)

    result_obj_points = []
    result_img_points = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, n_corners)

        if ret:
            result_img_points.append(corners)
            result_obj_points.append(obj_points)

            if show:
                cv2.drawChessboardCorners(img, n_corners, corners, ret)
                cv2.imshow("Chessboard corners", img)
                cv2.waitKey()

    cv2.destroyAllWindows();
    return result_obj_points, result_img_points


def camera_calib(img_files, output_file, n_corners=(9, 6), show=True):
    is_find = False
    images = []
    for index, filename in enumerate(img_files):
        images.append(cv2.imread(filename))

    if 0 != len(images):
        img_size = (images[0].shape[1], images[0].shape[0])
        obj_points, img_points = extract_pattern_corners(images, n_corners, show)
        is_find = 0 != len(img_points)
        if is_find:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
            dist_pickle = {'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
            pickle.dump(dist_pickle, open(output_file, 'wb'))
            if show:
                for img in images:
                    dst = cv2.undistort(img, mtx, dist, None, mtx)
                    cv2.imshow("Original img", img)
                    cv2.imshow("Undistorted img", dst)
                    cv2.waitKey()
                cv2.destroyAllWindows()

    return is_find


# Load camera parameters from file
def cam_param(calib_file):
    param = pickle.load(open(calib_file, 'rb'))
    return param['mtx'], param['dist']


def main():
    output_file = 'calib.p'
    img_files = glob.glob('camera_cal/calibration*.jpg')
    camera_calib(img_files, output_file, show=False)
    mtx, dist = cam_param(output_file)
    img = cv2.imread(img_files[0])
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow('Original image', img)
    cv2.imshow('Undistorted image', dst)
    cv2.waitKey()


if __name__ == "__main__":
    main()
