import cv2
import numpy as np
from calibration import cam_param
from binarize import binarize, region_of_interest
from transform import warping
import lane


def main():
    filename = 'project_video.mp4'
    is_first_img = True
    mtx, dist = cam_param('calib.p')
    left_fit = [np.array([False])]
    right_fit = [np.array([False])]

    video_capture = cv2.VideoCapture(filename)

    while video_capture.isOpened():
        ret_val, img = video_capture.read()
        if ret_val:
            img = cv2.undistort(img, mtx, dist, None, mtx)
            bin_img = binarize(img)
            bin_img = region_of_interest(bin_img)
            warped_img = warping(bin_img)

            if is_first_img:
                base_pos = lane.lane_histogram(warped_img[:, :, 0])
                left_pix, right_pix, _ = lane.lane_pixels(warped_img[:, :, 0], base_pos)
                is_first_img = False
            else:
                left_pix, right_pix, _ = lane.lane_pixels(warped_img[:, :, 0], lane_known=True, left_fit=left_fit,
                                                          right_fit=right_fit)

            left_fit, right_fit = lane.lane_polyfit(left_pix, right_pix)
            result = lane.draw_lane(img, warped_img, left_fit, right_fit)
            left_cur = lane.get_curvature(left_fit, img.shape)
            right_cur = lane.get_curvature(right_fit, img.shape)
            pos_car = lane.get_pos(left_fit, right_fit, img.shape)

            cv2.putText(result, 'Radius of left curvature: %.4fm' % left_cur, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1,
                        [0, 0, 255])
            cv2.putText(result, 'Radius of right curvature: %.4fm' % right_cur, (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1,
                        [0, 0, 255])
            cv2.putText(result, 'Position from center: %.4fm' % pos_car, (20, 120), cv2.FONT_HERSHEY_DUPLEX, 1,
                        [0, 0, 255])

            cv2.imshow("Result", result)
            cv2.waitKey(1)
        else:
            break


if __name__ == "__main__":
    main()
