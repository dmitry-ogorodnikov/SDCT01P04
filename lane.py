import numpy as np
import cv2
import glob
from calibration import cam_param
from binarize import binarize, region_of_interest
from transform import warping
from matplotlib import pyplot as plt


def lane_histogram(img, show=False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if show:
        plt.plot(histogram)
        plt.show()

    return leftx_base, rightx_base


def lane_pixels(img, base_pos=None, lane_known=False, left_fit=None, right_fit=None, draw=False):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    # Set the width of the windows +/- margin
    margin = 100
    out = None

    if lane_known:
        left_lane_idx = (
            (nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] - margin)) & (
                nonzero_x < (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] + margin)))
        right_lane_idx = (
            (nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] - margin)) & (
                nonzero_x < (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] + margin)))
    else:
        # Choose the number of sliding windows
        n_windows = 9
        # Set height of windows
        window_height = np.int(img.shape[0] / n_windows)
        # Current positions to be updated for each window
        leftx_current = base_pos[0]
        rightx_current = base_pos[1]

        # Set minimum number of pixels found to recenter window
        min_pix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_idx = []
        right_lane_idx = []

        if draw:
            out = 255 * np.dstack((img, img, img)).astype(np.uint8)

        # Step through the windows one by one
        for window in range(n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if draw:
                # Draw the windows on the visualization image
                cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                nonzero_x < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_idx.append(good_left_inds)
            right_lane_idx.append(good_right_inds)
            # If you found > min_pix pixels, recenter next window on their mean position
            if len(good_left_inds) > min_pix:
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > min_pix:
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)

    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane_idx]
    lefty = nonzero_y[left_lane_idx]
    rightx = nonzero_x[right_lane_idx]
    righty = nonzero_y[right_lane_idx]

    return (leftx, lefty), (rightx, righty), out


def lane_polyfit(left_pixels, right_pixels):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_pixels[1], left_pixels[0], 2)
    right_fit = np.polyfit(right_pixels[1], right_pixels[0], 2)
    return left_fit, right_fit


def visualization(img, left_fit, right_fit, left_pix, right_pix):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    img[left_pix[1], left_pix[0]] = [255, 0, 0]
    img[right_pix[1], right_pix[0]] = [0, 0, 255]

    img[ploty.astype(np.int), left_fitx.astype(np.int)] = [128, 128, 128]
    img[ploty.astype(np.int), right_fitx.astype(np.int)] = [128, 128, 128]


def draw_lane(img, warped_img, left_fit, right_fit):
    ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    color_warp = np.zeros_like(warped_img).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    new_warp = warping(color_warp, inverse=True)

    # new_warp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, new_warp, 0.3, 0)
    return result


def get_curvature(fit, img_shape):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30. / img_shape[0]  # meters per pixel in y dimension
    xm_per_pix = 3.7 / (img_shape[1] / 2)  # meters per pixel in x dimension
    y_eval = img_shape[0]

    val_1 = (2 * fit[0] * y_eval + fit[1]) * xm_per_pix / ym_per_pix
    val_2 = 2 * fit[0] * xm_per_pix / (ym_per_pix ** 2)

    return ((1 + val_1 * val_1) ** 1.5) / np.absolute(val_2)


def get_pos(left_fit, right_fit, img_shape):
    y_eval = img_shape[0]
    midpoint = img_shape[1] / 2
    xm_per_pix = 3.7 / (img_shape[1] / 2)  # meters per pixel in x dimension

    x_left_pix = left_fit[0] * (y_eval ** 2) + left_fit[1] * y_eval + left_fit[2]
    x_right_pix = right_fit[0] * (y_eval ** 2) + right_fit[1] * y_eval + right_fit[2]

    return ((x_left_pix + x_right_pix) / 2.0 - midpoint) * xm_per_pix


def main():
    img_files = glob.glob('test_images/*.jpg')
    mtx, dist = cam_param('calib.p')

    for file in img_files:
        img = cv2.imread(file)
        cv2.imshow("Original", img)
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        bin_img = binarize(undist_img)
        bin_img = region_of_interest(bin_img)
        warped_img = warping(bin_img)
        base_pos = lane_histogram(warped_img[:, :, 0])
        left_pix, right_pix, out = lane_pixels(warped_img[:, :, 0], base_pos, draw=True)
        left_fit, right_fit = lane_polyfit(left_pix, right_pix)
        result = draw_lane(undist_img, warped_img, left_fit, right_fit)
        cv2.imshow("Finding lines", result)
        cv2.waitKey()


if __name__ == '__main__':
    main()
