import numpy as np
import cv2
import glob


def sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, int(orient == 'x'),
                                      int(orient == 'y'), ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(grad_mag) / 255
    grad_mag = (grad_mag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= thresh[0]) & (grad_mag <= thresh[1])] = 1

    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate the x and y gradients
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    abs_grad_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = np.zeros_like(abs_grad_dir)
    binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1

    return binary_output


def color_thresh(img, thresh=(0, 255)):
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_output


def binarize(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    kernel_size = 15
    sobel_l = sobel_thresh(hls[:, :, 1], kernel_size, 'x', (30, 255))
    sobel_s = sobel_thresh(hls[:, :, 2], kernel_size, 'x', (30, 255))

    color_l = color_thresh(hls[:, :, 1], (100, 255))
    color_s = color_thresh(hls[:, :, 2], (100, 255))

    binary = np.zeros_like(sobel_l)
    binary[((sobel_l == 1) | (sobel_s == 1)) | ((color_l == 1) & (color_s == 1))] = 1

    binary = 255 * np.dstack((binary, binary, binary)).astype(np.uint8)
    # channels = 255*np.dstack((sobel_l, sobel_s, ((color_l == 1) & (color_s == 1)))).astype(np.uint8)

    return binary  # , channels


def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    vertices = np.array([[(0, img.shape[0]), (590, 445), (720, 445),
                          (img.shape[1], img.shape[0])]], dtype=np.int32)

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def main():
    img_files = glob.glob('test_images/*.jpg')

    for file in img_files:
        img = cv2.imread(file)
        bin_img = binarize(img)
        result = region_of_interest(bin_img)
        cv2.imshow('Init image', img)
        cv2.imshow('Combined', result)
        cv2.waitKey()


if __name__ == '__main__':
    main()
