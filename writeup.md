**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard.png "Original image"
[image2]: ./output_images/undistorted.png "Undistorted image"
[image3]: ./output_images/test_image.png "Test image"
[image4]: ./output_images/undist_test_image.png "Undist test image"
[image5]: ./output_images/bin_threshold.png "Binary threshold"
[image6]: ./output_images/channel_threshold.png "Binary channels threshold"
[image7]: ./output_images/bin_region.png "Binary region mask"
[image8]: ./output_images/before_warp.png "Before warping"
[image9]: ./output_images/after_warp.png "After warping"
[image10]: ./output_images/fit_visual.png "Fit visual"
[image11]: ./output_images/draw_lane.png "Draw lane"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 34-56 of the file called `calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection by function `cv2.findChessboardCorners`.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Original image][image1]
![Undistorted image][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I performed the calculation of the camera matrix and distortion coefficients once, after which I saved the result to the file `calib.p`.
To correct the distortion of a image, I load the calibration data from the file `calib.p` and use the `cv2.undistort` function.

![Test image][image3]
![Undist test image][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To create a thresholded binary image, I used HLS color space (namely, L and S channels). For both L and S channels I apply color (lines 50-53 in `binarize.py`) and gradient along x (lines 6-18 in `binarize.py`) thresholds. A combination of these filters is realized in a function `binarize` (lines 56-71 in `binarize.py`). In addition to thresholding step I apply region mask (lines 74-99 in `binarize.py`) to leave only objects in the front of the car.

Here's an example of my output for thresholding step:
![Binary threshold][image5]
![Binary channels threshold][image6]

Here's an example of my output for region mask step:
![Binary region mask][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warping()`, which appears in lines 7-33 in the file `transform.py`.  The `warping()` function takes as inputs an image.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
	[[190, img.shape[0]], 
	[600, 445], 
	[680, 445], 
	[img.shape[1] - 160, img.shape[0]]])

offset = 200
dst = np.float32(
	[[src[0, 0] + offset, src[0, 1]], 
	[src[0, 0] + offset, 0], 
	[src[3, 0] - offset, 0], 
	[src[3, 0] - offset, src[3, 1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 390, 720      | 
| 600, 445      | 390, 0        |
| 680, 445      | 920, 0        |
| 1120, 720     | 920, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Before warping][image8]
![After warping][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify lane line pixels I used sliding window algorithm that was proposed in the lessons. I first toke a histogram along all the columns in the lower half of the image (lines 10-23 in `lane.py`). The two largest peaks of this histogram are used how indicators of the x-position of the base of the lane lines. I applied these as starting points for where to search for the lines. Then I used a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame (lines 26-99 in `lane.py`). After the pixels of the right and left lines were extracted, fits a second order polynomial by the provided data (lines 102-106 in `lane.py`).

Here's an example of my output for sliding windows and fit a polynomial:
![Fit visual][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To compute the radius of line curvature used formula `R=[(1+(2Ay+B)^2)^3/2]/|2A|` for a second order polynomia `f(y)=Ay^2+By+C`. Calculations are performed after converting x and y from pixels space to meters (lines 146-155 in `lane.py`).

The distance from the center of the lane is computed by using assumptions that lane width is 3.7 m, the camera is mounted at the center of the car and by computation of distance between left and right lines (lines 158-166 in `lane.py`).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 122-143 in my code in `lane.py` in the function `draw_lane()`.  Here is an example of my result on a test image:
![Draw lane][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline works bad on challenge videos. Problems are:
1. Algorithms doesn't work on small radius turns.
2. Binarization works bad on difficult illumination.
3. No outliers filtering.

Possible improvements:
1. Add sanity check.
2. Add support for small radius turns.
3. Change the binarization algorithm to work with difficult illumination.
4. Add smoothing of lane positions to reduce jumps.
