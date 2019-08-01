import numpy as np
import cv2


def pad_image(image, size = 0.1):
    borderType = cv2.BORDER_CONSTANT
    top = int(size * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(size * src.shape[1])  # shape[1] = cols
    right = left
    dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, [0, 0, 0])
    return dst


def barrel_distort(src, ):
    width = src.shape[1]
    height = src.shape[0]

    distCoeff = np.zeros((4, 1), np.float64)

    # TODO: add your coefficients here!
    k1 = -5.0e-5  # negative to remove barrel distortion
    k2 = 0
    p1 = 0
    p2 = 0.0

    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

# assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)

    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = 15.  # define focal length x
    cam[1, 1] = 15.  # define focal length y
    src = pad_image(image=src, size=0.1)
# here the undistortion will be computed
    dst = cv2.undistort(src, cam, distCoeff)
    return dst


if __name__ == '__main__':
    src = cv2.imread("results/video1_1_detection_person.jpg")
    dst = barrel_distort(src)
    cv2.imwrite('temp.jpg', dst)

    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

