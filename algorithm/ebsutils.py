import hashlib
import os

import cv2


class EBSUtils:
    taken_screenshots_array = []

    def __init__(self, screenshots_path="screenshots/", color=None):
        if color is None:
            color = [212, 153, 199]
        self.createfolderifnotexist(screenshots_path)
        self.screenshots_path = screenshots_path
        self.color = color

    def takescreenshot(self, frame, type, detectionnumber, datetime):
        # define the file name
        filename = self.screenshots_path + "photo_%s_%d.jpg" % (type, detectionnumber)

        # save frame as JPEG file
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        # add it into array
        self.taken_screenshots_array.append({
            "type": type,
            "screenshot_path": filename,
            "datetime": datetime.strftime("%c")
        })

        return filename

    def createfolderifnotexist(self, directory):
        os.makedirs(directory, exist_ok=True)

    def creatacolormask(self, classID, frame, startY, endY, startX, endX, COLORS):
        # extract the ROI of the image but *only* extracted the
        # masked region of the ROI
        roi = frame[startY:endY, startX:endX]

        # grab the color used to visualize this particular class,
        # then create a transparent overlay by blending the color
        # with the ROI
        color = COLORS[classID]
        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

        # store the blended ROI in the original frame
        frame[startY:endY, startX:endX] = blended

        # draw the bounding box of the instance on the frame
        color = [int(c) for c in color]
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      color, 2)

        return frame

    def md5(self, fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
