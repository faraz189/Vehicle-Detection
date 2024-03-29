import sys
import cv2
from trackers.tracker import create_blob, add_new_blobs, remove_duplicates
from collections import OrderedDict
from detectors.detector import Detector
import uuid
import os
import contextlib
from datetime import datetime
import argparse
from utils.detection_roi import get_roi_frame, draw_roi
from counter import get_counting_line, is_passed_counting_line
from tqdm import tqdm


class VehicleCounter:
    def __init__(self, initial_frame, detector, tracker, droi, show_droi, mcdf, mctf, di, cl_position):
        self.frame = initial_frame  # current frame of video
        # self.detector = detector
        self.Detector = Detector(model_name=detector)
        self.tracker = tracker
        self.droi = droi  # detection region of interest
        self.show_droi = show_droi
        self.mcdf = mcdf  # maximum consecutive detection failures
        self.mctf = mctf  # maximum consecutive tracking failures
        self.di = di  # detection interval
        self.cl_position = cl_position  # counting line position

        self.blobs = OrderedDict()
        self.blob_id = 1
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0  # number of frames since last detection
        self.vehicle_count = 0  # number of vehicles counted
        self.counting_line = None if cl_position is None else get_counting_line(self.cl_position, self.f_width,
                                                                                self.f_height)

        # create blobs from initial frame
        droi_frame = get_roi_frame(self.frame, self.droi)
        initial_bboxes = self.Detector.get_bounding_boxes(droi_frame)
        for box in initial_bboxes:
            _blob = create_blob(box, frame, self.tracker)
            self.blobs[self.blob_id] = _blob
            self.blob_id += 1

    def get_count(self):
        return self.vehicle_count

    def get_blobs(self):
        return self.blobs

    def count(self, frame):
        _log = []
        self.frame = frame

        for _id, blob in list(self.blobs.items()):
            # update trackers
            success, box = blob.tracker.update(self.frame)
            if success:
                blob.num_consecutive_tracking_failures = 0
                blob.update(box)
            else:
                blob.num_consecutive_tracking_failures += 1

            # count vehicles that have left the frame if no counting line exists
            # or those that have passed the counting line if one exists
            if (self.counting_line is None and (
                    blob.num_consecutive_tracking_failures == self.mctf or
                    blob.num_consecutive_detection_failures == self.mcdf) and not blob.counted) \
                    or \
                    (self.counting_line is not None and is_passed_counting_line(blob.centroid, self.counting_line,
                                                                                self.cl_position) and not blob.counted):
                blob.counted = True
                self.vehicle_count += 1
                _log.append({'blob_id': _id, 'count': self.vehicle_count, 'datetime': datetime.now()})

            if blob.num_consecutive_tracking_failures >= self.mctf:
                # delete untracked blobs
                del self.blobs[_id]

        if self.frame_count >= self.di:
            # rerun detection
            droi_frame = get_roi_frame(self.frame, self.droi)
            boxes = self.Detector.get_bounding_boxes(droi_frame)
            self.blobs, current_blob_id = add_new_blobs(boxes, self.blobs, self.frame, self.tracker, self.blob_id,
                                                        self.counting_line, self.cl_position, self.mcdf)
            self.blob_id = current_blob_id
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1

        return _log

    def visualize(self):
        _frame = self.frame

        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(_frame, 'v_' + str(_id), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # draw counting line
        if self.counting_line is not None:
            cv2.line(_frame, self.counting_line[0], self.counting_line[1], (0, 255, 0), 3)
        # display vehicle count
        cv2.putText(_frame, 'Count: ' + str(self.vehicle_count), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2,
                    cv2.LINE_AA)
        # show detection roi
        if self.show_droi:
            _frame = draw_roi(_frame, self.droi)

        return _frame


def parse_args():
    # parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('video',
                        help='relative/absolute path to video or camera input of traffic scene')
    parser.add_argument('--iscam', action='store_true',
                        help='specify if video capture is from a camera')
    parser.add_argument('--droi',
                        help='specify a detection region of interest (ROI) \
                            i.e a set of vertices that represent the area (polygon) \
                            where you want detections to be made (format: 1,2|3,4|5,6|7,8|9,10 \
                            default: 0,0|frame_width,0|frame_width,frame_height|0,frame_height \
                            [i.e the whole video frame])')
    parser.add_argument('--showdroi', action='store_true', help='display/overlay the detection roi on the video')
    parser.add_argument('--mcdf', type=int, default=2,
                        help='maximum consecutive detection failures \
                            i.e number of detection failures before it\'s concluded that \
                            an object is no longer in the frame')
    parser.add_argument('--mctf', type=int, default=3,
                        help='maximum consecutive tracking failures \
                            i.e number of tracking failures before the tracker concludes \
                            the tracked object has left the frame')
    parser.add_argument('--di', type=int, default=10,
                        help='detection interval i.e number of frames \
                            before detection is carried out again (in order to find new vehicles \
                            and update the trackers of old ones)')
    parser.add_argument('--detector', default='yolo',
                        help='select a model/algorithm to use for vehicle detection \
                            (options: yolo, haarc, bgsub, ssd, tfoda | default: yolo)')
    parser.add_argument('--tracker', default='kcf', help='select a model/algorithm to use for vehicle tracking \
                            (options: csrt, kcf, camshift | default: kcf)')
    parser.add_argument('--record', action='store_true', help='record video and vehicle count logs')
    parser.add_argument('--headless', action='store_true', help='run VCS without UI display')
    parser.add_argument('--clposition', default='bottom', help='position of counting line (options: top, bottom, \
                            left, right | default: bottom)')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Skip number of frames from the start of the video, in case '
                             'of skipping any logos')
    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    args = parse_args()
    # capture traffic scene video
    video = int(args.video) if args.iscam else args.video
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        sys.exit('Error capturing video.')
    if args.skip_frames is not 1:
        cap.set(cv2.CAP_PROP_FRAME_COUNT, args.skip_frames - 1)

    ret, frame = cap.read()
    f_height, f_width, _ = frame.shape

    # create detection region of interest polygon
    if args.droi is None:
        droi = [(0, 0), (f_width, 0), (f_width, f_height), (0, f_height)]
    else:
        tmp_droi = []
        points = args.droi.replace(' ', '').split('|')
        for point_str in points:
            point = tuple(map(int, point_str.split(',')))
            tmp_droi.append(point)
        droi = tmp_droi

    vehicle_counter = VehicleCounter(initial_frame=frame,
                                     detector=args.detector,
                                     tracker=args.tracker,
                                     droi=droi,
                                     show_droi=args.showdroi,
                                     mcdf=args.mcdf,
                                     mctf=args.mctf,
                                     di=args.di,
                                     cl_position=args.clposition)

    if args.record:
        output_video_path = './videos/output_{0}_{1}.avi'.format(args.detector, args.tracker)
        # initialize video object and log file to record counting
        output_video = cv2.VideoWriter(output_video_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       30,
                                       (f_width, f_height))
        log_file_name = 'log.txt'
        with contextlib.suppress(FileNotFoundError):
            os.remove(log_file_name)
        log_file = open(log_file_name, 'a')
        log_file.write('vehicle_id, count, datetime\n')
        log_file.flush()

    print('Vehicle Counting System running...')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=frame_count - 1) as pbar:
        while args.iscam or cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            if ret:
                log = vehicle_counter.count(frame)
                output_frame = vehicle_counter.visualize()

                if args.record:
                    output_video.write(output_frame)
                    for item in log:
                        _row = '{0}, {1}, {2}\n'.format('v_' + str(item['blob_id']), item['count'], item['datetime'])
                        log_file.write(_row)
                        log_file.flush()

                if not args.headless:
                    resized_frame = cv2.resize(output_frame, (858, 480))
                    cv2.imshow('tracking', resized_frame)

            k = cv2.waitKey(1) & 0xFF
            # save frame if 's' key is pressed
            if k & 0xFF == ord('s'):
                cv2.imwrite(os.path.join('screenshots', 'ss_' + uuid.uuid4().hex + '.png'), output_frame)
                print('Screenshot taken.')
            # end video loop if 'q' key is pressed
            if k & 0xFF == ord('q'):
                print('Video exited.')
                break
            pbar.update(1)

            ret, frame = cap.read()

    # end capture, close window, close log file and video object if any
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    if args.record:
        log_file.close()
        output_video.release()
    print('End of video.')
