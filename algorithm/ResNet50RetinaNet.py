#!/usr/bin/env python
# coding: utf-8
# Author: Muhammad Faraz
# Part of the Vehicle counter project
# The file takes care of running object detection on input video
# and outputs another video that has superimposed bounding boxes

# ## Load necessary modules
import argparse
import logging
import os
import sys


import cv2
import keras
import json as libjson
from ebsutils import EBSUtils
import tensorflow as tf
# noinspection PyUnresolvedReferences
from keras_retinanet import models
# noinspection PyUnresolvedReferences
from keras_retinanet.utils.colors import label_color
# noinspection PyUnresolvedReferences
from keras_retinanet.utils.image import preprocess_image, resize_image
from tqdm import tqdm
from datetime import datetime

from sort import *
from utils import intersect
from utils import path_leaf
from barrel import barrel_distort


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def process_video(path_in, labels_to_names, model, skip_frames=1):
    _script_starting_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    _result_directory = os.path.join(os.getcwd(), 'results')
    if not os.path.isdir(_result_directory):
        os.makedirs(_result_directory)
    video_name = path_leaf(path_in)
    path_out = '{0}_result.mp4'.format(video_name.split(sep=".")[0])
    fps = 24
    cap = cv2.VideoCapture(path_in)
    counter = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vehicle_count = 0
    tracker = Sort()
    memory = {}
    LABELID_CARRO = 2
    LABELID_MOTO = 3
    LABELID_PERSON = 0
    LABELID_ONIBUS = 5
    LABELID_TREM = 6
    LABELID_CAMINHAO = 7
    LABELID_BICICLETA = 1
    vehicles_counter = {
        LABELID_CARRO: 0,
        LABELID_MOTO: 0,
        LABELID_ONIBUS: 0,
        LABELID_TREM: 0,
        LABELID_CAMINHAO: 0,
        LABELID_BICICLETA: 0
    }
    line = [(450, 0), (450, 1080)]

    _detections = []
    with tqdm(total=frame_count) as pbar:
        try:
            while True:
                ret, draw = cap.read()
                draw = barrel_distort(draw)
                if not ret:
                    break
                if counter % skip_frames == 0:
                    bgr = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
                    # bgr = barrel_distort(bgr)
                    # preprocess image for network
                    image = preprocess_image(bgr)
                    image, scale = resize_image(image)

                    # process image
                    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

                    if counter == 0:
                        out = cv2.VideoWriter(path_out,
                                              cv2.VideoWriter_fourcc(*"mp4v"),
                                              fps,
                                              (draw.shape[1], draw.shape[0]),
                                              True)

                    # correct for image scale
                    boxes /= scale
                    detection_data = []
                    dets = []
                    for box, score, label in zip(boxes[0], scores[0], labels[0]):
                        if score < 0.7:
                            continue
                        b = box.astype(int)

                        # visualize detections

                        color = label_color(label)
                        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, 6)
                        caption = "%s: %.1f%%" % (labels_to_names[label], score * 100)
                        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        dets.append([b[0], b[1], b[2], b[3], score])
                        detection_data.append([int(label), color])

                    # For car tracking
                    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
                    dets = np.asarray(dets)
                    tracks = tracker.update(dets)

                    boxes = []
                    indexIDs = []
                    c = []
                    previous = memory.copy()
                    memory = {}

                    # draw line
                    cv2.line(draw, line[0], line[1], (0, 255, 255), 3)

                    for track in tracks:
                        boxes.append([track[0], track[1], track[2], track[3]])
                        indexIDs.append(int(track[4]))
                        memory[indexIDs[-1]] = boxes[-1]

                    if len(boxes) > 0:
                        i = int(0)
                        for box in boxes:
                            # extract the bounding box coordinates
                            (x, y) = (int(box[0]), int(box[1]))
                            (w, h) = (int(box[2]), int(box[3]))

                            # draw a bounding box rectangle and label on the image
                            # color = [int(c) for c in COLORS[classIDs[i]]]
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                            # color = [int(c) for c in COLORS[classIDs[i] % len(COLORS)]]
                            # color = label_color()
                            # color = COLORS[classIDs[i]]
                            # color = classIDs[i]
                            cv2.rectangle(draw, (x, y), (w, h), detection_data[i][1], 2)

                            if indexIDs[i] in previous:
                                previous_box = previous[indexIDs[i]]
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                                cv2.line(draw, p0, p1, detection_data[i][1], 3)

                                # noinspection PyTypeChecker
                                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], scores[i])
                                # text = "{}".format(indexIDs[i])
                                # cv2.putText(draw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                                if intersect(p0, p1, line[0], line[1]):
                                    # count the vehicle
                                    vehicle_count += 1
                                    if int(detection_data[i][0]) is LABELID_PERSON:
                                        vehicles_counter[LABELID_MOTO] += 1
                                    else:
                                        vehicles_counter[int(detection_data[i][0])] += 1
                                    # saves image file
                                    _image_file_name = "{0}_{1}_detection_{2}.jpg".format(
                                        video_name.split(sep=".")[0],
                                        vehicle_count,
                                        labels_to_names[int(
                                            detection_data[i][0])])
                                    _detections.append({'detection_type': labels_to_names[int(
                                                                            detection_data[i][0])],
                                                        'datetime': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                                                        'file_screenshot': _image_file_name})

                                    cv2.imwrite(os.path.join(_result_directory, _image_file_name),draw)
                                i += 1

                    info = [
                        ("TOTAL", vehicle_count),
                        ("Caminhao", vehicles_counter[LABELID_CAMINHAO]),
                        ("Bitrem", vehicles_counter[LABELID_TREM]),
                        ("Onibus", vehicles_counter[LABELID_ONIBUS]),
                        ("Carros", vehicles_counter[LABELID_CARRO]),
                        ("Bike", vehicles_counter[LABELID_BICICLETA]),
                        ("Moto", vehicles_counter[LABELID_MOTO]),
                    ]
                    # loop over the info tuples and draw them on our frame
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(draw, text, (10, 500 - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    # after the code is stable / results are good, we will remove it
                    # cv2.putText(draw, str(vehicle_count), (100, 200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
                    cv2.imwrite('frames/img%08d.jpg' % counter, draw)
                    out.write(draw)
                counter = counter + 1
                pbar.update(1)
        except Exception as e:
            print(e)
            print(sys.exc_info()[0])
        finally:
            cap.release()
            cv2.destroyAllWindows()
            # export the result in a JSON file
            out.release()
            print("[INFO] Creating JSON file...")

            json_data = dict()
            json_data['header'] = {'started at': _script_starting_time,
                                   'finished_at': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                                   'path_screenshots': _result_directory,
                                   'direction': 'LTR (dummt value)'}
            json_data['video'] = {'path_original': path_in,
                                  'path_output': path_out,
                                  'recorded_at': '2019-05-04 17:51:31 (dummy value)'}
            json_data['counters'] = {'truck': vehicles_counter[LABELID_CAMINHAO],
                                     'bus': vehicles_counter[LABELID_ONIBUS],
                                     'car': vehicles_counter[LABELID_CARRO],
                                     'motorbike': vehicles_counter[LABELID_MOTO]}
            json_data['detections'] = _detections

            json = libjson.dumps(json_data, indent=4)
            f = open("json/result_%s.json" % (video_name.split(sep=".")[0]), "w")
            f.write(json)
            f.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="Set the name of the video file which would be used as input to the model")
    parser.add_argument("-m", "--model", default='model.h5',
                        help="Set the name of the file which contain the model weights")
    parser.add_argument("-s", '--skip', type=int, default=1,
                        help='Set the variable to skip every nth frame in the video for quick progression')

    args = parser.parse_args()
    args_output = dict(input_file=args.input_file, model=args.model, skip=args.skip)

    return args_output


def load_model(model_path):
    model = models.load_model(model_path, backbone_name='resnet50')
    return model


def print_logs(**kwargs):
    logging.info('Processing File = {0}'.format(kwargs['input_file']))
    logging.info('Loading weights from file = {0}'.format(kwargs['model']))
    logging.info(
        'Saving result video to file = {0}'.format('{0}_result.mp4'.format(kwargs['input_file'].split(sep=".")[0])))


def init(**kwargs):
    for keys in ['input_file', 'model']:
        assert os.path.isfile(kwargs[keys]), 'Could not find {0}. Please check input parameters. ' \
                                             'If not sure, run the python file with --help flag'.format(keys)

    # if not os.path.isdir('frames'):
    #     os.makedirs('frames')
    EBSUtils.createfolderifnotexist(EBSUtils, "json")


def main():
    logging.basicConfig(level=logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    keras.backend.tensorflow_backend.set_session(get_session())
    input_args = parse_arguments()
    # load retinanet model
    init(**input_args)
    logging.info("Loading model into memory")
    _model = load_model(model_path=input_args['model'])
    logging.info('Model loaded.')

    # load label to names mapping for visualization purposes
    _labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                        7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                        12: 'parking meter',
                        13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                        20: 'elephant',
                        21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                        27: 'tie',
                        28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                        38: 'tennis racket',
                        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                        52: 'hot dog',
                        53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                        66: 'keyboard',
                        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                        73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                        79: 'toothbrush'}
    print_logs(**input_args)
    process_video(path_in=input_args['input_file'],
                  labels_to_names=_labels_to_names,
                  model=_model,
                  skip_frames=input_args['skip']
                  )


if __name__ == '__main__':
    main()
