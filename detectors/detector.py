from detectors.retinanet.retinanet import RetinaNet


class Detector:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'retinanet':
            self.rnet = RetinaNet(model_name='model.h5')

    def get_bounding_boxes(self, frame):
        if self.model_name == 'yolo':
            from detectors.yolo.yolo_detector import get_bounding_boxes as yolo_gbb
            return yolo_gbb(frame)
        elif self.model_name == 'haarc':
            from detectors.haarc.hc_detector import get_bounding_boxes as hc_gbb
            return hc_gbb(frame)
        elif self.model_name == 'bgsub':
            from detectors.bgsub.bgsub_detector import get_bounding_boxes as bgsub_gbb
            return bgsub_gbb(frame)
        elif self.model_name == 'ssd':
            from detectors.ssd.ssd import get_bounding_boxes as ssd_gbb
            return ssd_gbb(frame)
        elif self.model_name == 'tfoda':
            from detectors.tfoda.tfoda_detector import get_bounding_boxes as tfoda_gbb
            return tfoda_gbb(frame)
        elif self.model_name == 'retinanet':
            # from detectors.retinanet.retinanet import get_bounding_boxes as retina_gbb
            # return retina_gbb(model=self.model, image=frame)
            return self.rnet.get_bounding_boxes(image=frame)
        else:
            raise Exception('Invalid detector model, algorithm or API specified '
                            '(options: yolo, tfoda, haarc, bgsub, ssd, retinanet)')

