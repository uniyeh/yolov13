from ultralytics.models import yolo
from ultralytics.models.yolo.detect.paired_train import PairedDectectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.paired_model import PairedDetectionModel


class PairedYOLO(YOLO):
    @property
    def task_map(self):
        return {
            **super().task_map,
            "detect":{
                "model": PairedDetectionModel,
                "trainer": PairedDectectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }