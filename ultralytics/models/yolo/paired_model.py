from ultralytics.models import yolo
from ultralytics.models.yolo.detect.paired_train import PairedDetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.paired_model import PairedDetectionModel


class PairedYOLO(YOLO):
    @property
    def task_map(self):
        return {
            **super().task_map,
            "detect":{
                "model": PairedDetectionModel,
                "trainer": PairedDetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }