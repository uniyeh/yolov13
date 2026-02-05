from ultralytics.models.yolo.paired_model import PairedYOLO

# Create a new YOLO model from scratch
model = PairedYOLO("yolov13.yaml")

# Train the model using the small test dataset for 3 epochs
results = model.train(data="cityscapes_small_foggy_clean.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format="onnx")