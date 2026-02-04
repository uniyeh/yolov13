from ultralytics.models.yolo.paired_model import PairedYOLO

# Create a new YOLO model from scratch
model = PairedYOLO("yolo13n.yaml")


# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="hazy_data.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format="onnx")