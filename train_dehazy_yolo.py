from ultralytics.models.yolo.paired_model import PairedYOLO

# Create a new YOLO model from scratch
model = PairedYOLO("yolov13.yaml")

# Train the model using the small test dataset for 3 epochs
results = model.train(
  data='cityscapes_foggy_clean.yaml',
  epochs=200, 
  batch=16, 
  imgsz=640,
  scale=0.5,  # S:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; L:0.5; X:0.6
  # device="0,1,2,3",
  project='weights',      # 1. 指定主目錄 (專案資料夾)
  name='yolo_foggy_clean_run', # 2. 指定子目錄 (這次實驗的名稱)
  exist_ok=True
)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format="onnx")