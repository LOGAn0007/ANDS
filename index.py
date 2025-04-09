from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.yaml')  # Replace 'yolov8n.pt' with your model if needed
    results = model.train(data="config.yaml", epochs=100, imgsz=640)
