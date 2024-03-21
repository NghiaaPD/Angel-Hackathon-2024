from ultralytics import YOLO

model = YOLO('./model/best.pt')

results = model(source=0, show=True, conf=0.25, save=True)