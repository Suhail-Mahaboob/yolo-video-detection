from ultralytics import YOLO
import cv2
import csv

model = YOLO("yolov8n.pt")

video_path = "testt.mp4"
cap = cv2. VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*"mp4") 
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.mp4", fourcc, fps, (width,height))

csv_file = open("detections.csv", mode = "w", newline = "")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame","class","confidence", "x1","y1","x2","y2"])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count+=1
    results = model(frame)

    for r in results:
        annotated_frame = r.plot()
        out.write(annotated_frame)
        cv2.imshow("YOLO Video",annotated_frame)

        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf)

            csv_writer.writerow([frame_count, cls_name, f"{conf:.2f}", int(x1), int(y1), int(x2), int(y2)])

            ##if conf>0.6:
            ##    csv_writer.writerow([frame_count, cls_name, f"{conf:.2f}", int(x1), int(y1), int(x2), int(y2)])


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
