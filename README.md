# YOLOv8 Video Object Detection with Logging

This project uses the **YOLOv8 model** from [Ultralytics](https://github.com/ultralytics/ultralytics) to perform object detection on video files.  
It annotates detections on each frame, saves the output as a video, and logs all detections into a CSV file.

---

## Features
- Detects objects in a video using YOLOv8.
- Saves annotated video as `output.mp4`.
- Logs detections into `detections.csv` with:
  - Frame number
  - Object class
  - Confidence score
  - Bounding box coordinates (`x1, y1, x2, y2`)
- Option to filter detections by confidence (default logs all).

---

## Installation
```bash
git clone https://github.com/Suhail-Mahaboob/yolo-video-detection.git
cd yolo-video-detection
pip install -r requirements.txt
```

Usage
Place your video in the project folder (e.g., test.mp4).

Run the script:

```bash
python video_detect.py
```

Output:

Annotated video: output.mp4

Detection log: detections.csv

Example CSV Output
csv
frame,class,confidence,x1,y1,x2,y2
1,person,0.89,150,200,300,400
1,car,0.76,400,220,550,350
2,dog,0.92,100,150,250,300


Requirements
>Python 3.8+
>ultralytics
>opencv-python

Notes
>Press Q during playback to quit early.

Uncomment the confidence filter in the script to only log detections above 60%.
