
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo26n-pose.pt")

cap = cv2.VideoCapture("cvvido.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

out = cv2.VideoWriter(
    "pose_result.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

LINKS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),
    (6,8),(8,10),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if results[0].keypoints is not None:
        for person in results[0].keypoints.xy.cpu().numpy():

            for a, b in LINKS:
                xa, ya = person[a]
                xb, yb = person[b]

                if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    cv2.line(
                        canvas,
                        (int(xa), int(ya)),
                        (int(xb), int(yb)),
                        (255, 255, 255),
                        3
                    )

            for x, y in person:
                if x > 0 and y > 0:
                    cv2.circle(
                        canvas,
                        (int(x), int(y)),
                        15,
                        (255, 255, 255),
                        -1
                    )

    out.write(canvas)

cap.release()
out.release()

print("Pose estimation completed!")
