import time

import cv2
from ultralytics import YOLO
import os
import numpy as np

# Load YOLO models
modelpath_handdetect = r"best.pt"
model_HandDetect = YOLO(modelpath_handdetect)
model_HandDetect.to('cpu')

modelpath_fingerdetect = r"yolov11_L.pt"
model_FingerDetect = YOLO(modelpath_fingerdetect)
model_FingerDetect.to('cpu')

save_result_folder = r'data2/predicted/labels_combine2'
output_video_path = "output_with_bounding_boxes.mp4"


def Hand_Detect(image):
    re_detect_flag = False
    count = 1
    imh, imw, _ = image.shape
    dis_image = image.copy()
    results = model_HandDetect(image)  # Predict on an image
    box_list = []
    conf_list = []

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, classes):
            label = "hand"
            conf = float("{:.2f}".format(float(conf)))
            x = int(box[0])
            y = int(box[1])
            w = int(box[2] - x)
            h = int(box[3] - y)
            box_list.append([x, y, w, h])
            conf_list.append(conf)

    indices = cv2.dnn.NMSBoxes(box_list, conf_list, 0.5, 0.4)
    if len(indices) > 0 and isinstance(indices, (list, np.ndarray)):  # Ensure indices is a list or array
        for j in indices.flatten():
            label = "hand"
            conf = conf_list[j]
            x, y, w, h = box_list[j]

            # Ensure the bounding box fits within the image
            x, y, w, h = max(0, x - 20), max(0, y - 20), min(w + 40, imw - x), min(h + 40, imh - y)
            crop_part = image[y:y + h, x:x + w]

            # Finger detection
            f_results = model_FingerDetect(crop_part)
            f_box_list, f_conf_list = [], []

            for fresult in f_results:
                f_boxes = fresult.boxes.xyxy
                f_confs = fresult.boxes.conf

                for f_box, f_conf in zip(f_boxes, f_confs):
                    f_conf = float("{:.2f}".format(f_conf))
                    f_x = int(f_box[0])
                    f_y = int(f_box[1])
                    f_w = int(f_box[2] - f_x)
                    f_h = int(f_box[3] - f_y)
                    f_box_list.append([f_x, f_y, f_w, f_h])
                    f_conf_list.append(f_conf)

            f_indices = cv2.dnn.NMSBoxes(f_box_list, f_conf_list, 0.5, 0.4)
            if len(f_indices) > 0 and isinstance(f_indices, (list, np.ndarray)):
                for i in f_indices.flatten():
                    f_label = "fingerprint"
                    f_conf = f_conf_list[i]
                    fx, fy, fw, fh = f_box_list[i]
                    # Larger font size and bolder text for fingerprints
                    cv2.putText(dis_image, f"{f_label}:{f_conf}", (fx + x, fy + y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (200, 100, 100), 2)  # Larger, bolder text and lighter color
                    # Lighter bounding box for fingerprints
                    cv2.rectangle(dis_image, (fx + x, fy + y), (fx + fw + x, fy + fh + y), (200, 200, 255), 2)

            # Larger font size and bolder text for hands
            cv2.putText(dis_image, f"{label}:{conf}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (50, 150, 50), 2)  # Larger, bolder text for hand label
            # Lighter bounding box for hands
            cv2.rectangle(dis_image, (x, y), (x + w, y + h), (150, 255, 150), 2)  # Light green bounding box

    return dis_image


def main():
    video_path = r"/Users/ariadmp48pr/Downloads/evaluation_code 8/data/v5.mov"
    cap = cv2.VideoCapture(video_path)

    # Get video dimensions and setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        processed_frame = Hand_Detect(frame)

        # Write processed frame to output video
        out.write(processed_frame)

    # Release resources
    cap.release()
    out.release()
    print("Video processing complete. Output saved as", output_video_path)


if __name__ == "__main__":
    main()
