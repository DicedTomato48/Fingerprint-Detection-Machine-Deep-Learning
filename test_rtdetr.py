import cv2
from ultralytics import RTDETR
import cv2
import time
import os
import logging

logging.getLogger('ultralytics').setLevel(logging.WARNING)
model = RTDETR("rtdetr.pt")  # load a custom model
# model.to('cpu')
model.to('cuda')

f_folder = r"data2/images"
save_result_folder=r'data2/predicted/labels_RTDETR'
f_list = os.listdir(f_folder)
for f_name in f_list:
    imgpath = os.path.join(f_folder, f_name)
    frame=cv2.imread(imgpath)
    imh,imw,_=frame.shape
    frame_c=frame.copy()
    s_time = time.time()
    results = model(frame)  # predict on an image
    box_list=[]
    conf_list=[]
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confs = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class IDs

        for box, conf, cls in zip(boxes, confs, classes):
            label = "fingerprint"
            conf = float(conf)
            conf = float("{:.2f}".format(conf))

            x = int(box[0])
            y = int(box[1])
            w = int(box[2] - x)
            h = int(box[3] - y)
            box_list.append([x, y, w, h])
            conf_list.append(conf)
    indices = cv2.dnn.NMSBoxes(box_list, conf_list, 0.45, 0.3)
    image_filename = os.path.splitext(f_name)[0]
    save_rtxtpath = save_result_folder+"/"+image_filename+".txt"
    tfile = open(save_rtxtpath, 'w')
    for j in range(0,len(indices)):
        label="fingerprint"
        conf=conf_list[indices[j]]
        x,y,w,h=box_list[indices[j]]
        x_center=(x+w/2)/imw
        y_center=(y+h/2)/imh
        sw=w/imw
        sh=h/imh
        cv2.putText(frame_c, label + ":" + str(conf), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255),2,cv2.LINE_AA)
        cv2.rectangle(frame_c, (x, y), (x + w, y + h), (255, 255, 0),2)
        tfile.write(str(0)+" "+str(x_center)+" "+str(y_center)+" "+str(sw)+" "+str(sh)+" "+str(conf)+"\n")
    tfile.close()

    cv2.imshow("FingerPrint Detection2",frame_c)
    print("predicted time:{}".format(time.time()-s_time))
    cv2.waitKey(1)
