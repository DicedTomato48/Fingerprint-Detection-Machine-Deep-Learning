import cv2
from ultralytics import YOLO
import cv2
import time
import os
import math
import logging
import numpy as np

logging.getLogger('ultralytics').setLevel(logging.WARNING)

modelpath_handdetect=r"best.pt"
model_HandDetect = YOLO(modelpath_handdetect)  # load a custom model
model_HandDetect.to('cpu')

modelpath_fingerdetect=r"yolov11_L.pt"
model_FingerDetect=YOLO(modelpath_fingerdetect)
model_FingerDetect.to('cpu')
save_result_folder = r'data2/predicted/labels_combine2'
def Hand_Detect(image,f_name):
    re_detect_flag=False
    count=1
    imh,imw,_=image.shape
    dis_image=image.copy()
    results = model_HandDetect(image)  # predict on an image
    box_list = []
    conf_list = []
    id_list = []
    save_rtxtpath = save_result_folder+"/"+f_name[:-4]+".txt"
    tfile = open(save_rtxtpath, 'w')
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confs = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class IDs

        for box, conf, cls in zip(boxes, confs, classes):
            label = "hand"
            conf = float(conf)
            conf = float("{:.2f}".format(conf))

            x = int(box[0])
            y = int(box[1])
            w = int(box[2] - x)
            h = int(box[3] - y)
            box_list.append([x, y, w, h])
            conf_list.append(conf)
    indices = cv2.dnn.NMSBoxes(box_list, conf_list, 0.5, 0.4)
    for j in range(0, len(indices)):
        label = "hand"
        conf = conf_list[indices[j]]
        x, y, w, h = box_list[indices[j]]
        if w>(2/3)*imw or h>(2/3)*imh:
            re_detect_flag=True
            break
        x=int(x)-20
        y=int(y)-20
        w=int(w)+40
        h=int(h)+40
        if x<0:
            x=0
        if y<0:
            y=0
        if x+w>imw:
            w=imw-x-1
        if y+h>imh:
            h=imh-y-1

        crop_part=image[y:y+h,x:x+w]
        # savepath=r'data2/crop_images/'+f_name[:-4]+"_"+str(count)+".jpg"
        # cv2.imwrite(savepath,crop_part)
        # count+=1
        f_results = model_FingerDetect(crop_part)  # predict on an image
        f_box_list = []
        f_conf_list = []
        for fresult in f_results:
            f_boxes = fresult.boxes.xyxy  # Bounding box coordinates
            f_confs = fresult.boxes.conf  # Confidence scores
            f_classes = fresult.boxes.cls  # Class IDs
            for f_box, f_conf, f_cls in zip(f_boxes, f_confs, f_classes):
                f_conf = float("{:.2f}".format(f_conf))
                f_x = int(f_box[0])
                f_y = int(f_box[1])
                f_w = int(f_box[2] - f_x)
                f_h = int(f_box[3] - f_y)
                f_box_list.append([f_x, f_y, f_w, f_h])
                f_conf_list.append(f_conf)
        f_indices = cv2.dnn.NMSBoxes(f_box_list, f_conf_list, 0.5, 0.4)
        for j in range(0, len(f_indices)):
            f_label = "fingerprint"
            f_conf =f_conf_list[f_indices[j]]
            fx, fy, fw, fh = f_box_list[f_indices[j]]
            x_center = (fx+x + fw / 2) / imw
            y_center = (fy+y + fh / 2) / imh
            sw = fw / imw
            sh = fh / imh

            tfile.write(str(0)+" "+str(x_center)+" "+str(y_center)+" "+str(sw)+" "+str(sh)+" "+str(conf)+"\n")

            cv2.putText(dis_image, f_label + ":" + str(f_conf), (fx+x, fy - 20+y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(dis_image, (fx+x, fy+y), (fx + fw+x, fy + fh+y), (255, 0, 0), 1)
        cv2.putText(dis_image, label + ":" + str(conf), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(dis_image, (x, y), (x + w, y + h), (255, 0, 255), 1)

    if re_detect_flag==True:
        # img = np.zeros((imw*2, imh*2, 3), np.uint8)
        img = np.full((imh*2, imw*2, 3), (255, 255, 255), np.uint8)

        st_x=int(imw/2)
        st_y=int(imh/2)
        img[st_y:st_y+imh,st_x:st_x+imw]=image
        dis_image=img.copy()
        results = model_HandDetect(img)  # predict on an image
        box_list = []
        conf_list = []
        id_list = []
        # save_rtxtpath = save_result_folder+"/"+f_name[:-4]+".txt"
        # tfile = open(save_rtxtpath, 'w')
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            confs = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Class IDs

            for box, conf, cls in zip(boxes, confs, classes):
                label = "hand"
                conf = float(conf)
                conf = float("{:.2f}".format(conf))

                x = int(box[0])
                y = int(box[1])
                w = int(box[2] - x)
                h = int(box[3] - y)
                box_list.append([x, y, w, h])
                conf_list.append(conf)
        indices = cv2.dnn.NMSBoxes(box_list, conf_list, 0.5, 0.4)
        for j in range(0, len(indices)):
            label = "hand"
            conf = conf_list[indices[j]]
            x, y, w, h = box_list[indices[j]]
            rx=int(x)-20
            ry=int(y)-20
            rw=int(w)+40
            rh=int(h)+40
            if rx<0:
                rx=0
            if ry<0:
                ry=0
            if rx+rw>2*imw:
                rw=(2*imw)-rx-1
            if ry+rh>2*imh:
                rh=(2*imh)-ry-1

            crop_part=img[ry:ry+rh,rx:rx+rw]
            # savepath = r'data2/new_dataset/crop_images/' + f_name[:-4] + "_" + str(count) + ".jpg"
            # cv2.imwrite(savepath, crop_part)
            count += 1
            f_results = model_FingerDetect(crop_part)  # predict on an image
            f_box_list = []
            f_conf_list = []
            for fresult in f_results:
                f_boxes = fresult.boxes.xyxy  # Bounding box coordinates
                f_confs = fresult.boxes.conf  # Confidence scores
                f_classes = fresult.boxes.cls  # Class IDs
                for f_box, f_conf, f_cls in zip(f_boxes, f_confs, f_classes):
                    f_conf = float("{:.2f}".format(f_conf))
                    f_x = int(f_box[0])
                    f_y = int(f_box[1])
                    f_w = int(f_box[2] - f_x)
                    f_h = int(f_box[3] - f_y)
                    f_box_list.append([f_x, f_y, f_w, f_h])
                    f_conf_list.append(f_conf)
            f_indices = cv2.dnn.NMSBoxes(f_box_list, f_conf_list, 0.5, 0.4)
            for j in range(0, len(f_indices)):
                f_label = "fingerprint"
                f_conf =f_conf_list[f_indices[j]]
                fx, fy, fw, fh = f_box_list[f_indices[j]]
                rfx=fx+rx-st_x
                rfy=fy+ry-st_y
                rfw=fw
                rfh=fh
                if rfx < 0:
                    rfx = 0
                if rfy < 0:
                    rfy = 0
                if rfx + rfw > imw:
                    rfw = imw - rfx - 1
                if rfy + rfh >imh:
                    rfh = imh - rfy - 1

                x_center = (rfx+rfw/ 2) / imw
                y_center = (rfy+rfh/ 2) / imh
                sw = rfw / imw
                sh = rfh / imh

                tfile.write(str(0)+" "+str(x_center)+" "+str(y_center)+" "+str(sw)+" "+str(sh)+" "+str(conf)+"\n")

                cv2.putText(dis_image, f_label + ":" + str(f_conf), (fx+rx, fy - 20+ry), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(dis_image, (fx+rx, fy+ry), (fx + fw+rx, fy + fh+ry), (255, 0, 0), 2)
            cv2.putText(dis_image, label + ":" + str(conf), (rx, ry - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(dis_image, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)

    tfile.close()
    print("done")
    # cv2.imwrite("result.jpg",dis_image)
    dw=800
    dh=int((dw/imw)*imh)
    dis_image=cv2.resize(dis_image,(dw,dh))
    cv2.imshow("Result", dis_image)
    cv2.waitKey(1)


def main():
    f_folder =r"data2/images"
    f_list = os.listdir(f_folder)
    for f_name in f_list:
        time.sleep(2)
        imgpath = os.path.join(f_folder, f_name)
        print(imgpath)
        frame=cv2.imread(imgpath)
        Hand_Detect(frame,f_name)

if __name__ == "__main__":
    main()
