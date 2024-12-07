import os
import torch
from src.config import COCO_CLASSES, colors
import cv2
import numpy as np

def test():
    class_names=['fingerprint']
    cls_threshold=0.2
    nms_threshold=0.4
    image_inputsize=640
    data_path= 'data2/images'
    modelpath=r'efficientdet.pt'
    model = torch.load(modelpath).module
    if torch.cuda.is_available():
        model.cuda()
    for img_name in os.listdir(data_path):
        image=cv2.imread(os.path.join(data_path,img_name))
        output_image = np.copy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        image = image.astype(np.float32) / 255
        image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        if height > width:
            scale = image_inputsize / height
            resized_height = image_inputsize
            resized_width = int(width * scale)
        else:
            scale = image_inputsize / width
            resized_height = int(height * scale)
            resized_width = image_inputsize

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((image_inputsize, image_inputsize, 3))
        new_image[0:resized_height, 0:resized_width] = image
        new_image = np.transpose(new_image, (2, 0, 1))
        new_image = new_image[None, :, :, :]
        new_image = torch.Tensor(new_image)
        if torch.cuda.is_available():
            new_image = new_image.cuda()
        with torch.no_grad():
            scores, labels, boxes = model(new_image)
            boxes /= scale
        if boxes.shape[0] == 0:
            continue
        box_list = []
        conf_list = []
        for box_id in range(boxes.shape[0]):
            pred_prob = float(scores[box_id])
            if pred_prob < cls_threshold:
                break
            pred_label = int(labels[box_id])
            xmin, ymin, xmax, ymax = boxes[box_id, :]
            xmin, ymin, xmax, ymax =int(xmin), int(ymin), int(xmax), int(ymax)
            # color = colors[pred_label]
            # cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            # text_size = cv2.getTextSize(class_names[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            # cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            # cv2.putText(
            #     output_image, class_names[pred_label] + ' : %.2f' % pred_prob,
            #     (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            #     (255, 255, 255), 1)
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            box_list.append([x, y, w, h])
            conf_list.append(float(pred_prob))

        indices = cv2.dnn.NMSBoxes(box_list, conf_list, 0.45, 0.3)
        imh, imw, _ = output_image.shape
        save_result_folder = r'data2/predicted/labels_EfficientDet'
        save_rtxtpath = save_result_folder + "/" + img_name[:-4] + ".txt"
        tfile = open(save_rtxtpath, 'w')
        for j in range(0, len(indices)):
            label = "fingerprint"
            conf = conf_list[indices[j]]
            x, y, w, h = box_list[indices[j]]
            x_center = (x + w / 2) / imw
            y_center = (y + h / 2) / imh
            sw = w / imw
            sh = h / imh
            cv2.putText(output_image, label + ":" + str(conf), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            tfile.write(str(0) + " " + str(x_center) + " " + str(y_center) + " " + str(sw) + " " + str(sh) + " " + str(
                conf) + "\n")
        tfile.close()
        cv2.imshow("result",output_image)
        cv2.waitKey(1)

if __name__ == "__main__":
    test()
