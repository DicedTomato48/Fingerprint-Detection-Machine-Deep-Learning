import uuid
import json
import os
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import csv


def yolo_folders_to_coco_json(images_dir, labels_dir, output_json, category_name="hand"):
    # Define the COCO JSON format data2 structure.
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Function to add a category to COCO data2.
    def add_category(category_id, category_name):
        category = {
            "id": category_id,
            "name": category_name,
            "supercategory": "person",
        }
        coco_data["categories"].append(category)

    # Function to add an image to COCO data2.
    def add_image(image_id, image_file, image_size):
        image_info = {
            "id": image_id,
            "file_name": image_file,
            "width": image_size[0],
            "height": image_size[1],
        }
        coco_data["images"].append(image_info)
        # image_id = generate_unique_int(image_file)
        return image_id

    def generate_unique_int(string):
        # Generate a UUID based on the string
        uuid_object = uuid.uuid3(uuid.NAMESPACE_DNS, string)

        # Convert the UUID object to an integer
        integer = int(uuid_object.hex, 16)

        return integer

    #    Function to add an annotation to COCO data2.
    def add_annotation(annotation_id, image_id, category_id, bbox, score, filename):
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,  # [x, y, width, height]
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "score": score,
            "filename": filename
        }
        coco_data["annotations"].append(annotation)

    def add_annotation_ns(annotation_id, image_id, category_id, bbox, filename):
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,  # [x, y, width, height]
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,

            "filename": filename
        }
        coco_data["annotations"].append(annotation)

    # Read YOLO annotations and convert them to COCO format.
    category_id = 0
    annotation_id = 1

    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            image_filename = os.path.splitext(filename)[0] + ".jpg"
            image_path = os.path.join(images_dir, image_filename)
            img = Image.open(image_path)
            image_size = img.size
            img.close()

            image_id = add_image(generate_unique_int(image_filename), image_filename, image_size)
            # image_id = add_image(len(coco_data["images"]) + 1, image_filename, image_size)

            with open(os.path.join(labels_dir, filename), "r") as yolo_file:
                for line in yolo_file:
                    data = line.strip().split()

                    try:
                        x_center, y_center, width, height, score = map(float, data[1:6])
                        x_min = (x_center - width / 2) * image_size[0]
                        y_min = (y_center - height / 2) * image_size[1]
                        width *= image_size[0]
                        height *= image_size[1]
                        # annotation_id = generate_unique_int(filename)
                        add_annotation(annotation_id, image_id, category_id, [x_min, y_min, width, height], score,
                                       filename)

                    except Exception as e:
                        x_center, y_center, width, height = map(float, data[1:5])
                        x_min = (x_center - width / 2) * image_size[0]
                        y_min = (y_center - height / 2) * image_size[1]
                        width *= image_size[0]
                        height *= image_size[1]
                        # annotation_id = generate_unique_int(filename)
                        add_annotation_ns(annotation_id, image_id, category_id, [x_min, y_min, width, height], filename)
                    annotation_id += 1  # generate_unique_int(filename)

                # Add a category (you can customize this part according to your dataset)
                add_category(category_id, category_name)

        # Save the COCO JSON file
        try:
            with open(output_json, "w") as json_file:
                json.dump(coco_data, json_file)
        except Exception as e:
            print(e)
    print(f"COCO JSON file saved as {os.getcwd(), output_json}")


if __name__ == '__main__':
    ground_truth_images = 'data2/images'
    ground_truth_images = os.path.join(os.getcwd(), ground_truth_images)

    ground_truth_labels = 'data2/labels'
    ground_truth_labels = os.path.join(os.getcwd(), ground_truth_labels)

    ground_truth_json_file = 'data2/Jsons/ground_truth_json_file.json'
    yolo_folders_to_coco_json(ground_truth_images, ground_truth_labels, ground_truth_json_file)

    predicted_labels_yolov10_n = 'data2/predicted/labels_yolov10_N'
    predicted_json_file_yolov10_n = 'data2/Jsons/predicted_json_file_yolov10_n.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov10_n)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov10_n)

    predicted_labels_yolov10_m = 'data2/predicted/labels_yolov10_M'
    predicted_json_file_yolov10_m = 'data2/Jsons/predicted_json_file_yolov10_m.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov10_m)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov10_m)

    predicted_labels_yolov10_s = 'data2/predicted/labels_yolov10_S'
    predicted_json_file_yolov10_s = 'data2/Jsons/predicted_json_file_yolov10_s.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov10_s)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov10_s)

    predicted_labels_yolov10_x = 'data2/predicted/labels_yolov10_X'
    predicted_json_file_yolov10_x = 'data2/Jsons/predicted_json_file_yolov10_x.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov10_x)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov10_x)

    predicted_labels_rtdetr = 'data2/predicted/labels_RTDETR'
    predicted_json_file_rtdetr = 'data2/Jsons/predicted_json_file_rtdetr.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_rtdetr)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_rtdetr)

    predicted_labels_retinanet = 'data2/predicted/labels_RetinaNet'
    predicted_json_file_retinanet = 'data2/Jsons/predicted_json_file_retinanet.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_retinanet)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_retinanet)

    predicted_labels_efficientdet = 'data2/predicted/labels_EfficientDet'
    predicted_json_file_efficientdet = 'data2/Jsons/predicted_json_file_efficientdet.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_efficientdet)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_efficientdet)

    predicted_labels_yolov11_n = 'data2/predicted/labels_yolov11_N'
    predicted_json_file_yolov11_n = 'data2/Jsons/predicted_json_file_yolov11_n.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov11_n)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov11_n)

    predicted_labels_yolov11_s = 'data2/predicted/labels_yolov11_S'
    predicted_json_file_yolov11_s = 'data2/Jsons/predicted_json_file_yolov11_s.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov11_s)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov11_s)

    predicted_labels_yolov11_m = 'data2/predicted/labels_yolov11_M'
    predicted_json_file_yolov11_m = 'data2/Jsons/predicted_json_file_yolov11_m.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov11_m)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov11_m)

    predicted_labels_yolov11_l = 'data2/predicted/labels_yolov11_L'
    predicted_json_file_yolov11_l = 'data2/Jsons/predicted_json_file_yolov11_l.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov11_l)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov11_l)

    predicted_labels_yolov11_x = 'data2/predicted/labels_yolov11_X'
    predicted_json_file_yolov11_x = 'data2/Jsons/predicted_json_file_yolov11_x.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov11_x)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov11_x)

    predicted_labels_combine = 'data2/predicted/labels_combine'
    predicted_json_file_combine = 'data2/Jsons/predicted_json_file_combine.json'
    # predicted_labels = os.path.join(os.getcwd(), predicted_labels_yolov11_x)
    # yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file_yolov11_x)

    # Initialize COCO annotation objects
    coco_gt = COCO(ground_truth_json_file)
    # coco_gt.dataset = coco_data
    coco_gt.createIndex()

    coco_pre_yolov10_n = COCO(predicted_json_file_yolov10_n)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov10_n, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    str_array = ["Model Name", "AP", "AP@.50", "AP#.75", "AP-S","AP-M","AP-L","AR1","AR10","AR100","AR-S","AR-M","AR-L"]
    # Open the CSV file in write mode
    # with open('combined_test_result.csv', 'w', newline='') as csvfile:
    csvfile=open('Result_Tables/table_combined_test_dataset.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(str_array)

    print("#############################################################################")
    print("Evaluation esult of Yolov10_N model")
    print(coco_eval.stats)
    list_re=coco_eval.stats
    model_name="Yolov10_N"
    re_arr=[]
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)  # Write the entire array as a single row

    print("#############################################################################")
    coco_pre_yolov10_m = COCO(predicted_json_file_yolov10_m)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov10_m, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of Yolov10_M model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Yolov10_M"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_yolov10_s = COCO(predicted_json_file_yolov10_s)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov10_s, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of Yolov10_S model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Yolov10_S"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_yolov10_x = COCO(predicted_json_file_yolov10_x)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov10_x, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of Yolov10_X model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Yolov10_X"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_rtdetr = COCO(predicted_json_file_rtdetr)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_rtdetr, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of RtDetr model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "RtDetr"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_retinanet = COCO(predicted_json_file_retinanet)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_retinanet, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of RetinaNet  model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "RetinaNet"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_efficientdet = COCO(predicted_json_file_efficientdet)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_efficientdet, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of EfficientDet model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "EfficientDet"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_yolov11_n = COCO(predicted_json_file_yolov11_n)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov11_n, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of Yolov11_N model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Yolov11_N"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_yolov11_s = COCO(predicted_json_file_yolov11_s)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov11_s, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of Yolov11_S model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Yolov11_S"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_yolov11_m = COCO(predicted_json_file_yolov11_m)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov11_m, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of Yolov11_M model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Yolov11_M"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_yolov11_l = COCO(predicted_json_file_yolov11_l)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov11_l, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of Yolov11_L model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Yolov11_L"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_yolov11_x = COCO(predicted_json_file_yolov11_x)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_yolov11_x, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation esult of Yolov11_X model")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Yolov11_X"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
    coco_pre_combine = COCO(predicted_json_file_combine)
    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre_combine, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("#############################################################################")
    print("Evaluation result of Combined Method")
    print(coco_eval.stats)
    list_re = coco_eval.stats
    model_name = "Combined-v11-N"
    re_arr = []
    re_arr.append(model_name)
    for item in list_re:
        re_arr.append(str(item))
    writer.writerow(re_arr)
    print("#############################################################################")
