import uuid
import json
import os
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
    # ground_truth_images = r'D:\Works\Works\fingerprint_detection\dataset\dataset/valid\images'
    ground_truth_images = os.path.join(os.getcwd(), ground_truth_images)

    ground_truth_labels = 'data2/labels'
    # ground_truth_labels = r'D:\Works\Works\fingerprint_detection\dataset\dataset/valid\labels'
    ground_truth_labels = os.path.join(os.getcwd(), ground_truth_labels)

    # predicted_labels = 'data2/predicted/labels'
    # predicted_labels = 'data2/predicted/labels_RTDETR'
    predicted_labels = 'data2/predicted/labels_yolov10_N'
    predicted_labels = os.path.join(os.getcwd(), predicted_labels)

    ground_truth_json_file = 'ground_truth_json_file.json'
    yolo_folders_to_coco_json(ground_truth_images, ground_truth_labels, ground_truth_json_file)
    predicted_json_file = 'predicted_json_file.json'
    yolo_folders_to_coco_json(ground_truth_images, predicted_labels, predicted_json_file)

    # Initialize COCO annotation objects
    coco_gt = COCO(ground_truth_json_file)
    # coco_gt.dataset = coco_data
    coco_gt.createIndex()

    coco_pre = COCO(predicted_json_file)

    # Perform the evaluation
    coco_eval = COCOeval(coco_gt, coco_pre, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(coco_eval.stats)