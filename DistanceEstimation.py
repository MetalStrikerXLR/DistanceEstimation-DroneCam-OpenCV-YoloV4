import cv2 as cv

# Distance constants
KNOWN_DISTANCE = 67  # INCHES
PERSON_WIDTH = 16  # INCHES
DEF_DETECTION_DISTANCE = 120.0 # INCHES (3m)

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


# object detector function /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2), (box[0], box[1] - 14), box, color, label])

        # return list
    return data_list


def test_object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])

        # return list
    return data_list


# Setting up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# reading the reference image from dir
ref_person = cv.imread('ReferenceImages/image1.png')
person_data = test_object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf}")

# finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
cap = cv.VideoCapture(0)
distance = 0

while True:
    ret, frame = cap.read()

    detected_objects = object_detector(frame)
    for objects in detected_objects:
        if objects[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, objects[1])
            x, y = objects[2]

            # draw rectangle and label on object normally for distance > defined distance
            if round(distance, 2) <= DEF_DETECTION_DISTANCE:
                print(objects[0] + ": " + str(distance/39.37) + "m")
                cv.rectangle(frame, objects[4], RED, 2)
                cv.putText(frame, objects[6], objects[3], FONTS, 0.5, RED, 2)

            if round(distance, 2) > DEF_DETECTION_DISTANCE:
                print(objects[0] + ": " + str(distance / 39.37) + "m")
                cv.rectangle(frame, objects[4], objects[5], 2)
                cv.putText(frame, objects[6], objects[3], FONTS, 0.5, objects[5], 2)

    cv.imshow('frame', frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()

