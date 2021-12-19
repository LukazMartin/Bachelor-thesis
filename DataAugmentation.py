import tensorflow as tf
import numpy as np
import math
import random
import os
import scipy.io as sio
from config import train_config_yolo_depth
import cv2


class NewDataAugmentation:

    @staticmethod
    def data_augmentation(image, label, filename):
        path = os.path.join(train_config_yolo_depth.UpdateTrainConfiguration.root_of_datasets,
                     train_config_yolo_depth.UpdateTrainConfiguration.dataset_name)
        if np.random.rand() < train_config_yolo_depth.UpdateTrainConfiguration.data_aug_opts.prob and train_config_yolo_depth.UpdateTrainConfiguration.data_aug_opts.apply_data_augmentation_new:

            if train_config_yolo_depth.UpdateTrainConfiguration.data_aug_opts.horizontal_flip:
                if np.random.rand() > 0.5:
                    image, label = NewDataAugmentation.horizontal_flipping(image, label, filename)

            if train_config_yolo_depth.UpdateTrainConfiguration.data_aug_opts.masking:
                if np.random.rand() > 0.5:
                    if np.random.rand() > 0.5:
                        image, label = NewDataAugmentation.masking(image, label, filename)
                    else:
                        image, label = NewDataAugmentation.maskingNotHumans(image, label, filename)

            if train_config_yolo_depth.UpdateTrainConfiguration.data_aug_opts.rotation:
                if np.random.rand() > 0.5:
                    image, label = NewDataAugmentation.rotation(image, label, filename)

            if train_config_yolo_depth.UpdateTrainConfiguration.data_aug_opts.resize:
                if np.random.rand() > 0.5:
                    if np.random.rand() > 0.5:
                        image, label = NewDataAugmentation.cropping(image, label, filename)
                    else:
                        image, label = NewDataAugmentation.scaling(image, label, filename)

        return image, label

    @staticmethod
    def horizontal_flipping(image, label, filename):
        image = np.flip(image, axis=1)
        lines = []
        for line in label:
            line_split = line.split(' ')
            classId = int(line_split[0])
            x = 1 - float(line_split[1])
            y = float(line_split[2])
            W = float(line_split[3])
            H = float(line_split[4])
            lines.append(str(classId) + " " + str(x) + " " + str(y) + " " + str(W) + " " + str(H) + "\n")
        return image, lines

    @staticmethod
    def cropping(image, label, filename):
        lines = label
        xmin, ymin, xmax, ymax = -1, -1, -1, -1
        boundingBox = []
        for line in lines:
            line_split = line.split(' ')
            classId = int(line_split[0])
            x = float(line_split[1])
            y = float(line_split[2])
            W = float(line_split[3])
            H = float(line_split[4])
            x1 = x - (W / 2)
            y1 = y - (H / 2)
            x2 = x + (W / 2)
            y2 = y + (H / 2)
            boundingBox.append([classId, x, y, W, H])
            if x1 < xmin or xmin == -1:
                xmin = x1
            if y1 < ymin or ymin == -1:
                ymin = y1
            if x2 > xmax:
                xmax = x2
            if y2 > ymax:
                ymax = y2
        xmin = xmin * image.shape[1]
        xmax = xmax * image.shape[1]
        ymin = ymin * image.shape[0]
        ymax = ymax * image.shape[0]
        startX = 0
        startY = 0
        endX = image.shape[1]
        endY = image.shape[0]
        if xmin > image.shape[1] * 0.25:
            startX = math.floor(image.shape[1] * 0.25)
        if ymin > image.shape[0] * 0.25:
            startY = math.floor(image.shape[0] * 0.25)
        if xmax < image.shape[1] * 0.75:
            endX = math.ceil(image.shape[1] * 0.75)
        if ymax < image.shape[0] * 0.75:
            endY = math.ceil(image.shape[0] * 0.75)

        if math.floor(xmin) < startX:
            xmin = startX
        if math.floor(ymin) < startY:
            ymin = startY
        if math.ceil(xmax) > endX:
            xmax = endX
        if math.ceil(ymax) > endY:
            ymax = endY
        xmin = random.randint(startX, math.floor(xmin))
        ymin = random.randint(startY, math.floor(ymin))
        xmax = random.randint(math.ceil(xmax), endX)
        ymax = random.randint(math.ceil(ymax), endY)
        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        image = image[ymin:ymin + (ymax-ymin), xmin:xmin + (xmax-xmin)]
        lines = []
        for i in boundingBox:
            output = ""
            i[1] = ((i[1] * imageWidth) - xmin) / float(image.shape[1])
            i[2] = ((i[2] * imageHeight) - ymin) / float(image.shape[0])
            i[3] = (i[3] * imageWidth) / float(image.shape[1])
            i[4] = (i[4] * imageHeight) / float(image.shape[0])
            for num in range(4):
                output = output + (str(i[num]) + " ")
            output = output + (str(i[4]) + "\n")
            lines.append(output)
        return image, lines

    @staticmethod
    def scaling(image, label, filename):
        lines = label
        xmin, ymin, xmax, ymax = -1, -1, -1, -1
        boundingBox = []
        for line in lines:
            line_split = line.split(' ')
            classId = int(line_split[0])
            x = float(line_split[1])
            y = float(line_split[2])
            W = float(line_split[3])
            H = float(line_split[4])
            x1 = x - (W / 2)
            y1 = y - (H / 2)
            x2 = x + (W / 2)
            y2 = y + (H / 2)
            boundingBox.append([classId, x, y, W, H])
            if x1 < xmin or xmin == -1:
                xmin = x1
            if y1 < ymin or ymin == -1:
                ymin = y1
            if x2 > xmax:
                xmax = x2
            if y2 > ymax:
                ymax = y2
        xmin = xmin * image.shape[1]
        xmax = xmax * image.shape[1]
        ymin = ymin * image.shape[0]
        ymax = ymax * image.shape[0]
        if xmin > image.shape[1] * 0.1:
            xmin = image.shape[1] * 0.1 * random.random()
        if ymin > image.shape[0] * 0.1:
            ymin = image.shape[0] * 0.1 * random.random()
        if xmax < image.shape[1] * 0.9:
            xmax = image.shape[1] * 0.9 + image.shape[1] * 0.1 * random.random()
        if ymax < image.shape[0] * 0.9:
            ymax = image.shape[0] * 0.9 + image.shape[0] * 0.1 * random.random()
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if i < xmin or i > xmax:
                    if filename[0] != "R":
                        image[j, i] = 0
                    else:
                        image[j, i] = (0, 0, 0)
                if j < ymin or j > ymax:
                    if filename[0] != "R":
                        image[j, i] = 0
                    else:
                        image[j, i] = (0, 0, 0)
        return image, lines

    @staticmethod
    def rotation(image, label, filename):
        lines = label
        boundingBox = []
        for line in lines:
            line_split = line.split(' ')
            classId = int(line_split[0])
            x = float(line_split[1])
            y = float(line_split[2])
            W = float(line_split[3])
            H = float(line_split[4])
            x1 = x - (W / 2)
            y1 = y - (H / 2)
            x2 = x + (W / 2)
            y2 = y + (H / 2)
            x1 = x1 * image.shape[1]
            y1 = y1 * image.shape[0]
            x2 = x2 * image.shape[1]
            y2 = y2 * image.shape[0]
            x = x * image.shape[1]
            y = y * image.shape[0]
            W = W * image.shape[1]
            H = H * image.shape[0]
            boundingBox.append([x1, y1, x2, y2, x, y, W, H, classId])
        rotation = 0
        while rotation == 0:
            rotation = random.randint(-15, 15)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        angle = abs(rotation) * math.pi / 180.0
        lines = []
        clockwise = True
        if rotation > 0:
            clockwise = False
        for box in boundingBox:
            if clockwise:
                ymin = (box[0] - (image.shape[1] / 2.0)) * math.sin(angle) + (
                            box[1] - (image.shape[0] / 2.0)) * math.cos(angle) + (image.shape[0] / 2.0)
                xmax = (box[2] - (image.shape[1] / 2.0)) * math.cos(angle) - (
                            box[1] - (image.shape[0] / 2.0)) * math.sin(angle) + (image.shape[1] / 2.0)
                xmin = (box[0] - (image.shape[1] / 2.0)) * math.cos(angle) - (
                            box[3] - (image.shape[0] / 2.0)) * math.sin(angle) + (image.shape[1] / 2.0)
                ymax = (box[2] - (image.shape[1] / 2.0)) * math.sin(angle) + (
                            box[3] - (image.shape[0] / 2.0)) * math.cos(angle) + (image.shape[0] / 2.0)
            else:
                ymin = (-(box[2] - (image.shape[1] / 2.0)) * math.sin(angle)) + (
                            box[1] - (image.shape[0] / 2.0)) * math.cos(angle) + (image.shape[0] / 2.0)
                xmax = (box[2] - (image.shape[1] / 2.0)) * math.cos(angle) + (
                            box[3] - (image.shape[0] / 2.0)) * math.sin(angle) + (image.shape[1] / 2.0)
                xmin = (box[0] - (image.shape[1] / 2.0)) * math.cos(angle) + (
                            box[1] - (image.shape[0] / 2.0)) * math.sin(angle) + (image.shape[1] / 2.0)
                ymax = (-(box[0] - (image.shape[1] / 2.0))) * math.sin(angle) + (
                            box[3] - (image.shape[0] / 2.0)) * math.cos(angle) + (image.shape[0] / 2.0)
            if (ymin > 0 and xmax < image.shape[1] and xmin > 0 and ymax < image.shape[0]):
                width = float(xmax - xmin)
                height = float(ymax - ymin)
                xcenter = xmin + (width / 2.0)
                ycenter = ymin + (height / 2.0)
                width = width / image.shape[1]
                height = height / image.shape[0]
                xcenter = xcenter / image.shape[1]
                ycenter = ycenter / image.shape[0]
                output = ""
                output += str(box[8]) + " "
                output += str(xcenter) + " "
                output += str(ycenter) + " "
                output += str(width) + " "
                output += str(height) + "\n"
                lines.append(output)
            else:
                width = xmax - xmin
                height = ymax - ymin
                area = width * height
                outsideXmax = 0
                outsideYmax = 0
                outsideXmin = 0
                outsideYmin = 0
                if xmax > image.shape[1]:
                    outsideXmax = xmax - image.shape[1]
                    xmax = image.shape[1]
                if ymax > image.shape[0]:
                    outsideYmax = ymax - image.shape[0]
                    ymax = image.shape[0]
                if xmin < 0:
                    outsideXmin = abs(xmin)
                    xmin = 0
                if ymin < 0:
                    outsideYmin = abs(ymin)
                    ymin = 0
                areaOutside = 0
                areaOutside = areaOutside + (outsideXmax * height)
                areaOutside = areaOutside + (outsideYmax * width)
                areaOutside = areaOutside + (outsideXmin * height)
                areaOutside = areaOutside + (outsideYmin * width)
                if area - areaOutside > (area / 2.0):
                    width = float(xmax - xmin)
                    height = float(ymax - ymin)
                    xcenter = xmin + (width / 2.0)
                    ycenter = ymin + (height / 2.0)
                    width = width / image.shape[1]
                    height = height / image.shape[0]
                    xcenter = xcenter / image.shape[1]
                    ycenter = ycenter / image.shape[0]
                    output = ""
                    output += str(box[8]) + " "
                    output += str(xcenter) + " "
                    output += str(ycenter) + " "
                    output += str(width) + " "
                    output += str(height) + "\n"
                    lines.append(output)
        return image, lines


    @staticmethod
    def masking(image, label, filename):
        path = os.path.join(train_config_yolo_depth.UpdateTrainConfiguration.root_of_datasets,
                            train_config_yolo_depth.UpdateTrainConfiguration.dataset_name + "/")
        data = []
        lines = label
        image_area = 0
        bounding_boxes = []
        for line in lines:
            line_split = line.split(' ')
            x = float(line_split[1])
            y = float(line_split[2])
            W = float(line_split[3])
            H = float(line_split[4])
            x = x * image.shape[1]
            y = y * image.shape[0]
            W = W * image.shape[1]
            H = H * image.shape[0]
            image_area = image_area + W * H
            bounding_boxes.append([x - (W / 2), y - (H / 2), x + (W / 2), y + (H / 2)])
        if len(lines) == 0:
            image_area = 1000
        else:
            image_area = image_area / len(lines)
        if image_area < 1000:
            image_size = 0
        elif image_area < 10000:
            image_size = 1
        else:
            image_size = 2

        if len(data) == 0:
            if filename[0] == "R":
                data = sio.loadmat(
                    path + 'rgb' + str(image_size) + '.mat')[
                    'my_struct']
                aux = []
                for f in data:
                    pos = f.find(' ')
                    if pos != -1:
                        f = f[:pos]
                    aux.append(f)
                data = aux
            elif filename[0] == "d":
                data = sio.loadmat(
                    path + 'depth' + str(image_size) + '.mat')[
                    'my_struct']
            else:
                data = sio.loadmat(
                    path + 'thermal' + str(image_size) + '.mat')[
                    'my_struct']
        findPosition = True
        pasteImage = True
        scaling = True
        while findPosition:
            infiltred_file = filename
            infiltred_label_position = 0
            while infiltred_file == filename and len(data) > 0:
                randomFile = data[random.randint(0, len(data) - 1)]
                pos = randomFile.find('.')
                pos_space = randomFile.find(' ')
                if pos_space != -1:
                    pos_space -= 1
                infiltred_file = randomFile[:pos + 4]
                infiltred_label_position = randomFile[pos + 4:pos_space]
                if len(data) == 1:
                    break
            infiltred_label = open(path + "annotations/" + infiltred_file, 'r')
            infiltred_lines = infiltred_label.readlines()
            infiltred_label.close()
            imagefile = path + "images/" + infiltred_file[:-3] + "png"
            if infiltred_file[0] == "R":
                infiltred_image = cv2.imread(imagefile)
                infiltred_image = cv2.cvtColor(infiltred_image, cv2.COLOR_BGR2RGB).astype(np.float32)
            elif 'thermal' in infiltred_file:
                infiltred_image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED).astype(np.float32)
                infiltred_image[infiltred_image < 28315] = 28315  # clip at 283.15 K (or 10 degrees)
                infiltred_image[infiltred_image > 31315] = 31315  # clip at 313.15 K (or 40 degrees)
                infiltred_image = np.expand_dims(infiltred_image, -1)
                infiltred_image = np.concatenate([infiltred_image, infiltred_image, infiltred_image], axis=-1)
            elif 'depth' in infiltred_file:
                infiltred_image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED).astype(
                    np.float32)  # check here if you need to use cv2.IMREAD_UNCHANGED
                infiltred_image = np.expand_dims(infiltred_image, -1)
                infiltred_image = np.concatenate([infiltred_image, infiltred_image, infiltred_image], axis=-1)
            random_human = infiltred_lines[int(infiltred_label_position)]
            line_split = random_human.split(' ')
            x = float(line_split[1])
            y = float(line_split[2])
            W = float(line_split[3])
            H = float(line_split[4])
            x = x * infiltred_image.shape[1]
            W = W * infiltred_image.shape[1]
            y = y * infiltred_image.shape[0]
            H = H * infiltred_image.shape[0]
            ymin = int(math.floor(y - H / 2))
            ymax = int(math.ceil(y + H / 2))
            xmin = int(math.floor(x - W / 2))
            xmax = int(math.ceil(x + W / 2))
            if ymin < 0:
                ymin = 0
            if xmin < 0:
                xmin = 0
            if ymax > infiltred_image.shape[0]:
                ymax = infiltred_image.shape[0]
            if xmax > infiltred_image.shape[1]:
                xmax = infiltred_image.shape[1]
            infiltred_image = infiltred_image[ymin:ymax,xmin:xmax]
            if infiltred_image.shape[1] == 0 or infiltred_image.shape[0] == 0:
                return image, label
            dimensionY = float(infiltred_image.shape[1]) / float(infiltred_image.shape[0])
            dimensionY = float(image_area) / float(dimensionY)
            dimensionY = math.sqrt(dimensionY)
            if dimensionY == 0:
                return image, label
            dimensionX = image_area / dimensionY
            random_scale = random.randint(-1, 1)
            random_scale = 1 + random_scale * 0.1
            dimensionY = dimensionY * random_scale
            dimensionX = dimensionX * random_scale
            if scaling:
                infiltred_image = cv2.resize(infiltred_image, (int(math.floor(dimensionX)), int(math.floor(dimensionY))))
            if infiltred_image.shape[1] >= image.shape[1] or infiltred_image.shape[0] >= image.shape[0]:
                if int(math.floor(image.shape[1] * 0.1)) > 0 and int(math.floor(image.shape[0] * 0.1)) > 0:
                    infiltred_image = cv2.resize(infiltred_image,(int(math.floor(image.shape[1] * 0.1)), int(math.floor(image.shape[0] * 0.1))))
                else:
                    return image, label
            im = np.ones((image.shape[0], image.shape[1]))
            for box in bounding_boxes:
                height = box[3] - box[1]
                width = box[2] - box[0]
                y0 = max(0, int(box[1] + (height * 0.1) - (infiltred_image.shape[0])))
                y1 = min(image.shape[0], int(box[3]+1 - (height * 0.1)))
                x0 = max(0, int(box[0] + (width * 0.1) - (infiltred_image.shape[1])))
                x1 = min(image.shape[1], int(box[2]+1 - (width * 0.1)))
                im[y0:y1, x0:x1] = 0
            im[int(max(0,image.shape[0]-infiltred_image.shape[0])):image.shape[0]+1,:] = 0
            im[:,int(max(0,image.shape[1]-infiltred_image.shape[1])):image.shape[1]+1] = 0
            possibleOptions = np.argwhere(im > 0)
            if len(possibleOptions) > 0:
                randomPosition = random.randint(0, len(possibleOptions) - 1)
                randomY = possibleOptions[randomPosition][0]
                randomX = possibleOptions[randomPosition][1]
                findPosition = False
            if len(possibleOptions) == 0 and image_size > 0:
                image_size = image_size - 1
                if filename[0] == "R":
                    data = sio.loadmat(
                        path + 'rgb' + str(image_size) + '.mat')[
                        'my_struct']
                    aux = []
                    for f in data:
                        pos = f.find(' ')
                        if pos != -1:
                            f = f[:pos]
                        aux.append(f)
                    data = aux
                elif filename[0] == "d":
                    data = sio.loadmat(
                        path + 'depth' + str(image_size) + '.mat')[
                        'my_struct']
                else:
                    data = sio.loadmat(
                        path + 'thermal' + str(
                            image_size) + '.mat')['my_struct']
                scaling = False
            if len(possibleOptions) == 0 and image_size == 0:
                findPosition = False
                pasteImage = False
        if pasteImage:
            image[randomY:randomY + infiltred_image.shape[0], randomX:randomX + infiltred_image.shape[1]] = infiltred_image
        labels = []
        for line in lines:
            labels.append(line)
        if pasteImage:
            output = ""
            output += str(0) + " "
            output += str(((infiltred_image.shape[1] / 2.0) + randomX) / image.shape[1]) + " "
            output += str(((infiltred_image.shape[0] / 2.0) + randomY) / image.shape[0]) + " "
            output += str(infiltred_image.shape[1] / float(image.shape[1])) + " "
            output += str(infiltred_image.shape[0] / float(image.shape[0])) + "\n"
            labels.append(output)

        return image, labels

    @staticmethod
    def maskingNotHumans(image, label, filename):
        path = os.path.join(train_config_yolo_depth.UpdateTrainConfiguration.root_of_datasets,
                             train_config_yolo_depth.UpdateTrainConfiguration.dataset_name+"/")
        files = os.listdir(path + "annotations")
        files_rgb = [x for x in files if x[0] == "R"]
        files_depth = [x for x in files if x[0] == "d"]
        files_thermal = [x for x in files if x[0] == "t"]
        if filename[0] == "R":
            data = files_rgb
        elif filename[0] == "d":
            data = files_depth
        else:
            data = files_thermal
        lines = label
        bounding_boxes = []
        for line in lines:
            line_split = line.split(' ')
            x = float(line_split[1])
            y = float(line_split[2])
            W = float(line_split[3])
            H = float(line_split[4])
            x = x * image.shape[1]
            y = y * image.shape[0]
            W = W * image.shape[1]
            H = H * image.shape[0]
            bounding_boxes.append([x - W / 2, y - H / 2, x + W / 2, y + H / 2])
        smallest_area = 10000
        random_scale = random.randint(-1, 1)
        random_scale = 1 + random_scale * 0.1
        smallest_area_H = math.floor(math.sqrt(smallest_area) * random_scale)
        smallest_area_W = smallest_area_H
        times = 0
        n = random.randint(1, 5)
        im = np.ones((image.shape[0], image.shape[1]))
        for box in bounding_boxes:
            height = box[3] - box[1]
            width = box[2] - box[0]
            y0 = max(0,int(box[1] + (height * 0.1) - smallest_area_H))
            y1 = min(image.shape[0],int(box[3] + 1 - (height * 0.1)))
            x0 = max(0,int(box[0] + (width * 0.1) - smallest_area_W))
            x1 = min(image.shape[1],int(box[2] + 1 - (width * 0.1)))
            im[y0:y1,x0:x1] = 0
        im[int(max(0,image.shape[0] - smallest_area_H)):image.shape[0] + 1, :] = 0
        im[:,int(max(0,image.shape[1] - smallest_area_W)):image.shape[1] + 1] = 0
        possibleOptions_target = np.argwhere(im > 0)
        while times < n and len(possibleOptions_target) > 0:
            infiltred_file = filename
            while infiltred_file == filename and len(data) > 0:
                infiltred_file = data[random.randint(0, len(data) - 1)]
                if len(data) == 1:
                    break
            infiltred_label = open(path + "annotations/" + infiltred_file, 'r')
            infiltred_lines = infiltred_label.readlines()
            infiltred_label.close()
            imagefile = path + "images/" + infiltred_file[:-3] + "png"
            if infiltred_file[0] == "R":
                infiltred_image = cv2.imread(imagefile)
                infiltred_image = cv2.cvtColor(infiltred_image, cv2.COLOR_BGR2RGB).astype(np.float32)
            elif 'thermal' in infiltred_file:
                infiltred_image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED).astype(np.float32)
                infiltred_image[infiltred_image < 28315] = 28315  # clip at 283.15 K (or 10 degrees)
                infiltred_image[infiltred_image > 31315] = 31315  # clip at 313.15 K (or 40 degrees)
                infiltred_image = np.expand_dims(infiltred_image, -1)
                infiltred_image = np.concatenate([infiltred_image, infiltred_image, infiltred_image], axis=-1)
            elif 'depth' in infiltred_file:
                infiltred_image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED).astype(
                    np.float32)  # check here if you need to use cv2.IMREAD_UNCHANGED
                infiltred_image = np.expand_dims(infiltred_image, -1)
                infiltred_image = np.concatenate([infiltred_image, infiltred_image, infiltred_image], axis=-1)
            infiltred_boxes = []
            for line in infiltred_lines:
                line_split = line.split(' ')
                x = float(line_split[1])
                y = float(line_split[2])
                W = float(line_split[3])
                H = float(line_split[4])
                x = x * infiltred_image.shape[1]
                W = W * infiltred_image.shape[1]
                y = y * infiltred_image.shape[0]
                H = H * infiltred_image.shape[0]
                infiltred_boxes.append([x - W / 2, y - H / 2, x + W / 2, y + H / 2])
            im1 = np.ones((infiltred_image.shape[0], infiltred_image.shape[1]))
            for box in infiltred_boxes:
                y0 = max(0, int(box[1] - smallest_area_H))
                y1 = min(infiltred_image.shape[0], int(box[3] + 1))
                x0 = max(0, int(box[0] - smallest_area_W))
                x1 = min(infiltred_image.shape[1],int(box[2] + 1))
                im1[y0:y1,x0:x1] = 0
            im1[int(max(0,infiltred_image.shape[0] - smallest_area_H)):infiltred_image.shape[0] + 1, :] = 0
            im1[:, int(max(0,infiltred_image.shape[1] - smallest_area_W)):infiltred_image.shape[1] + 1] = 0
            possibleOptions = np.argwhere(im1 > 0)
            possiblePaste = False
            if len(possibleOptions) > 0:
                randomSpot = possibleOptions[random.randint(0, len(possibleOptions) - 1)]
                randomY = randomSpot[0]
                randomX = randomSpot[1]
                possiblePaste = True
            if possiblePaste and len(possibleOptions_target) > 0:
                infiltred_image = infiltred_image[int(math.floor(randomY)):int(math.ceil(randomY + smallest_area_H)),
                                  int(math.floor(randomX)):int(math.ceil(randomX + smallest_area_W))]
                randomPos = random.randint(0, len(possibleOptions_target) - 1)
                finalY = possibleOptions_target[randomPos][0]
                finalX = possibleOptions_target[randomPos][1]
                image[finalY:finalY + infiltred_image.shape[0],finalX:finalX + infiltred_image.shape[1]] = infiltred_image
            elif n < 5:
                n = n + 1
            times = times + 1
        return image, lines
