import process as pc
from cv2 import cv2 as cv
import numpy as np
import copy
import json as js
import os


class Segmentation:
    def __init__(self):
        self.myjson = {}
        self.json_data = {}

    def jsonsplit(self):
        self.images = self.json_data['images'] 
        self.categories = self.json_data['categories']
        self.annotations = self.json_data['annotations']
        self.categoryname = {}
        for categorie in self.categories:
            self.categoryname.setdefault(categorie['id'], categorie['name'])

    def imread(self, imagepath,imagename):
        image_path = "{}/{}".format(imagepath, imagename)
        img = cv.imread(image_path)
        if img is None:
            print("Image load faild")
        return img

    def return_contours(self,segmt):
        segmt = np.array(segmt)
        if segmt.shape[0] != 1:  # segment정보중에서 정재할때 실수로 완료후 점을 찍은 경우가 있었음. 제일 큰 세그먼트만 처리하기 위해서
            segmt_len_list = []
            for seg in segmt:
                segmt_len_list.append(len(seg))
            segmet_index = segmt_len_list.index(max(segmt_len_list))
            segmt = segmt[segmet_index]
            segmt = np.array(segmt)
        segmt = segmt.reshape(segmt.size)
        segmt_x = segmt[::2]
        segmt_y = segmt[1::2]
        contours = np.zeros((len(segmt_x), 2))
        contours[:, 0] = segmt_x.astype(np.int32);
        contours[:, 1] = segmt_y.astype(np.int32)
        contours = contours.astype(np.int32)
        return contours

    def mask_image(self, img, contours):
        img2 = img.copy()
        contours = contours.reshape((contours.shape[0], 1, contours.shape[1]))  # cv.fillPoly 형식에 맞춰서 넣어주기 위해서
        mask = np.zeros(img.shape[:-1], np.uint8)
        cv.fillPoly(mask, [contours], 255, cv.LINE_AA)
        img2[mask == 0] = (255, 255, 255)
        return img2

    def save_json(self, filename, json_save_path, category_id):
        if not os.path.isdir(json_save_path):
            os.mkdir(json_save_path)
        json_name = "{}.json".format(filename)
        file_path = "{}/{}".format(json_save_path,json_name)
        self.myjson["images"][0]["file_name"] = filename
        category = [category for category in self.categories if category["id"] == category_id]
        annotation = [annotation for annotation in self.annotations if annotation["category_id"] == category_id]
        myjson_ = {
            "images" : self.myjson["images"],
            "categories" : category,
            "annotations" : annotation
        }
        with open(file_path, 'w') as outfile:
            js.dump(myjson_, outfile)            

    def save(self, json_save_path,category_id, imagename ,image_save_path, result_img):
        if not os.path.isdir(image_save_path):
            os.mkdir(image_save_path)
        path_list = imagename.split('.')
        category = self.categoryname[category_id]
        filename = "{}_{}_.{}".format(path_list[0], category, path_list[1])
        filepath = "{}/{}".format(image_save_path,filename)
        cv.imwrite(filepath, result_img)
        self.save_json(filename, json_save_path, category_id)

    def segment(self,imagepath,json_save_path,image_save_path):
        for annotation in self.annotations:
            seg, _, category_id = annotation['segmentation'], annotation['image_id'], annotation['category_id']
            if category_id == 21 or category_id == 22:
                imagename = self.images[0]['file_name']
                img = self.imread(imagepath, imagename)         
                contours = self.return_contours(seg)
                result_img = self.mask_image(img, contours)
                self.save(json_save_path, category_id, imagename,image_save_path,result_img)

    def mask_image2(self, img, contours):
        img2 = img.copy()
        contours = contours.reshape((contours.shape[0], 1, contours.shape[1]))
        mask = np.zeros(img.shape[:-1], np.uint8)
        cv.fillPoly(mask, [contours], 255, cv.LINE_AA)
        mask_inv = cv.bitwise_not(mask)
        img2[mask_inv == 0] = (255, 255, 255)
        return img2

    def segment_v2(self,imagepath, json_save_path, image_save_path):
        imagename = self.images[0]["file_name"]
        img = self.imread(imagepath, imagename)
        for annotation in self.annotations:
            seg, category_id = annotation['segmentation'], annotation['category_id']
            if category_id == 21 or category_id == 22:
                contours = self.return_contours(seg)
                img = self.mask_image2(img,contours)
            else:
                file_category_name = category_id
        self.save(json_save_path, file_category_name, imagename ,image_save_path, img)

    def run(self, json_load_path, imagepath, json_save_path,image_save_path):
        json_path_list = os.listdir(json_load_path)
        jsonfile = pc.ProcessJSON()
        for json_name in json_path_list:
            jsonfile.jsonFileLoad(json_load_path,json_name)
            self.json_data = jsonfile.json
            self.myjson = copy.deepcopy(self.json_data)
            self.jsonsplit()
            self.segment(imagepath, json_save_path, image_save_path)
            self.segment_v2(imagepath, json_save_path, image_save_path)



#setting
####################################################################
koo = Segmentation()
koo.run('./exports', './image','./myjson','./output')
####################################################################


