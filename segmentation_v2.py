import process as pc
from cv2 import cv2
import numpy as np
import copy


class Segmentation:
    def __init__(self):
        self.segmentations = {}
        self.categoryname = {}
        self.imagename = {}
        self.segment_image = {}
        self.segment_image2 = {}
        self.key_segmnet = {}
        
    def datasetting(self, json_data):
        index = 0
        for annotations in json_data['annotations']:
            self.segmentations.setdefault(index, (annotations['segmentation'], annotations['image_id'], annotations['category_id']))
            index += 1
        for images in json_data['images']:
            self.imagename.setdefault(images['id'], images['file_name'])
        for categories in json_data['categories']:
            self.categoryname.setdefault(categories['id'], categories['name'])

    def imread(self,imagepath,imagename):
        image_path = "{}/{}".format(imagepath, imagename)
        img = cv2.imread(image_path)
        if img is None:
            print("Image load faild")
        return img
        
    def mask_image(self,img,contours):
        img2 = img.copy()
        contours = contours.reshape((contours.shape[0],1,contours.shape[1]))
        mask = np.zeros(img.shape[:-1],np.uint8)
        cv2.fillPoly(mask, [contours], 255, cv2.LINE_AA)
        img2[mask ==0] = (255,255,255)
        return img2

    def mask_image2(self,img,contours):
        img2 = img.copy()
        contours = contours.reshape((contours.shape[0],1,contours.shape[1]))
        mask = np.zeros(img.shape[:-1],np.uint8)
        cv2.fillPoly(mask, [contours], 255, cv2.LINE_AA)
        mask_inv = cv2.bitwise_not(mask)
        img2[mask_inv ==0] = (255,255,255)
        return img2
        
    def return_contours(self,segmt):
        segmt = np.array(segmt)
        if segmt.shape[0] != 1:
            segmt_len_list = []
            for seg in segmt:
                segmt_len_list.append(len(seg))
            segmet_index = segmt_len_list.index(max(segmt_len_list))
            segmt = segmt[segmet_index] 
            segmt = np.array(segmt)   
        segmt = segmt.reshape(segmt.size)
        segmt_x = segmt[::2]
        segmt_y = segmt[1::2]
        contours = np.zeros((len(segmt_x),2))
        contours[:,0] = segmt_x.astype(np.int32); contours[:,1] = segmt_y.astype(np.int32)
        contours = contours.astype(np.int32)
        return contours

    def get_segmentImg(self):
        return self.segment_image

    def imshow(self,img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_file(self,image_dic, path):
        for category, imageinfo in image_dic.items():
            for imagename, img,_ in imageinfo:
                path_list = imagename.split('.')
                filename = "{}/{}_{}_.{}".format(path,path_list[0],category,path_list[1])
                cv2.imwrite(filename,img)

    def segment(self,imagepath):
        for key in self.categoryname.values():
            self.segment_image.setdefault(key)
            self.segment_image[key] =[]
        for index in range(len(self.segmentations)):
            seg, image_id, category_id = self.segmentations[index]
            imagename = self.imagename[image_id]
            category= self.categoryname[category_id]
            img = self.imread(imagepath,imagename)
            contours = self.return_contours(seg)
            if contours is None:
                print("segmnet load falid")     
                continue
            result_img = self.mask_image(img,contours)
            self.segment_image[category].append((imagename, result_img,image_id))

    def segment_v2(self,imagepath):
        for key in self.imagename.keys():
            self.key_segmnet.setdefault(key)
            self.key_segmnet[key] =[]      
        self.segment_image2 = copy.deepcopy(self.key_segmnet)
        for seg,image_id,category_id in self.segmentations.values():
            self.key_segmnet[image_id].append((seg,category_id))         
        for image_id in self.key_segmnet.keys():
            seg_category = self.key_segmnet[image_id]
            imagename = self.imagename[image_id]
            img = self.imread(imagepath,imagename)
            for seg, category_id in seg_category:
                if category_id == 21 or category_id == 22:
                    contours = self.return_contours(seg)
                    img = self.mask_image2(img,contours)
            self.segment_image2[image_id].append((imagename, img,image_id))

jsonfile = pc.ProcessJSON()
koo = Segmentation()

jsonfile.jsonFileLoad("./.exports", "coco-1594624682.9582064.json")
json_data = jsonfile.json

koo.datasetting(json_data)
# koo.segment("./emblem_image")
koo.segment_v2("./emblem_image")
# koo.save_file(koo.segment_image,'./output')
koo.save_file(koo.segment_image2,'./output2')

