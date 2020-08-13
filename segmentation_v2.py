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

    def datasetting(self, json_data): #이미지이름 세그멘테이션 카테고리정보를 ID에 매칭해서 가져오기 위해서 딕셔너리로 따로 저장 
        index = 0#세그멘테이션 별로 이미지가 분류되면 되니 딕셔너리 키를 임의의 인덱스로 두고 값으로 세그멘테이션 정보, 이미지아이디 카티고리 아이디를 가져옴
        for annotations in json_data['annotations']: 
            self.segmentations.setdefault(index, (annotations['segmentation'], annotations['image_id'], annotations['category_id']))
            index += 1
        for images in json_data['images']: #이미지 아이디에 해당하는 이미지네임 매칭
            self.imagename.setdefault(images['id'], images['file_name'])
        for categories in json_data['categories']: #카테고리 아이디에 해당하는 카테고리 네임 매칭
            self.categoryname.setdefault(categories['id'], categories['name'])
        
    def imread(self,imagepath,imagename): #이미지를 읽어옴
        image_path = "{}/{}".format(imagepath, imagename)
        img = cv2.imread(image_path)
        if img is None:
            print("Image load faild")
        return img
        
    def mask_image(self,img,contours): #외곽선 정보를 통해서 마스크를 만들고 세그멘테이션
        img2 = img.copy()
        contours = contours.reshape((contours.shape[0],1,contours.shape[1])) #cv2.fillPoly 형식에 맞춰서 넣어주기 위해서 
        mask = np.zeros(img.shape[:-1],np.uint8)
        cv2.fillPoly(mask, [contours], 255, cv2.LINE_AA)
        img2[mask ==0] = (255,255,255)
        return img2

    def mask_image2(self,img,contours): #segment_V2에 해당하는 기능 mask_image의 마스크의 반전만 다르고 나머지 동일
        img2 = img.copy()
        contours = contours.reshape((contours.shape[0],1,contours.shape[1]))
        mask = np.zeros(img.shape[:-1],np.uint8)
        cv2.fillPoly(mask, [contours], 255, cv2.LINE_AA)
        mask_inv = cv2.bitwise_not(mask)
        img2[mask_inv ==0] = (255,255,255)
        return img2
        
    def return_contours(self,segmt): #json파일에 segment정보를 opencv에 contour정보와 같은 형식으로 맞춰서 리텅해줌
        segmt = np.array(segmt)
        if segmt.shape[0] != 1: #segment정보중에서 정재할때 실수로 완료후 점을 찍은 경우가 있었음. 제일 큰 세그먼트만 처리하기 위해서 
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

    def get_segmentImg2(self):
        return self.segment_image2

    def imshow(self,img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_file(self,image_dic, path): #이미지 딕셔너리를 받아와서 저장
        for category, imageinfo in image_dic.items():
            for imagename, img,_ in imageinfo:
                path_list = imagename.split('.')
                filename = "{}/{}_{}_.{}".format(path,path_list[0],category,path_list[1])
                cv2.imwrite(filename,img)

    def save_file_v2(self,image_dic, path): #이미지 딕셔너리를 받아와서 저장
        for _, imageinfo in image_dic.items():
            for imagename, img, category_list in imageinfo:
                category_name = '_'.join(category_list)
                path_list = imagename.split('.')
                filename = "{}/{}_{}_.{}".format(path,path_list[0],category_name,path_list[1])
                cv2.imwrite(filename,img)

    def segment(self,imagepath): #이미지 세분화  
        for key in self.categoryname.values(): #세분화한 이미지를 카테고리 이름별로 저장하기 위해서 딕셔너리를 만듦
            self.segment_image.setdefault(key)
            self.segment_image[key] =[]
        for index in range(len(self.segmentations)): #세그멘테이션 개수만큼 반복
            seg, image_id, category_id = self.segmentations[index] 
            imagename = self.imagename[image_id]
            category= self.categoryname[category_id]
            img = self.imread(imagepath,imagename)
            contours = self.return_contours(seg)
            if contours is None:
                print("segmnet load falid")     
                continue
            result_img = self.mask_image(img,contours)
            self.segment_image[category].append((imagename, result_img,image_id)) #처리한 이미지를 카테고리 별로 필요한 정보를 append후 저장

    def segment_v2_imagename(self,imagename):
        name_list = imagename.split('.')
        new_imagename = "{}_{}.{}".format(name_list[0],"v2",name_list[1])
        return new_imagename

    def segment_v2(self,imagepath): #이미지 세분화 v2    
        #v2를 실행하려니 한 이미지 안에 세그멘트가 여러개 있는 경우가 있었음, 이미지 id에 해당하는 모든 세멘트를 저장할 필요가 있었음
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
            imagename = self.segment_v2_imagename(imagename)
            category_name = []
            for seg, category_id in seg_category: # 이미지 아이디에 해당하는 세그멘테이션, 카테고리 이름을 하나씩 불러와서 처리 
                if category_id == 21 or category_id == 22: 
                    contours = self.return_contours(seg)
                    img = self.mask_image2(img,contours) #이미지에 중첩해서 처리
                else:
                    category_name.append(self.categoryname[category_id])
            self.segment_image2[image_id].append((imagename, img,category_name)) #segment_image2에 저장

jsonfile = pc.ProcessJSON()
koo = Segmentation()

jsonfile.jsonFileLoad("./.exports", "coco-1594624682.9582064.json")
json_data = jsonfile.json

koo.datasetting(json_data)
# koo.segment("./emblem_image")
koo.segment_v2("./emblem_image")
# koo.save_file(koo.segment_image,'./output')
koo.save_file_v2(koo.segment_image2,'./output2')

