import cv2 as cv
import numpy as np
import pandas as pd
import zipfile as zf
import json as js
import random as rd
import io
import os

## ZIP
class ProcessZIP:
    # 생성자
    def __init__(self):
        pass

    # 압축 해제
    def decompress(self, zipFilePath, zipFileName, outFilePath):
        try:
            zipFilePathName = zipFilePath +'/'+zipFileName
            processZIP = zf.ZipFile(zipFilePathName)
            processZIP.extractall(outFilePath)
            processZIP.close()
        except:
            return False
        return True
    
    # 파일 압축
    def compress(self, outFilePath, outFileName, zipFilePath):
        try:
            outFilePathName = outFilePath +'/'+outFileName
            processZIP = zf.ZipFile(outFilePathName, 'w')
            for folder, subfolders, files in os.walk(zipFilePath):
                for file in files:
                    processZIP.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), zipFilePath), compress_type = zf.ZIP_DEFLATED)
            processZIP.close()
        except:
            return False
        return True
    
## IMAGE
class ProcessImage:
    # init
    def __init__(self):
        self.image = None
        self.information = {}

    # get
    def getImage(self):
        return self.image

    def getInformation(self):
        return self.information

    # image 로드 
    def imageFileLoad(self, imageFilePath, imageFileName):
        try:
            imageFilePathName = imageFilePath +'/'+imageFileName
            self.image = cv.imread(imageFilePathName)
            shape = self.image.shape
            self.information = {'path':imageFilePath,'name':imageFileName,'shape':shape}
        except:
            return False
        return True
    
    # image List 로드
    def imageFileListLoad(self, imageFilePath):
        try:
            imageFileList = os.listdir(imageFilePath)
            return imageFileList
        except:
            return False
        return True

    # image 뷰
    def imageFileView(self, image):
        cv.imshow('image', image)
        cv.waitKey(0)
        cv.destroyAllWindows()

## JSON
class ProcessJSON:
    # init
    def __init__(self):
        self.json = None
        self.jsonMerge = None
        self.jsonSplit = []

    # 객체 초기화
    def valueClear(self):
        self.json = None
        self.jsonMerge = None
        self.jsonSplit = []

    # get
    def getJson(self):
        return self.json

    def getJsonSplit(self):
        return self.jsonSplit

    def getJsonMerge(self):
        return self.jsonMerge

    # json 불러오기
    def jsonFileLoad(self, jsonFilePath, jsonFileName):
        try:
            jsonFilePathName = jsonFilePath +'/'+jsonFileName
            processJSON = open(jsonFilePathName,'r')
            self.json = js.load(processJSON)
        except Exception as ex:
            return False
        return True

    # json 파일 분할하기
    def jsonFileSplit(self, jsonFilePath, jsonFileName):
        try:
            jsonFilePathName = jsonFilePath +'/'+jsonFileName
            processJSON = open(jsonFilePathName,'r')
            # 원본 json
            self.jsonMerge = js.load(processJSON)
            self.jsonSplit = []

            #1 대 목록별 분할
            images = self.jsonMerge['images']
            categories = self.jsonMerge['categories']
            annotations = self.jsonMerge['annotations']
            
            #2 이미지를 기준으로 카테고리 및 어노테이션 추가
            for image in images:
                #2.1 dataset 초기화
                oneJson = {'images':[],'categories':[],'annotations':[]}
                #2.2 이미지추가
                oneJson['images'].append(image)

                #2.3 categories & annotations
                for category in categories:
                    for annotation in annotations:
                        #2.4 어노테이션을 기준으로 카테고리와 이미지의 id를 비교하여 둘다 동일하다면
                        if image['id'] == annotation['image_id'] and category['id'] == annotation['category_id']:
                            oneJson['categories'].append(category)
                            oneJson['annotations'].append(annotation)

                self.jsonSplit.append(oneJson)
        except Exception as ex:
            return False
        return True

    # json 분할 데이터 저장하기
    def jsonFileSplitSave(self, jsonFilePath):
        for jsonSplit in self.jsonSplit:
            try:
                #1. 파일명 재명
                jsonFileName = jsonSplit['images'][0]['file_name']+'.json'
                #2. 경로 및 설치 지정
                jsonFilePathName = jsonFilePath +'/'+jsonFileName
                #3. 파일 쓰기 준비
                processJSON = open(jsonFilePathName,'w')
                #4. json 파일 저장
                js.dump(jsonSplit, processJSON)
                #5. 저장 파일 닫기
                processJSON.close()
            except Exception as ex:
                pass
        
    # json 파일 합치기
    def jsonFileMerge(self, jsonFilePath):
        try:
            #1. 파일 리스트 불러오기
            fileList = os.listdir(jsonFilePath)
            
            #2. json 뭉칠 파일 설정
            self.jsonMerge = {'images':[],'categories':[],'annotations':[]}
            for jsonFileName in fileList:
                jsonFilePathName = jsonFilePath +'/'+jsonFileName
                processJSON = open(jsonFilePathName,'r')
                oneJson = js.load(processJSON)
                
                self.jsonMerge['images'].append(oneJson['images'][0])
                for category in oneJson['categories']:
                    check = True
                    #3. 카테고리 중복확인
                    for mergeCategory in self.jsonMerge['categories']:
                        if category == mergeCategory:
                            check = False
                            break

                    if check == True:
                        self.jsonMerge['categories'].append(category)
                
                for annotation in oneJson['annotations']:
                    self.jsonMerge['annotations'].append(annotation)
        except:
            return False
        return True
    
    # json 파일 합치고 자동으로 ID 부여하기
    def jsonFileMergeAndRedefineIndex(self, jsonFilePath, options):
        #0. 전처리 시작점 분할하기
        datasetID = int(options['datasetID'])
        imagesID = int(options['imagesID'])
        categoriesID = int(options['categoriesID'])
        annotationsID = int(options['annotationsID'])
        filePath = options['filePath']
        
        #1. 파일 리스트 불러오기
        fileList = os.listdir(jsonFilePath)
        jsonFileList = [file for file in fileList if file.endswith(".json")]

        self.jsonMerge = {'images':[],'categories':[],'annotations':[]}
        for jsonFileName in jsonFileList:
            jsonFilePathName = jsonFilePath +'/'+jsonFileName
            processJSON = open(jsonFilePathName,'r')
            oneJson = js.load(processJSON)
            
            oneJson['images'][0]['id'] = imagesID
            oneJson['images'][0]['dataset_id'] = datasetID
            oneJson['images'][0]['path'] = filePath + '/' +oneJson['images'][0]['file_name']
            self.jsonMerge['images'].append(oneJson['images'][0])

            for category in oneJson['categories']:
                check = True
                for mergeCategory in self.jsonMerge['categories']:
                    if category == mergeCategory:
                        check = False
                        break

                if check == True:
                    self.jsonMerge['categories'].append(category)
                    
            for annotation in oneJson['annotations']:
                annotation['id'] = annotationsID
                annotation['image_id'] = imagesID
                self.jsonMerge['annotations'].append(annotation)
                annotationsID += 1

            imagesID += 1
        
    # 합친 json 파일 저장
    def jsonFileMergeSave(self, jsonFilePath, jsonFileName):
        try:
            jsonFilePathName = jsonFilePath + '/' + jsonFileName
            #4. json 파일 저장
            processJSON = open(jsonFilePathName,'w')
            #4. json 파일 저장
            js.dump(self.jsonMerge, processJSON)
            #5. 저장 파일 닫기
            processJSON.close()
        except Exception as ex:
            pass

## Viewer
class Viewer:
    def __init__(self):
        pass

    def setImageAndJson(self, image, json):
        #self.image = image
        if len(image.shape) == 3:
            self.image = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
        if len(image.shape) == 2:
            self.image = cv.cvtColor(image, cv.COLOR_GRAY2RGBA)
        self.json = json

    def convertBbox(self,bbox):
        return (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))

    def convertSegmentation(self, segmentation):
        segmentation = np.array(segmentation,dtype=np.int32)
        result = np.reshape(segmentation,( int(len(segmentation[0])/2) , 2))
        return result

    def convertHexToRGBA(self, hexColor):
        red = int(hexColor[1:3],16)
        green = int(hexColor[3:5],16)
        blue = int(hexColor[5:7],16)
        alpha = 0.5
        return (red,green,blue,alpha)

    def save(self,saveFilePath, saveFileName):
        try:
            saveFilePathName = saveFilePath +'/'+saveFileName
            cv.imwrite(saveFilePathName,self.image)
            
        except Exception as ex:
            pass

    def view(self):
        boxTickness = 1
        segmentationTickness = 1
        fontTickness = 1
        fontScale = 0.4
        fontColor = (0,0,0)
        #
        images = self.json['images'][0]
        categories = self.json['categories']
        annotations = self.json['annotations']
        #
        overlay = np.zeros(self.image.shape, dtype = self.image.dtype)
        #
        for category in categories:
            #category['color']
            #category['name']
            #category['id']
            for annotation in annotations:
                #annotation['bbox']
                #annotation['segmentation']
                #annotation['category_id']
                if category['id'] == annotation['category_id']:
                    #category['id']
                    #category['color']
                    #annotation['bbox']
                    name = category['name']
                    color = self.convertHexToRGBA(category['color'])
                    point,size = self.convertBbox(annotation['bbox'])
                    segmentation = self.convertSegmentation(annotation['segmentation'])
                    
                    cv.putText(self.image, name, (point[0],point[1]-10), cv.FONT_ITALIC, fontScale, fontColor, fontTickness)
                    cv.rectangle(self.image, point, size, color, boxTickness)
                    cv.polylines(self.image, [segmentation], True, color, segmentationTickness)
                    
                    #cv.rectangle(overlay, point, size, color, boxTickness)
                    #cv.polylines(overlay, [segmentation], True, color, segmentationTickness)
                    #cv.fillPoly(overlay, [segmentation], color)
                    #cv.addWeighted(overlay, 0.5, self.image, 0.5, 0, self.image)

        cv.imshow('image', self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

## Mask
class ProcessMask:
    
    def __init__(self):
        self.image = None
        self.json = None

    def setImageAndJson(self, image, json):
        #self.image = image
        if len(image.shape) == 3:
            self.image = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
        if len(image.shape) == 2:
            self.image = cv.cvtColor(image, cv.COLOR_GRAY2RGBA)
        self.json = json

    def convertBbox(self,bbox):
        return (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))

    def convertSegmentation(self, segmentation):
        segmentation = np.array(segmentation,dtype=np.int32)
        result = np.reshape(segmentation,( int(len(segmentation[0])/2) , 2))
        return result

    def convertHexToRGBA(self, hexColor):
        red = int(hexColor[1:3],16)
        green = int(hexColor[3:5],16)
        blue = int(hexColor[5:7],16)
        alpha = 0.5
        return (red,green,blue,alpha)

    def selectCategoryAndSave(self,categoryName,filePath,fileName):
        #
        images = self.json['images'][0]
        categories = self.json['categories']
        annotations = self.json['annotations']
        
        #
        shape = self.image.shape
        saveImage = np.full(shape, 255, np.uint8)
                
        #
        oneJson = {'images':[],'categories':[],'annotations':[]}
        oneJson['images'].append(images)

        #
        for category in categories:
            if category['name'] == categoryName:
                oneJson['categories'].append(category)
                for annotation in annotations:
                    if category['id'] == annotation['category_id'] and category['name'] == categoryName:
                        point, size = self.convertBbox(annotation['bbox'])
                        saveImage[point[1]:size[1],point[0]:size[0]] = self.image[point[1]:size[1],point[0]:size[0]]
                        oneJson['annotations'].append(annotation)
        #
        try:
            if len(oneJson['annotations']) != 0:
                filePathName = filePath + '/' + fileName + '.json'
                processJSON = open(filePathName,'w')
                js.dump(oneJson, processJSON)
                processJSON.close()
        except Exception as ex:
            pass
        #
        try:
            if len(oneJson['annotations']) != 0:
                filePathName = filePath + '/' + fileName + '.jpg'
                cv.imwrite(filePathName,saveImage)
        except Exception as ex:
            pass
        
    def selectCategoryAndMaskView(self,categoryName):
        #
        boxTickness = 2
        segmentationTickness = 1
        fontTickness = 1
        fontScale = 0.4
        fontColor = (0,0,0)
        #
        images = self.json['images'][0]
        categories = self.json['categories']
        annotations = self.json['annotations']

        #
        for category in categories:
            #category['color']
            #category['name']
            #category['id']
            for annotation in annotations:
                #annotation['bbox']
                #annotation['segmentation']
                #annotation['category_id']
                if category['id'] == annotation['category_id'] and category['name'] == categoryName:
                    #category['id']
                    #category['color']
                    #annotation['bbox']
                    name = category['name']
                    color = self.convertHexToRGBA(category['color'])
                    point,size = self.convertBbox(annotation['bbox'])
                    segmentation = self.convertSegmentation(annotation['segmentation'])
                    
                    cv.putText(self.image, name, (point[0],point[1]-10), cv.FONT_ITALIC, fontScale, fontColor, fontTickness)
                    cv.rectangle(self.image, point, size, color, boxTickness)
                    cv.polylines(self.image, [segmentation], True, color, segmentationTickness)

        cv.imshow('image', self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def maskView(self):
        boxTickness = 2
        segmentationTickness = 1
        fontTickness = 1
        fontScale = 0.4
        fontColor = (0,0,0)
        #
        images = self.json['images'][0]
        categories = self.json['categories']
        annotations = self.json['annotations']
        #
        for category in categories:
            #category['color']
            #category['name']
            #category['id']
            for annotation in annotations:
                #annotation['bbox']
                #annotation['segmentation']
                #annotation['category_id']
                if category['id'] == annotation['category_id']:
                    #category['id']
                    #category['color']
                    #annotation['bbox']
                    name = category['name']
                    color = self.convertHexToRGBA(category['color'])
                    point,size = self.convertBbox(annotation['bbox'])
                    segmentation = self.convertSegmentation(annotation['segmentation'])
                    
                    cv.putText(self.image, name, (point[0],point[1]-10), cv.FONT_ITALIC, fontScale, fontColor, fontTickness)
                    cv.rectangle(self.image, point, size, color, boxTickness)
                    cv.polylines(self.image, [segmentation], True, color, segmentationTickness)

        cv.imshow('image', self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

##Composition
class ProcessComposition:
    def __init__(self):
        self.backgroundImage = None
        self.backgroundJson = None
        self.specialImage = None
        self.specialJson = None

        self.compositionImage = None
        self.compositionJson = None

    def setBackgroundImageAndJson(self, image, json):
        self.backgroundImage = image
        self.backgroundJson = json
        
    def setSpecialImageAndJson(self, image, json):
        self.specialImage = image
        self.specialJson = json

    def convertCenter(self,bbox):
        return (int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2))

    def convertXYWidthHeight(self,bbox):
        return (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))

    def convertBbox(self,bbox):
        return (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))

    def convertSegmentation(self, segmentation):
        segmentation = np.array(segmentation,dtype=np.int32)
        result = np.reshape(segmentation,( int(len(segmentation[0])/2) , 2))
        return result

    def convertHexToRGBA(self, hexColor):
        red = int(hexColor[1:3],16)
        green = int(hexColor[3:5],16)
        blue = int(hexColor[5:7],16)
        alpha = 0.5
        return (red,green,blue,alpha)

    def randomRange(self,start,end):
        result = int(rd.randint(start,end)/2)
        if rd.randint(0,10) > 5:
            result *= -1
        return result

    def compositionAndSave(self, filePath, fileName):
        #배경 annotaion
        backgroundImageAnnotation = self.backgroundImageAnnotation()
        #특정 개채 annotation 
        specialImageAnnotation = self.specialImageAnnotation()
        #특정 개체 복사
        startSpecialPosition ,endSpecialPosition = self.convertBbox(specialImageAnnotation['bbox'])
        #좌표 정의
        specialX, specialY, specialWidth, specialHeight = self.convertXYWidthHeight(specialImageAnnotation['bbox'])
        #ROI
        specialRoi = self.specialImage[startSpecialPosition[1]:endSpecialPosition[1],startSpecialPosition[0]:endSpecialPosition[0]]
        #특정 개체 중점 찾기
        specialCenterX, specialCenterY = self.convertCenter(specialImageAnnotation['bbox'])
        #배경 중점 찾기
        backgroundCenterX, backgroundCenterY = self.convertCenter(backgroundImageAnnotation['bbox'])
        _, _, backgroundWidth, backgroundHeight = self.convertXYWidthHeight(backgroundImageAnnotation['bbox'])
        #렌덤 좌표 설정 width, height
        randomWidth = self.randomRange(specialWidth,backgroundWidth-specialWidth)
        #randomWidth = 0
        randomHeight = self.randomRange(specialHeight,backgroundHeight-specialHeight)
        #randomHeight = 0
        #배경 중점 에 랜덤좌표 추가
        backgroundCenterX = backgroundCenterX-int(specialWidth/2) + randomWidth
        backgroundCenterY = backgroundCenterY-int(specialHeight/2) + randomHeight
        #합성할 이미지 복사
        self.compositionImage = self.backgroundImage.copy()
        #합성 위치 좌표 지정
        self.compositionImage[
            backgroundCenterY:backgroundCenterY+specialHeight,
            backgroundCenterX:backgroundCenterX+specialWidth
        ] = specialRoi
        
        #annotation 좌표 변경
        #중점이동을 통한 좌표변경 
        movementXposition = backgroundCenterX-specialCenterX+int(specialWidth/2)
        movementYposition = backgroundCenterY-specialCenterY+int(specialHeight/2)

        # 각각의 중점을 구한 후 중점 대칭이동
        # bbox & segmentation
        #print(specialImageAnnotation)
        for i in range(0,len(specialImageAnnotation['segmentation'][0]),2):
            specialImageAnnotation['segmentation'][0][i] += movementXposition
            specialImageAnnotation['segmentation'][0][i+1] += movementYposition

        specialImageAnnotation['bbox'][0] += movementXposition
        specialImageAnnotation['bbox'][1] += movementYposition
        #print(specialImageAnnotation)

        # 다차원
        height = 0
        width = 0

        if len(self.compositionImage) == 3:
            height, width, channel = self.compositionImage.shape

        # 1차원
        if len(self.compositionImage) == 2:
            height, width = self.compositionImage.shape

        # json 재정의
        #"/datasets/200707_emblem/BAGGAGE_20200707_113201_73729_B.jpg"
        oneJson = {
            'images':[
                {
                    'id': 0,
                    'dataset_id': 0,
                    'category_ids': [ ],
                    'path': str(filePath+'/'+fileName+'.jpg'),
                    'width': width,
                    'height': height,
                    'file_name': str(fileName+'.jpg'),
                    'annotated': False,
                    'annotating': [ ],
                    'num_annotations': 0,
                    'metadata': { },
                    'deleted': False,
                    'milliseconds': 0,
                    'events': [ ],
                    'regenerate_thumbnail': False
                }
            ],
            'categories':[],
            'annotations':[]
        }

        # categories
        backgroundCategoires = self.backgroundJson['categories']
        for backgroundCategory in backgroundCategoires:
            oneJson['categories'].append(backgroundCategory)

        specialCategoires = self.specialJson['categories']
        for specialCategory in specialCategoires:
            for oneJsonCategory in oneJson['categories']:
                if oneJsonCategory != specialCategory:
                    oneJson['categories'].append(specialCategory)

        # annotation
        backgroundAnnotations = self.backgroundJson['annotations']
        for backgroundAnnotation in backgroundAnnotations:
            backgroundAnnotation['image_id'] = 0
            oneJson['annotations'].append(backgroundAnnotation)
        specialImageAnnotation['image_id'] = 0
        oneJson['annotations'].append(specialImageAnnotation)
        self.compositionJson = oneJson

        # image 저장
        try:
            filePathName = filePath + '/' + fileName + '.jpg'
            cv.imwrite(filePathName, self.compositionImage)
        except Exception as ex:
            pass

        # json 저장
        try:
            filePathName = filePath + '/' + fileName + '.json'
            processJSON = open(filePathName,'w')
            js.dump(self.compositionJson, processJSON)
            processJSON.close()
        except Exception as ex:
            pass

        #print(oneJson)
        #color = (255,0,0)
        #point,size = self.convertBbox(specialImageAnnotation['bbox'])
        #segmentation = self.convertSegmentation(specialImageAnnotation['segmentation'])
        #segmentation = specialImageAnnotation['segmentation']
        #cv.rectangle(compositionImage, point, size, color, 1)
        #cv.polylines(compositionImage, [segmentation], True, color, 1)
        #cv.imshow('image', compositionImage)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

    def backgroundImageAnnotation(self):
        annotations = self.backgroundJson['annotations']
        bboxSize = 0
        saveMaxAnnotation = 0
        for annotation in annotations:
            width = annotation['bbox'][0]+annotation['bbox'][2]
            height = annotation['bbox'][1]+annotation['bbox'][3]
            #area로 해도되나 가늘고 긴 이미지일 경우 제외
            if width * height > bboxSize:
                saveMaxAnnotation = annotation
        #print(saveMaxAnnotation)

        #2. 해당 값을 통해 bbox 중앙 값 조정
        #centerBackground = [int( saveMaxAnnotation['bbox'][0]+(saveMaxAnnotation['bbox'][2]/2)), int( saveMaxAnnotation['bbox'][1]+(saveMaxAnnotation['bbox'][3]/2) ) ]
        return saveMaxAnnotation

    def specialImageAnnotation(self):
        annotations = self.specialJson['annotations']
        bboxSize = 0
        saveMaxAnnotation = 0
        for annotation in annotations:
            width = annotation['bbox'][0]+annotation['bbox'][2]
            height = annotation['bbox'][1]+annotation['bbox'][3]
            #area로 해도되나 가늘고 긴 이미지일 경우 제외
            if width * height > bboxSize:
                saveMaxAnnotation = annotation
        #print(saveMaxAnnotation)

        #2. 해당 값을 통해 bbox 중앙 값 조정
        #centerBackground = [ int(saveMaxAnnotation['bbox'][2]), int(saveMaxAnnotation['bbox'][3]) ]
        return saveMaxAnnotation

# Refine
class Refine:

    def __init__(self):
        self.initCocoJson()
    
    def initCocoJson(self):
        self.cocoJson = {
            'images':[
                {
                    'id': 0,
                    'dataset_id': 0,
                    'category_ids': [ ],
                    'path': '',
                    'width': 0,
                    'height': 0,
                    'file_name': '',
                    'annotated': False,
                    'annotating': [ ],
                    'num_annotations': 0,
                    'metadata': { },
                    'deleted': False,
                    'milliseconds': 0,
                    'events': [ ],
                    'regenerate_thumbnail': False
                },
            ],
            'categories':[
                {
                    'id': 0,
                    'name': '',
                    'supercategory': '',
                    'color': '',
                    'metadata': { },
                    'keypoint_colors': [ ]
                },
            ],
            'annotations':[
                {
                    'id': 0,
                    'image_id': 0,
                    'category_id': 0,
                    'segmentation': [],
                    'area': 0,
                    'bbox': [],
                    'iscrowd': False,
                    'isbbox': False,
                    'color': '',
                    'metadata': { }
                },
            ]
        }
    
    def convertSegmentation(self, segmentation):
        #print(segmentation[0])
        segmentation = np.array(segmentation[0],dtype=np.int32)
        result = np.reshape(segmentation,( int(len(segmentation)*2) )).tolist()
        #print(result)
        return result

    # 하나의 이미지 & 하나의 라벨 에서 자동으로 라벨을 검출
    def AutoOneByOneAndSave(self, imageFilePath, imageFileName, saveFilePath, options):
        self.initCocoJson()

        lower = options['lower']
        upper = options['upper']
        minRectSize = options['minRectSize']
        name = options['name']
        color = options['color']
        categoryID = options['categoryID']

        # 이미지 불러오기(이미지 차원에 따라)
        imageFilePathName = imageFilePath + '/' + imageFileName
        image = cv.imread(imageFilePathName, cv.IMREAD_COLOR)
        try:
            # 이미지 복사
            contoursImage = image.copy()
            # 이미지 차원 및 크기(height, width, channel)
            imageShape = image.shape
            # 이미지 3차원 -> 1차원 rgb to gray( rgb to binary ) 
            filterRGBToGray = cv.cvtColor(contoursImage, cv.COLOR_RGB2GRAY)
            # 임계값 설정
            _ , threshold = cv.threshold(filterRGBToGray, lower, upper, cv.THRESH_BINARY)
            # 입계값 역치
            thresholdReverse = cv.bitwise_not(threshold)
            # 역치값 확대
            kernel = np.ones((3,3), np.uint8)
            thresholdReverse = cv.dilate(thresholdReverse,kernel,iterations=2)
            #cv.imshow('thresholdReverse', thresholdReverse)
            #cv.waitKey(0)
            #cv.destroyAllWindows()
            
            # 외각 알고리즘 수행(점은 픽셀단위 모든 개수)
            contours = cv.findContours(thresholdReverse, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # 외각 좌표 추출
            x,y,width,height = cv.boundingRect(contours[0][0])
            boundingRect = [x,y,width,height]
            # 영역 크기
            area = cv.contourArea(contours[0][0],False)
            
            # 외각 알고리즘으로 추출된 개수가 1개 이상이고 오로지 하나의 라벨만 존재할때 
            if len(contours) >= 1 and len(contours[0]) == 1 and int(width*height) > minRectSize:
                #view
                #originColor = self.colorHexToRGB(color)
                #reverseColor = self.colorHexToRGBReverse(color)
                
                #cv.drawContours(contoursImage, contours[0],0,originColor,1)
                #cv.rectangle(contoursImage, (x, y),  (x+width, y+height), reverseColor, 1)
                #viewer opencv
                #cv.imshow('image', image)
                #cv.imshow('contoursImage', contoursImage)
                #cv.waitKey(0)
                #cv.destroyAllWindows()
                
                #json
                self.cocoJson['images'][0]['path'] = imageFilePathName
                self.cocoJson['images'][0]['width'] = imageShape[1]
                self.cocoJson['images'][0]['height'] = imageShape[0]
                self.cocoJson['images'][0]['file_name'] = imageFileName
                
                self.cocoJson['categories'][0]['id'] = categoryID
                self.cocoJson['categories'][0]['name'] = name
                self.cocoJson['categories'][0]['color'] = color

                self.cocoJson['annotations'][0]['category_id'] = categoryID
                self.cocoJson['annotations'][0]['segmentation'] = [self.convertSegmentation(contours[0])]
                self.cocoJson['annotations'][0]['area'] = area
                self.cocoJson['annotations'][0]['bbox'] = boundingRect
                self.cocoJson['annotations'][0]['color'] = color

                # Save 
                # image 저장
                try:
                    saveFilePathName = saveFilePath + '/' + imageFileName
                    cv.imwrite(saveFilePathName, image)
                except Exception as ex:
                    pass

                # json 저장
                try:
                    #print(self.cocoJson['images'])
                    #print(self.cocoJson['categories'])
                    #print(self.cocoJson['annotations'])
                    
                    jsonFileName = imageFileName.split('.')[0]+'.json'
                    saveFilePathName = saveFilePath + '/' + jsonFileName
                    processJSON = open(saveFilePathName,'w')
                    js.dump(self.cocoJson, processJSON)
                    processJSON.close()
                except Exception as ex:
                    pass

        except Exception as ex:
            #print('error : ' , ex)
            pass
          
    def colorHexToRGB(self,hexColor):
        result = hexColor.replace('#','')
        red = int(result[:2],16)
        green = int(result[2:4],16)
        blue = int(result[4:6],16)
        alpha = int(255)
        return (blue,green,red,alpha)

    def colorHexToRGBReverse(self,hexColor):
        result = hexColor.replace('#','')
        red = 255-int(result[:2],16)
        green = 255-int(result[2:4],16)
        blue = 255-int(result[4:6],16)
        alpha = int(255)
        return (blue,green,red,alpha)