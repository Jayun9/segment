from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    plt.imshow(img)
    plt.show()


# int(782.5):int(830), int(26.2):int(192.7)
img = cv2.imread("BAGGAGE_20200707_112723_73729_A_10%.jpg")
img_ycbcr = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
mask = np.zeros(img_ycbcr.shape[:2],dtype=np.uint8)
mask[int(782.5):int(830), int(26.2):int(192.7)] = 255
img_ycbcr[mask == 0] = 0
y, cr, cb = cv2.split(img_ycbcr)
equalized = cv2.equalizeHist(y[y!=0])
y[y!=0] = equalized.reshape(equalized.shape[0])
img_ycbcr2 = cv2.merge([y,cr,cb])

dst = cv2.cvtColor(img_ycbcr2,cv2.COLOR_YCrCb2BGR)
dst[mask == 0] =255,255,255
# dst[dst[:,:,0] == 0], dst[dst[:,:,2] == 0], dst[dst[:,:,2] == 0] = 255, 255,255

imshow(dst)