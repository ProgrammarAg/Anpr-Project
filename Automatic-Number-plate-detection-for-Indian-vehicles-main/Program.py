import sys
import glob
import os
import glob
import numpy as np
import cv2
import pymongo
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

import re

states={"AN":"Andaman and Nicobar", "AP":"Andra Pradesh", "AR":"Arunachal Pradesh", "AS":"Assam", "BR": "Bihar", "CG":"Chattisgarh","DL":"Delhi",
        "GA":"Goa","GJ":"Gujarat","HR":"Haryana","HP":"Himachal Pradesh","JK":"Jammu and Kashmir","JH":"Jharkhand","KA":"Karnataka","KL":"Kerala","LD":"Lakshadweep Islands","TN":"Tamil Nadu",
         "MP":"Madhya Pradesh", "MH":"Maharashtra","PB":"Punjab","RJ":"Rajasthan","UP":"Uttar Pradesh","UK/UA":"Uttarakhand","CH":"Chandigarh","CH":"Chandigarh","OT":"Other Territory"}
#Detecting numberplate
def number_plate_detection(img):
    def clean2_plate(plate):
        gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
        _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
        if cv2.waitKey(0) & 0xff == ord('q'):
            pass
        num_contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
        if num_contours:
            contour_area = [cv2.contourArea(c) for c in num_contours]
            max_cntr_index = np.argmax(contour_area)
    
            max_cnt = num_contours[max_cntr_index]
            max_cntArea = contour_area[max_cntr_index]
            x,y,w,h = cv2.boundingRect(max_cnt)
    
            if not ratioCheck(max_cntArea,w,h):
                return plate,None
    
            final_img = thresh[y:y+h, x:x+w]
            return final_img,[x,y,w,h]
    
        else:
            return plate,None
    
    def ratioCheck(area, width, height):
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
            return False
        return True
    
    def isMaxWhite(plate):
        avg = np.mean(plate)
        if(avg>=115):
            return True
        else:
            return False
    
    def ratio_and_rotation(rect):
        (x, y), (width, height), rect_angle = rect
    
        if(width>height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle
    
        if angle>15:
            return False
    
        if height == 0 or width == 0:
            return False
    
        area = height*width
        if not ratioCheck(area,width,height):
            return False
        else:
            return True
    
    
    img2 = cv2.GaussianBlur(img, (5,5), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    img2 = cv2.Sobel(img2,cv2.CV_8U,1,0,ksize=3)	
    _,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = img2.copy()
    cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    num_contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2, num_contours, -1, (0,255,0), 1)
    
    
    
    for i,cnt in enumerate(num_contours):
        min_rect = cv2.minAreaRect(cnt)
        if ratio_and_rotation(min_rect):
            x,y,w,h = cv2.boundingRect(cnt)
            plate_img = img[y:y+h,x:x+w]
            if(isMaxWhite(plate_img)):
                clean_plate, rect = clean2_plate(plate_img)
                if rect:
                    fg=0
                    x1,y1,w1,h1 = rect
                    x,y,w,h = x+x1,y+y1,w1,h1
                    plate_im = Image.fromarray(clean_plate)
                    text = pytesseract.image_to_string(plate_im, lang='eng')
                    return text

#Quick sort
def partition(arr,low,high): 
    i = ( low-1 )         
    pivot = arr[high]    
  
    for j in range(low , high): 
        if   arr[j] < pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 

def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high) 
  
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high)
        
    return arr
 
#Binary search   
def binarySearch (arr, l, r, x): 
  
    if r >= l: 
        mid = l + (r - l) // 2
        if arr[mid] == x: 
            return mid 
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
        else: 
            return binarySearch(arr, mid + 1, r, x) 
    else: 
        return -1
    

print("                  HELLO!!")
print("Welcome to the Number Plate Detection System.\n")
print("Detected Number Plates are :")
array=[]

dir = os.path.dirname(__file__)

for img in glob.glob(dir+"/Dataset/*.jpeg") :
    img=cv2.imread(img)
    number_plate=number_plate_detection(img)
    # print(number_plate)
    
    res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))
    res2=res2.upper()
    
    print(res2)

    img2=cv2.resize(img,(600,600))
    cv2.imshow("Image of car ",img2)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    array.append(res2)
    # print(array)
#Sorting
array=quickSort(array,0,len(array)-1)

client=pymongo.MongoClient("mongodb://localhost:27017/")
# print(client)
db=client['Registed']
collection=db['carNumbers']
abc={'Registered Number Plate': array}
collection.insert_one(abc)


# collection.insert_many(array)



print ("\n")
print("The Vehicle numbers registered (in sorted order) are:-")
for i in array:
    print(i)
print ("\n")    

#Searching
for img in glob.glob(dir+"/search_Image/*.jpeg") :
    img=cv2.imread(img)
    
    number_plate=number_plate_detection(img)
    # print(number_plate)
    res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))

    print("The car number to search is:- ",res2)
    stat=res2[0:2]
# print(states)
# print(states[stat])
    try:
      print('\nCar Belongs to',states[stat])
    except:
      print('State not recognised')        
 
    result = binarySearch(array,0,len(array)-1,res2)
    if result != -1:
       print ("The Vehicle is allowed to visit.\n\n" ) 
    else: 
       print ("The Vehicle is  not allowed to visit.\n\n")
        

    			
















# import sys
# import glob
# import os
# import glob
# import numpy as np
# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# # tess.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
# from PIL import Image
# import re

# #Detecting numberplate
# def number_plate_detection(img):
#     def clean2_plate(plate):
#         gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
#         _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
#         if cv2.waitKey(0) & 0xff == ord('q'):
#             pass
#         num_contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
#         if num_contours:
#             contour_area = [cv2.contourArea(c) for c in num_contours]
#             max_cntr_index = np.argmax(contour_area)
    
#             max_cnt = num_contours[max_cntr_index]
#             max_cntArea = contour_area[max_cntr_index]
#             x,y,w,h = cv2.boundingRect(max_cnt)
    
#             if not ratioCheck(max_cntArea,w,h):
#                 return plate,None
    
#             final_img = thresh[y:y+h, x:x+w]
#             return final_img,[x,y,w,h]
    
#         else:
#             return plate,None
    
#     def ratioCheck(area, width, height):
#         ratio = float(width) / float(height)
#         if ratio < 1:
#             ratio = 1 / ratio
#         if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
#             return False
#         return True
    
#     def isMaxWhite(plate):
#         avg = np.mean(plate)
#         if(avg>=115):
#             return True
#         else:
#             return False
    
#     def ratio_and_rotation(rect):
#         (x, y), (width, height), rect_angle = rect
    
#         if(width>height):
#             angle = -rect_angle
#         else:
#             angle = 90 + rect_angle
    
#         if angle>15:
#             return False
    
#         if height == 0 or width == 0:
#             return False
    
#         area = height*width
#         if not ratioCheck(area,width,height):
#             return False
#         else:
#             return True
    
    
#     img2 = cv2.GaussianBlur(img, (5,5), 0)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
#     img2 = cv2.Sobel(img2,cv2.CV_8U,1,0,ksize=3)	
#     _,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
#     element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
#     morph_img_threshold = img2.copy()
#     cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
#     num_contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
#     cv2.drawContours(img2, num_contours, -1, (0,255,0), 1)
    
    
#     for i,cnt in enumerate(num_contours):
#         min_rect = cv2.minAreaRect(cnt)
#         if ratio_and_rotation(min_rect):
#             x,y,w,h = cv2.boundingRect(cnt)
#             plate_img = img[y:y+h,x:x+w]
#             if(isMaxWhite(plate_img)):
#                 clean_plate, rect = clean2_plate(plate_img)
#                 if rect:
#                     # pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
#                     fg=0
#                     x1,y1,w1,h1 = rect
#                     x,y,w,h = x+x1,y+y1,w1,h1
#                     plate_im = Image.fromarray(clean_plate)
#                     text = pytesseract.image_to_string(plate_im, lang='eng')
#                     return text

# #Quick sort
# def partition(arr,low,high): 
#     i = ( low-1 )         
#     pivot = arr[high]    
  
#     for j in range(low , high): 
#         if   arr[j] < pivot: 
#             i = i+1 
#             arr[i],arr[j] = arr[j],arr[i] 
  
#     arr[i+1],arr[high] = arr[high],arr[i+1] 
#     return ( i+1 ) 

# def quickSort(arr,low,high): 
#     if low < high: 
#         pi = partition(arr,low,high) 
  
#         quickSort(arr, low, pi-1) 
#         quickSort(arr, pi+1, high)
        
#     return arr
 
# #Binary search   
# def binarySearch (arr, l, r, x): 
  
#     if r >= l: 
#         mid = l + (r - l) // 2
#         if arr[mid] == x: 
#             return mid 
#         elif arr[mid] > x: 
#             return binarySearch(arr, l, mid-1, x) 
#         else: 
#             return binarySearch(arr, mid + 1, r, x) 
#     else: 
#         return -1
    

# print("HELLO!!")
# print("Welcome to the Number Plate Detection System.\n")

# array=[]

# dir = os.path.dirname(__file__)
# # print(dir)
# for img in glob.glob(dir+"/Image/*.jpeg"):
#     img=cv2.imread(img)

#     img2 = cv2.resize(img, (600, 600))
#     cv2.imshow("Image of car ",img2)
#     cv2.waitKey(1000)
#     cv2.destroyAllWindows()
   
    
    
#     number_plate=number_plate_detection(img)

#     res2=res2.upper()
#     print(res2)

   

#     array.append(res2)

# #Sorting
# array=quickSort(array,0,len(array)-1)
# print ("\n\n")
# print("The Vehicle numbers registered are:-")
# # print(array)
# for i in array:
#     print(i)
# print ("\n\n")    

# #Searching
# # print(dir)
# # print("The car number to search is:- ")
# for img in glob.glob(dir+"/Search/*.jpeg") :
#     img=cv2.imread(img)
    
#     number_plate=number_plate_detection(img)
#     res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))
#     print("The car number to search is:- ",res2)
    

#     result = binarySearch(array,0,len(array)-1,res2)
#     if result != -1: 
#         print ("\n\nThe Vehicle is allowed to visit." ) 
#     else: 
#         print ("\n\nThe Vehicle is  not allowed to visit.")
        

    			
