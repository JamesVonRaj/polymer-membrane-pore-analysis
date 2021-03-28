import pytesseract
import numpy as np
import cv2
import matplotlib.pylab as plt
import imutils


#--------------------------------- clean image -------------------------------#
effective_Hight_max = 895 # crop bottom pixels
effective_Hight_min = 100 # crop top pixel

img = cv2.imread('test_pore2.bmp') # image to analyze
plt.figure(dpi=400)
plt.imshow(img)

crop_img = img[effective_Hight_min:effective_Hight_max, :, :] # crop out text border
height, width, channels = crop_img.shape

gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY ) # convert to greyscale
contrast_img = cv2.equalizeHist(gray_img)

blur_kernel = (2,2) #blur gaussian dist width
smooth_img = cv2.blur(contrast_img, blur_kernel) #smooths out image w/ gaussian blurring

# otsu threshholding automatically determines threshhold averages
ret,bi_img = cv2.threshold(smooth_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

inverted_img = cv2.bitwise_not(bi_img) 

kernel = np.ones((2,2),np.uint8)
clean_img = cv2.dilate(inverted_img,kernel,iterations = 1) #dilate pores
#-----------------------------------------------------------------------------#


#-------------------------------- get contours -------------------------------#
bordersize = 3
border_img = cv2.copyMakeBorder( #add white border around image to get contours
    clean_img,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0])

edged = cv2.Canny(border_img,50,100) #gives edges/contour plot
plt.figure(dpi=400)
plt.imshow(edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
mask = np.ones(border_img.shape[:2], dtype="uint8") * 255
areas = [cv2.contourArea(c) for c in cnts] # list of areas of each contour

for c in cnts:
	if cv2.contourArea(c) == max(areas) and max(areas) > 10000:
        # removes largest contour because it is the top block of pore that is not real
		cv2.drawContours(mask, [c], -1, 0, -1) 
    
image = cv2.bitwise_and(border_img, border_img, mask=mask)
h, w = image.shape[:2]

# removes border previously added to get origonal image size
finished_image = image[bordersize:h - bordersize, bordersize:w - bordersize]

np.save('clean.npy',finished_image) # gives cleaned image for analysis
#-----------------------------------------------------------------------------#


#------------------------- get micron scaling value --------------------------#

# NOTE! If you do not want to use this section of code to extract the scaling
# label, you can input it yourself in line 90.

tess_img = img[915:960,1050:1280] #Location of micron text. Change with respect 
                                  #to which image you use.  

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

gray = cv2.cvtColor(tess_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - tess_img # tess_img normally is 'opening'

microns = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
microns = float(microns.replace('um', ''))
#-----------------------------------------------------------------------------#


#---------------------------- find scaling factor ----------------------------#

# This section finds the pixel length associated with the scaling label in the
# bottom of the image. 

# cropped_img must be manually adjusted to only include the scaling tick marks.
cropped_img = img[909:910,800:1200] #cropped scaling tick marks
scale = np.array(cropped_img)
scale = scale[0][:,0]
pixel_scale = np.where(scale == 255)
pix_width = pixel_scale[0][-1] - pixel_scale[0][0]
scaling_factor = microns/pix_width
#-----------------------------------------------------------------------------#


#------------------ build weighted average pore size array -------------------#
clr = np.load('clean.npy')
plt.figure(dpi=400)
plt.imshow(clr)

# Gives both domain dize and number or domains for each film depth pixel value.
avg_pore_size = []
for i in range(len(clr)): # iterating through the y axis pixel values
    pix = 0
    X = [] # array of pore sizes which are pixel lengths
    for j in range(clr.shape[1] - 1): # iterating through x axis pixel values
        #If jth pixel is a pore pixel and j+1 pixel is also a pore pixel, add one to pore size.       
        if clr[i,j] == 255 and clr[i,j+1] == 255: 
            
            pix = pix + 1                 
            
        if clr[i,j] == 255 and clr[i,j+1] == 0: #identifies domain boundary
            pix = pix + 1
            X.append(pix) # adds pore size array of a single domain in the slice
            pix = 0
            
        if j == clr.shape[1] - 2: 
            X = np.array(X)
            wgt = X # weight array for creating weighted average
            X_w = wgt*X

            wgt_sum = np.sum(wgt)
            if wgt_sum == 0:
                wgt_sum = np.nan
            X_w_sum = np.sum(X_w)
            if X_w_sum == 0:
                X_w_sum = np.nan
            
            W = X_w_sum/wgt_sum # weighted average
            avg_pore_size.append(W)
#-----------------------------------------------------------------------------#            


#-------------------------- plot average pore size ---------------------------#
avg_pore_size = np.array(avg_pore_size)
avg_pore_size[np.isinf(avg_pore_size)] = 0
avg_pore_size = np.nan_to_num(avg_pore_size)*scaling_factor

plt.figure(dpi=400,figsize=(4,6))
film_depth = np.arange(effective_Hight_max - effective_Hight_min)*scaling_factor

#create least squares fit
ones = np.ones(effective_Hight_max - effective_Hight_min)
A = np.column_stack((film_depth,ones))
AT = np.transpose(A)
b = avg_pore_size

# gives slope and y-intercept
m,yo = np.matmul(np.matmul(np.linalg.inv(np.matmul(AT,A)),AT),b) #linear algebra
y = m*film_depth + yo

plt.plot(avg_pore_size, film_depth,'.', color = '#fd8d3c')
plt.plot(y,film_depth,color = '#e31a1c',linewidth=3)
plt.axis([0,max(avg_pore_size) + 0.25,max(film_depth) + 0.5,0])
plt.xlabel(u'Domain Size (${\mu}m$)',fontsize=16)
plt.ylabel(u'Film Depth (${\mu}m$)',fontsize=16)
#-----------------------------------------------------------------------------#
