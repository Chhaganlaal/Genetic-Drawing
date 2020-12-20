import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import string
import random
from util import *

class DNA:

    def __init__(self, bound, img_gradient, brushstrokes_range, canvas=None, sampling_mask=None):
        self.DNASeq = []
        self.bound = bound
        
        #CTRLS
        self.minSize = brushstrokes_range[0] #0.1 #0.3
        self.maxSize = brushstrokes_range[1] #0.3 # 0.7
        self.maxBrushNumber = 4
        self.brushSide = 300 #brush image resolution in pixels
        self.padding = int(self.brushSide*self.maxSize / 2 + 5)
        
        self.canvas = canvas
        
        #IMG GRADIENT
        self.imgMag = img_gradient[0]
        self.imgAngles = img_gradient[1]
        
        #OTHER
        self.brushes = self.preload_brushes('brushes/watercolor/', self.maxBrushNumber)
        self.sampling_mask = sampling_mask
        
        #CACHE
        self.cached_image = None
        self.cached_error = None
        
    def preload_brushes(self, path, maxBrushNumber):
        imgs = []
        for i in range(maxBrushNumber):
            imgs.append(cv2.imread(path + str(i) +'.jpg'))
        return imgs
    
    def gen_new_positions(self):
        if self.sampling_mask is not None:
            pos = util_sample_from_img(self.sampling_mask)
            posY = pos[0][0]
            posX = pos[1][0]
        else:
            posY = int(random.randrange(0, self.bound[0]))
            posX = int(random.randrange(0, self.bound[1]))
        return [posY, posX]
     
    def initRandom(self, target_image, count, seed):
        #initialize random DNA sequence
        for i in range(count):
            #random color
            color = random.randrange(0, 255)
            #random size
            random.seed(seed-i+4)
            size = random.random()*(self.maxSize-self.minSize) + self.minSize
            #random pos
            posY, posX = self.gen_new_positions()
            #random rotation
            '''
            start with the angle from image gradient
            based on magnitude of that angle direction, adjust the random angle offset.
            So in places of high magnitude, we are more likely to follow the angle with our brushstroke.
            In places of low magnitude, we can have a more random brushstroke direction.
            '''
            random.seed(seed*i/4.0-5)
            localMag = self.imgMag[posY][posX]
            localAngle = self.imgAngles[posY][posX] + 90 #perpendicular to the dir
            rotation = random.randrange(-180, 180)*(1-localMag) + localAngle
            #random brush number
            brushNumber = random.randrange(1, self.maxBrushNumber)
            #append data
            self.DNASeq.append([color, posY, posX, size, rotation, brushNumber])
        #calculate cache error and image
        self.cached_error, self.cached_image = self.calcTotalError(target_image)
        
    def get_cached_image(self):
        return self.cached_image
            
    def calcTotalError(self, inImg):
        return self.__calcError(self.DNASeq, inImg)
        
    def __calcError(self, DNASeq, inImg):
        #draw the DNA
        myImg = self.drawAll(DNASeq)

        #compare the DNA to img and calc fitness only in the ROI
        diff = cv2.absdiff(inImg.astype(np.int32), myImg.astype(np.int32))
        totalDiff = np.sum(diff)

        return (totalDiff, myImg)
        
    def drawAll(self, DNASeq):
        #set image to pre generated
        if self.canvas is None: #if we do not have an image specified
            inImg = np.zeros((self.bound[0], self.bound[1]), np.uint8)
        else:
            inImg = np.copy(self.canvas)
        #apply padding
        p = self.padding
        inImg = cv2.copyMakeBorder(inImg, p,p,p,p,cv2.BORDER_CONSTANT,value=[0,0,0])
        #draw every DNA
        for i in range(len(DNASeq)):
            inImg = self.__drawDNA(DNASeq[i], inImg)
        #remove padding
        y = inImg.shape[0]
        x = inImg.shape[1]
        return inImg[p:(y-p), p:(x-p)]       
        
    def __drawDNA(self, DNA, inImg):
        #get DNA data
        color = DNA[0]
        posX = int(DNA[2]) + self.padding #add padding since indices have shifted
        posY = int(DNA[1]) + self.padding
        size = DNA[3]
        rotation = DNA[4]
        brushNumber = int(DNA[5])

        #load brush alpha
        brushImg = self.brushes[brushNumber]
        #resize the brush
        brushImg = cv2.resize(brushImg,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)
        #rotate
        brushImg = self.__rotateImg(brushImg, rotation)
        #brush img data
        brushImg = cv2.cvtColor(brushImg,cv2.COLOR_BGR2GRAY)
        rows, cols = brushImg.shape
        
        #create a colored canvas
        myClr = np.copy(brushImg)
        myClr[:, :] = color

        #find ROI
        y_min = int(posY - rows/2)
        y_max = int(posY + rows/2)
        x_min = int(posX - cols/2)
        x_max = int(posX + cols/2)
        
        # Convert uint8 to float
        foreground = myClr[0:rows, 0:cols].astype(float)
        background = inImg[y_min:y_max,x_min:x_max].astype(float) #get ROI
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = brushImg.astype(float)/255.0
        

        try:
            # Multiply the foreground with the alpha matte
            foreground = cv2.multiply(alpha, foreground)
            
            # Multiply the background with ( 1 - alpha )
            background = cv2.multiply(np.clip((1.0 - alpha), 0.0, 1.0), background)
            # Add the masked foreground and background.
            outImage = (np.clip(cv2.add(foreground, background), 0.0, 255.0)).astype(np.uint8)
            
            inImg[y_min:y_max, x_min:x_max] = outImage
        except:
            print('------ \n', 'in image ',inImg.shape)
            print('pivot: ', posY, posX)
            print('brush size: ', self.brushSide)
            print('brush shape: ', brushImg.shape)
            print(" Y range: ", rangeY, 'X range: ', rangeX)
            print('bg coord: ', posY, posY+rangeY, posX, posX+rangeX)
            print('fg: ', foreground.shape)
            print('bg: ', background.shape)
            print('alpha: ', alpha.shape)
        
        return inImg

        
    def __rotateImg(self, img, angle):
        rows, cols, channels = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        return dst
        
              
    def __evolveDNA(self, index, inImg, seed):
        #create a copy of the list and get its child  
        DNASeqCopy = np.copy(self.DNASeq)           
        child = DNASeqCopy[index]
        
        #mutate the child
        #select which items to mutate
        random.seed(seed + index)
        indexOptions = [0,1,2,3,4,5]
        changeIndices = []
        changeCount = random.randrange(1, len(indexOptions)+1)
        for i in range(changeCount):
            random.seed(seed + index + i + changeCount)
            indexToTake = random.randrange(0, len(indexOptions))
            #move it the change list
            changeIndices.append(indexOptions.pop(indexToTake))
        #mutate selected items
        for changeIndex in changeIndices:
            if changeIndex == 0:# if color
                child[0] = int(random.randrange(0, 255))
            elif changeIndex == 1 or changeIndex == 2:#if pos Y or X
                child[1], child[2] = self.gen_new_positions()
            elif changeIndex == 3: #if size
                child[3] = random.random()*(self.maxSize-self.minSize) + self.minSize
            elif changeIndex == 4: #if rotation
                localMag = self.imgMag[int(child[1])][int(child[2])]
                localAngle = self.imgAngles[int(child[1])][int(child[2])] + 90 #perpendicular
                child[4] = random.randrange(-180, 180)*(1-localMag) + localAngle
            elif changeIndex == 5: #if  brush number
                child[5] = random.randrange(1, self.maxBrushNumber)
        #if child performs better replace parent
        child_error, child_img = self.__calcError(DNASeqCopy, inImg)
        if  child_error < self.cached_error:
            #print('mutation!', changeIndices)
            self.DNASeq[index] = child[:]
            self.cached_image = child_img
            self.cached_error = child_error
        
    def evolveDNASeq(self, inImg, seed):
        for i in range(len(self.DNASeq)):
            self.__evolveDNA(i, inImg, seed)
