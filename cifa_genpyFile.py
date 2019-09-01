import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle


# data directory
DATA_PATH = os.getcwd() + "/data"
TEST_PATH = os.getcwd() + "/test"
RESULT_PATH = os.getcwd() + "/result"


imageSize = 32
imageDepth = 3
debugEncodedImage = False





# convert to binary bitmap given image and write to law output file
def append_python_array( imagePath,imageFile, label):
    img = Image.open(imagePath)
    img = img.resize((imageSize, imageSize), PIL.Image.ANTIALIAS)
    img = (np.array(img))

    r = img[:,:,0].flatten()
    g = img[:,:,1].flatten()
    b = img[:,:,2].flatten()
    
    out = np.array(  list(r) + list(g) + list(b), np.uint8)
    
    labels.append(label)
    datas.append(out)
    filenames.append(imageFile)
    #dict = {'labels':label,'data':img  }
    
    #pickle.dump(dict, outputFile)
#===================================

def gen_python_data(imagepath, resultname,  resultpath):
    
    if not os.path.isdir(resultpath):
        os.mkdir(RESULT_PATH)
    istest = resultname == 'test'
    output = os.path.join(resultpath, resultname)
    try:
        os.remove(output)
    except OSError:
        pass

    label = -1
    totalImageCount = 0
    labelMap = []

    if not istest:
        subDirs = os.listdir(imagepath)
        numberOfClasses = len(subDirs)
         
        
        
        for subDir in subDirs:

            subDirPath = os.path.join(imagepath, subDir)

            # filter not directory
            if not istest and not os.path.isdir(subDirPath):
                continue

            imageFileList = os.listdir(subDirPath)
            label += 1

            print("writing %3d images, %s" % (len(imageFileList), subDirPath))
            totalImageCount += len(imageFileList)
            labelMap.append([label, subDir])

            for imageFile in imageFileList:
                imagePath = os.path.join(subDirPath, imageFile)
                append_python_array(imagePath,imageFile, label)
        print("Total image count: ", totalImageCount)
          
        print("Label MAP: ", labelMap)
    else:
        imageFileList = os.listdir(imagepath)
        for imageFile in imageFileList:
                totalImageCount +=1
                imagePath = os.path.join(imagepath, imageFile)
                append_python_array(imagePath,imageFile, label)         
        print("Test image count: ", totalImageCount)
    
    
    outputFile = open(output, "ab")

    dict = {'filenames':filenames ,'labels':labels,'data':datas,  }
        
    pickle.dump(dict, outputFile)
    outputFile.close()
    
    print("Succeed, Generate the Binary file")
    print("You can find the binary file : ", output)

labels=[]
datas=[]
filenames=[]

gen_python_data(DATA_PATH,'train', RESULT_PATH )

labels=[]
datas=[]
filenames=[]

gen_python_data(TEST_PATH,'test', RESULT_PATH )

