'''
See train_classifier.py for more info
'''


''' Import the required packages '''
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import *
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2

''' Define all required functions '''
def printClassLabels(label_names):
    with open(os.path.abspath(label_names)) as f:
        classes = [line.strip() for line in f.readlines()]
        ind = 0
        for cls in classes:
            ind += 1
            print(f'Class {ind}: {cls}')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def validateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def loadClassNames(labelNames):
    with open(os.path.abspath(labelNames)) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def createWhiteBG(img,h,w):
    w_plus = w
    imgBG = np.zeros([h,w_plus,3], dtype=np.uint8)
    imgBG[:] = 255
    imgBG[:img.shape[0],:img.shape[1],:] = img
    return imgBG

def createOverlayBG(img):
    imgBG = np.zeros([450,360,3], dtype=np.uint8)
    imgBG[:] = 0

    imgBG[90:img.shape[0]+90,:img.shape[1],:] = img
    # cv2.imshow('imgBG', imgBG)
    # cv2.waitKey(0)
    return imgBG

def makeImgHor(img):
    # Make image horizontal
    h,w,c = img.shape
    if h > w:
        img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def stackImage(img,squarifyRatio,h,w_plus):
    # cv2.imshow('Original', img)
    wChunk = int(w_plus/squarifyRatio)
    hTotal = int(h*squarifyRatio)
    imgBG = np.zeros([hTotal,wChunk,3], dtype=np.uint8)
    imgBG[:] = 255

    wStart = 0
    wEnd = wChunk
    for i in range(1,squarifyRatio+1):
        wStartImg = (wChunk*i)-wChunk
        wEndImg =  wChunk*i
        
        hStart = (i*h)-h
        hEnd = i*h
        imgBG[hStart:hEnd,wStart:wEnd] = img[:,wStartImg:wEndImg]
    return imgBG

def calcSquarifyRatio(img):
    doStack = False
    h,w,c = img.shape

    # Extend width so it's a multiple of h
    ratio = w/h
    ratio_plus = math.ceil(ratio)
    w_plus = ratio_plus*h

    ratio_go = w/h
    if ratio_go > 4:
        doStack = True

    squarifyRatio = 0
    if doStack:
        # print(f'This should equal 0 --> {w_plus % h}')
        for i in range(1,ratio_plus):
            if ((i*h) < (w_plus/i)):
                continue
            else:
                squarifyRatio = i - 1
                break
        # print(f'Optimal stack_h: {squarifyRatio}')
        while (w % squarifyRatio) != 0:
            w += 1
    return doStack,squarifyRatio,w,h

def squarify(imgSquarify,makeSquare,sz):
    imgSquarify = makeImgHor(imgSquarify)
    doStack,squarifyRatio,w_plus,h = calcSquarifyRatio(imgSquarify)

    if doStack:
        imgBG = createWhiteBG(imgSquarify,h,w_plus)
        imgSquarify = stackImage(imgBG,squarifyRatio,h,w_plus)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)
    return imgSquarify

def preprocessImage(img,trans):
    img_sq = squarify(img,makeSquare=True,sz=360)
    img_t = trans(img_sq)
    img_tensor = torch.unsqueeze(img_t, 0)
    return img_tensor,img_sq

def detectRuler(net,classes,img,img_tensor,img_sq,img_name,dir_save,use_cuda):
    img_name = img_name.split('.')[0]

    if use_cuda:
        out = net(img_tensor.cuda())
    else:
        out = net(img_tensor)

    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentage = round(percentage[index[0]].item(),2)
    pred_class = classes[index[0]]

    imgBG = createOverlayBG(img_sq)
    addText = "True Class: "+str(img_name)
    addText1 = "Pred Class: "+str(pred_class)
    addText2 = "Certainty: "+str(percentage)
    imgOverlay = cv2.putText(img=imgBG, text=addText, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    imgOverlay = cv2.putText(img=imgBG, text=addText1, org=(10, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    imgOverlay = cv2.putText(img=imgOverlay, text=addText2, org=(10, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    cv2.imwrite(os.path.abspath(os.path.join(dir_save,'Prediction_'+img_name+'.jpg')),imgOverlay)

    print(f"{bcolors.BOLD}Image: {img_name}.jpg{bcolors.ENDC}")
    if (img_name.find('fail') != -1):
      print(f"{bcolors.BOLD}     True Class: {str(img_name.split('_')[0])}{bcolors.ENDC}")
      print(f"{bcolors.BOLD}     Pred Class: {pred_class}{bcolors.ENDC}")
      print(f"{bcolors.BOLD}     Certainty: {percentage}{bcolors.ENDC}")
    else:
      print(f"{bcolors.BOLD}     True Class: {str(img_name)}{bcolors.ENDC}")
      print(f"{bcolors.BOLD}     Pred Class: {pred_class}{bcolors.ENDC}")
      print(f"{bcolors.BOLD}     Certainty: {percentage}{bcolors.ENDC}")

    if img_name == pred_class:
      print(f"{bcolors.OKGREEN}     CORRECT!!!\n{bcolors.ENDC}")
    elif (img_name.find('fail') != -1):
      if img_name.split('_')[0] == pred_class:
        print(f"{bcolors.OKGREEN}     CORRECT!!!\n{bcolors.ENDC}")
      else:
        print(f"{bcolors.FAIL}     nope :(\n{bcolors.ENDC}")
    else:
      print(f"{bcolors.FAIL}     nope :(\n{bcolors.ENDC}")
    return pred_class,percentage




''' Train or Evaluate'''
# It's unlikely that the model will complete training on a cpu in our short timeframe
# To train, set doTrain = True
doTrain = True

# We can load a model that I pretrained on the ruler dataset
# and 
doEvaluate = True


''' Setting Directories '''
# Directory for our training images
dir_large = os.path.join(os.getcwd(),'data')

# Directory for our training images
dir_small = os.path.join(os.getcwd(),'data_small')

# Directory for our training images
dir_tiny = os.path.join(os.getcwd(),'data_tiny')

# Assign one of the data folder to be the training data
dir_train = dir_tiny

# Directory to save our final predictions
dir_save = os.path.join(os.getcwd(),'results')
# This function will create dir_save if it does not exist
validateDir(dir_save)

# Set the directory of images to evaluate our moddel
dir_test = os.path.join(os.getcwd(),'test')

# Set the save location for our trained model
dir_model = os.path.join(os.getcwd(),'model')

#  Get the class labels of our dataset
dir_class_names = os.path.join(os.getcwd(),'ruler_classes.txt')
# Load the class names
classes = loadClassNames(dir_class_names)
# See the class label names
printClassLabels(dir_class_names)


''' Setting Model Names '''
# Name our new model, ".pt" is the suffix for model files
# Each time you train, provide a different name to avoide overwriting 
new_resnet_model = 'model_scripted_resnet_Workshop.pt'

# The name of the model used in the "Evaluate Model" section
# Do not change the name below
trained_resnet_model = 'model_scripted_resnet.pt'

# Set the path to our training data
dir_train = dir_train
# Set the path to our validation data (for workshop purposes only, should NOT be the same as the training)
dir_val = dir_train


''' Setting Variables '''
# Define the size of our input images.
# The squarify.py function will make sure that the imput images are 
# the correct size. 
img_size = 360

# If your computer has a gpu, True
# Otherwise, False
use_cuda = False

# Set batch size - the number of images that the ML will 
# train with at a time. Go as big as you can without running out
# of system memory. If it crashes, decrease the size.
# Usually 72, 48, 32, 24, 12 
batch_size = 72

# Learning rate
learning_rate = 1e-3

# How many times the ML will train on the entire dataset. 
# n_epochs = 10+ to get a well-trained network
# Use n_epochs = 1 for testing 
n_epochs = 2

# Set the number of classes
num_classes = 22

# Print progress every n batches
print_freq = 1

# For each input image, we need to transform the image into a tensor image
# We are using a Pytorch class with one transformation --> transforms.ToTensor()
# Other transformations can be stacked, such as image augmentation tools,
# which can be used to increase the size of training datasets and increase
# the variability of inputs that the ML algorithm is exposed to, usually 
# increasing training accuracy and generalization power
# Transforms = transforms.Compose([transforms.ToTensor()])
trans = transforms.Compose([transforms.ToTensor()])

# Define the datasets using a pytorch class
train_dataset = datasets.ImageFolder(root= dir_train, transform=trans)
val_dataset = datasets.ImageFolder(root= dir_val, transform=trans)

# Load the datasets using a pytorch class
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


''' GPU or CPU '''
# Pytorch will use the gpu if it's available, otherwise it will use cpu
# This may conflict with 'use_cuda' from above, so double check
if use_cuda:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    # If you want to use the cpu, you may use the line below
    device = torch.device('cpu')
    

''' Load in the Resnet18 Model '''
net = models.resnet18(pretrained=True)
# If gpu and cuda are available, then the net is sent to the gpu
# If this line causes trouble comment it out
if use_cuda:
    net = net.cuda() if device else net
# Visualize the layers in the ML network
net

# Define the desired loss and optimzer funtions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# Set the number of feature to be the number of classes
# By default resnet will try to detect 1,000 classes
number_of_features = net.fc.in_features
net.fc = nn.Linear(number_of_features, num_classes)
net.fc = net.fc.cuda() if use_cuda else net.fc

# Define some lists to hold training information
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)


''' Evaluate a Trained Model '''
# Use a model trained during the workshop
selected_model = new_resnet_model
# selected_model = trained_resnet_model

# IMPORTANT
# If you are evaluating a model using a cpu that was trained on a GPU (this is the case for the Colab code)
# Then we must map the device
#        see: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html

if use_cuda:
    device = torch.device('cuda')
    map="cuda:0"
else:
    device = torch.device('cpu')
    map = device

if doEvaluate:
    # Load our class names
    classes = loadClassNames(dir_class_names)
    # Get the file names of the images we want to process
    img_list = os.listdir(dir_test)
    
    # Load our trained object classifier network
    net = torch.jit.load(os.path.join(dir_model,selected_model),map_location=map)
    if use_cuda:
        net.to(device)
    # Set the network to evaluate mode
    # This freezes layers that would otherwise change 
    net.eval()

    # We will iterate through our test directory of images 
    for img_name in img_list:
        # Load the image
        # img = Image.open(os.path.join(dir_test,img_name))
        img = cv2.imread(os.path.join(dir_test,img_name))
        # Preprocess the image, create a tensor image to imput into the network
        img_tensor,img_sq = preprocessImage(img,trans)
        # Detect the ruler class, save the output overlay image to dir_save
        pred_class,percentage = detectRuler(net,classes,img,img_tensor,img_sq,img_name,dir_save,use_cuda)