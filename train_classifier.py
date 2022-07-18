'''
Botany Workshop 2022
Will Weaver

www.williamweaver.xyz
www.leafmachine.org

Umiversity of Michigan

Here, we will walk through the process of training an object classifier.
For the demo, we will train the object classifier to identify different
types of rulers that are often found in digitized herbarium images.

Link to a Google Colab implementation: https://colab.research.google.com/drive/1koHbxPoTn_lGU-Y9upCa_BOI3JuD3Wrj?usp=sharing
Save it to your own account first.

First, we neet to make sure that we setup our virtual environment, which 
contains all required python packages. The virtual environment is inside
of the 'venv' folder. To install the venv packages, use:

1) Install python version 3.9.13 :
    Windows: https://www.python.org/downloads/release/python-3913/

2) Install virtualenv:
        python3 -m pip install --user virtualenv
    
3) Create the virtual environment. 
    See: https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html
    It will inherit the default version of python. 
    It's good to check this version after step 5. 
    The 2nd 'venv' is the name of the folder that will contain the 
    virtual environment, can be named differently.

    To make sure you end up with python 3.9.13 :
        python3 -m virtualenv --python C:\Path\To\Python\python.exe venv

    Or if this was the first install of python or you are experienced then simply:
        python3 -m venv venv
        
5) Activate the venv
        ./venv/Scripts/activate

6) Now any packages that we want to install using pip will be inside this venv. 
    Start with installing -r requirements.txt
        pip install -r requirements.txt

7) Then install pytorch
    See: https://pytorch.org/get-started/previous-versions/
        pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

8) If you have trouble with pytorch, you might need to make sure that the version of CUDA 
    installed matches with the version of CUDA that pytorch wants


We will use the lightweight Resnet-18 backbone to construct our convolutional 
neural network (CNN). 
Link to paper: https://arxiv.org/pdf/1512.03385.pdf

Luckily, we can use PyTorch to load the architecture of the CNN to 
give us a starting point.

For the purposes of this workshop we will just use the training data
as the validation data, BUT THIS IS NOT BEST PRACTICE AND WILL RESULT 
IN OVERFITTING AND OTHER PROBLEMS. Ideally, we want to split out groundtruth 
training data into 3 groups: 
1) Training 70%   - for training the ML algorithm
2) Validation 15% - correcting the ML algorithm during training
3) Testing 15%    - unseen images testing "real-world" performance and the ML algorithm's 
                    ability to generalize
Note that the split does not have to be 70/15/15. 

Please see this article for the inspriation behind some of this code:
https://www.pluralsight.com/guides/introduction-to-resnet
Thanks Gaurav Singhal for your well written article!
'''


''' Import the required packages '''
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import *
from torchvision import transforms
from torch.utils.data import DataLoader
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

def preprocessImage(img):
    img_sq = squarify(img,makeSquare=True,sz=360)
    img_t = transforms(img_sq)
    img_tensor = torch.unsqueeze(img_t, 0)
    return img_tensor,img_sq

def detectRuler(net,classes,img,img_tensor,img_sq,img_name,dir_save):
    img_name = img_name.split('.')[0]

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


''' Setting Directories '''
# Directory for our training images
dir_large = os.path.join(os.getcwd(),'data')

# Directory for our training images
dir_small = os.path.join(os.getcwd(),'data_small')

# Directory for our training images
dir_tiny = os.path.join(os.getcwd(),'data_tiny')


''' Pick training dataset '''
# Assign one of the data folder to be the training data
dir_train = dir_tiny


''' Setting Directories '''
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
new_resnet_model = 'model_scripted_resnet_tiny_cpu.pt'

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
n_epochs = 20

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
transforms = transforms.Compose([transforms.ToTensor()])

# Define the datasets using a pytorch class
train_dataset = datasets.ImageFolder(root= dir_train, transform=transforms)
val_dataset = datasets.ImageFolder(root= dir_val, transform=transforms)

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


''' Begin Training '''
if doTrain:
    start = time.time()
    # For loop for each epoch (each loop will process the entire dataset)
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total=0
        print(f'\n')
        print(f"{bcolors.OKCYAN}Epoch {epoch}\n{bcolors.ENDC}")

        # Each batch contains a subset of the entire traiing dataset
        # The number of batches is determined by the size of the trianing
        # dataset divided by the batch size
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            # Sending the images to the cpu or gpu
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()
            
            # Running the batch throught the ML algorithm
            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            # Calculate the training trajectory of for this batch
            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)

            # Print progress
            if (batch_idx) % print_freq == 0:
                print(f"{bcolors.BOLD}     Epoch [{epoch}/{n_epochs}], Step [{batch_idx}/{total_step}], Loss: {loss.item():.4f}{bcolors.ENDC}")
        
        # At the end of the batches we record our progress
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)

        print(f'\n')
        print(f"{bcolors.BOLD}     Epoch {str(epoch)} Summary:{bcolors.ENDC}")
        print(f"{bcolors.BOLD}     Training Loss: {np.mean(train_loss):.4f}, Training Accuracy: {(100 * correct/total):.4f}{bcolors.ENDC}")

        batch_loss = 0
        total_t=0
        correct_t=0

        # This is the validation component
        # Images that were not involved in the training process are used to determine
        # whether the network accuracy improved. If yes, we save the network progress and procede
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (val_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(val_dataloader))
            network_learned = batch_loss < valid_loss_min

            print(f"{bcolors.OKBLUE}     Validation loss: {np.mean(val_loss):.4f}, Validation Accuracy: {(100 * correct_t/total_t):.4f}\n{bcolors.ENDC}")

            # Save the network progress if it has improved
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), os.path.join(dir_model,'resnet_progress.pt'))

                print(f"{bcolors.OKGREEN}     Improvement has been detected. Saving this model.\n{bcolors.ENDC}")

        # This is critical. net.train() makes sure that the dropout layers, batch normalization layers,
        # etc. continue to be modifiable as training procedes. When weights in these layers change, the
        # network will either get better at the classification task or get worse. Once these
        # layers are frozen [ by calling net.eval() ] the weights will not change, allowing us to use the 
        # network to make predicions.
        net.train()

    end = time.time()
    print(f"{bcolors.OKGREEN}Training is complete! :){bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}     Elapsed time: {end - start}\n{bcolors.ENDC}")

    # Now we need to save our trained machine learning network
    # In pytorch there are two ways to save the network:
    #      1. We can wave the weights. This is helpful if you want to retrain the network later
    #      2. We can save a scripted model. This allows us to quickly load our trained model and immediately use it to make predictions
    # We will use option 2 
    model_scripted = torch.jit.script(net) # Export to TorchScript
    model_scripted.save(os.path.join(dir_model,new_resnet_model)) # Save

    print(f"{bcolors.OKGREEN}Model Saved \n{bcolors.ENDC}")