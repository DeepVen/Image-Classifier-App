import argparse
import torch
import torch.nn.functional as F
from load_save_model import load_checkpoint
from PIL import Image
from torchvision import models, transforms
import json
import numpy as np
from torch.autograd import Variable

ap = argparse.ArgumentParser()
ap.add_argument('input_img', default='/home/workspace/aipnd-project/flowers/test/73/image_00320.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/checkpoint_final_v1.pth', action="store",type = str)
ap.add_argument('--top_k', dest="top_k", default=5,  action="store", type=int)
ap.add_argument('--category_names', default='/home/workspace/aipnd-project/cat_to_name.json', dest="category_names",  action="store")
ap.add_argument('--gpu', dest="device", default="gpu", action="store", type=str)

arg_data = ap.parse_args()
input_img = arg_data.input_img
checkpoint_path = arg_data.checkpoint
top_k = arg_data.top_k
category_names = arg_data.category_names
device = arg_data.device


def process_image(image):    
    
    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))    
    return npImage


def predict(image_path, model, idx_to_class, topk, device):    
    for i in image_path:
        pass
    # once we read image using PIL convert to a float tensor for further processing
    image = torch.FloatTensor([process_image(Image.open(i))])
    
    # set model to eval model  
    model.eval()

    #move model to cpu/cpu as required and run image through the model
    if torch.cuda.is_available and device=='gpu':
        output = model.forward(Variable(image.cuda()))
    else:
        output = model.forward(Variable(image))
    
    #get prob of each prediction
    probabilities = torch.exp(output.cpu()).data.numpy()[0]

    # get topk prob and identified respective classes
    top_idx = np.argsort(probabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = probabilities[top_idx]

    return top_probability, top_class


# get model from checkpoint file
model, idx_to_class = load_checkpoint(checkpoint_path) 
# move model to cpu/gpu as required
if device=='gpu':
    if torch.cuda.is_available:
        model = model.cuda()
        print ("Using GPU")
    else:
        print("Using CPU since GPU is not available")
else:
    print("Using CPU")
    
# get category mapping information    
with open(category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

# run images through model and get prediction probabilities
top_probability, top_class = predict(input_img, model, idx_to_class, top_k, device)  

cat_x = [cat_to_name[x] for x in top_class]

count=0
while count < top_k:
    print("model has predicted -- {} -- as the label with a probability of {}".format(cat_x[count], top_probability[count]))
    count += 1

    
