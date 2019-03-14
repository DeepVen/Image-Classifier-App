import torch
from torch import nn, optim, utils
from torchvision import datasets, models, transforms
import numpy as np
from collections import OrderedDict
#from utils import model_measure
from prepare import get_data
from load_save_model import save_model
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler


class train_model(object):      
    
    def __init__(self,data_dir,save_dir,arch,learning_rate,hidden_units,epochs,device,dropout):      
        self.data_dir = data_dir 
        self.save_dir = save_dir
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs        
        self.dropout = dropout
        self.device = device
        self.model = None
    
    def validate(self, dataloader, device, criterion, optimizer):    
        
        #defining local variables
        running_loss=0
        running_corrects=0
        
        #set model to eval mode to avoid backprop etc
        self.model.eval()
        
        data_count = 0
        for images,labels in dataloader:     
            
            # sent data to cpu/gpu and pass data through model and get prediction
            images, labels = images.to(device), labels.to(device)
            pred = self.model.forward(images)        
            
            # calculate loss using required criterion    
            loss =  criterion(pred, labels)   

            # use prediction to calculate running loss and accuracy.  
            ps = torch.exp(pred.data)
            _, output = torch.max(ps, 1)
            equality = torch.sum(output == labels.data)
            running_corrects += equality.type_as(torch.FloatTensor())       
            running_loss += loss.item() * images.size(0)

            # to get data size and calc mean at later stage
            data_count += images.size(0)        

        print('val loss:  {:.4f}'.format(running_loss/data_count))
        print('val accuracy: {:.4f}'.format(running_corrects/data_count))    
        
    def model_definition(self):        
                
                        
        #disable gradient calculation for the model - when new classifier is created, gradient calc will be set to true by default
        for params in self.model.parameters():
            params.requires_grad= False                                     
        
        # get number of input features for first conn layer of the classifier
        if self.arch == 'densenet121':    
            num_features = self.model.classifier.in_features       
        else:
            num_features = self.model.classifier[0].in_features       
    
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, 512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=self.dropout)),
                              ('hidden', nn.Linear(512, self.hidden_units)),                       
                              ('fc2', nn.Linear(self.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
        
        self.model.classifier = classifier
        #return model
     
    def train(self, data_loader, device, criterion, optimizer, valid_dataloader, decay_schedule):
        #defining local variables
        running_loss = 0.0        
        running_corrects = 0  
        
        #send model to cpu/gpu as required
        self.model.to(device)
        
        #set decay and turn train mode on
        decay_schedule.step()
        self.model.train()
        
        
        data_count=0
        for images,labels in iter(data_loader):         
            
            # sent data to cpu/gpu and pass data through model and get prediction
            images, labels = images.to(device), labels.to(device)
            pred = self.model.forward(images)         

            # calculate loss using required criterion    
            loss =  criterion(pred, labels)   
            # set optimizer to zero before iteration to prevent calculation getting accumulated
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # use prediction to calculate running loss and accuracy.  
            ps = torch.exp(pred.data)
            _, output = torch.max(ps, 1)
            equality = torch.sum(output == labels.data)
            running_corrects += equality.type_as(torch.FloatTensor())       
            running_loss += loss.item() * images.size(0)
            
            # to get data size and calc mean at later stage
            data_count += images.size(0)        
        
           
        print('train loss:  {:.4f}'.format(running_loss/data_count))
        print('train accuracy {:.4f}'.format(running_corrects/data_count))    

        self.validate(valid_dataloader, device, criterion, optimizer)    
            
        
    def train_execution(self):           
        
        if self.arch == 'vgg19':
            self.model = models.vgg19(pretrained=True)
        elif self.arch == 'vgg16':    
            self.model = models.vgg16(pretrained=True)
        elif self.arch == 'densenet121':    
            self.model = models.densenet121(pretrained=True)
        else:
            print("Sorry {} is not a valid model for this exercise. Please use vgg16, vgg19, or densenet121".format(self.arch))                     
        
        if self.device == 'gpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
                        
        # get the models see loss function and optimizer
        self.model_definition()        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        
        # get images/data
        train_dataloader, valid_dataloader, test_dataloader, train_datasets = get_data(self.data_dir)
        
        # define delay of LR as training goes through
        decay_schedule = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # train and print validation score
        for e in range(self.epochs):
            print('epoch {}/{}'.format(e+1, self.epochs))
            self.train(train_dataloader, self.device, criterion, optimizer,valid_dataloader,decay_schedule)
        
        save_model(self.model, self.save_dir, self.arch, self.hidden_units, self.dropout, self.epochs, self.learning_rate, train_datasets) 
        
