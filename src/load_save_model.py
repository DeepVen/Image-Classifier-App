from torch import nn, optim, utils
from torchvision import datasets, models, transforms
import torch

def save_model(model, save_dir, arch, hidden_units, dropout, epochs, learning_rate, train_datasets):
    
    # set dataset's class index to that of the model so we can save and reload this later as required
    model.class_to_idx = train_datasets.class_to_idx

    model.cpu()
    torch.save({'arch':arch,
                'classifier':model.classifier,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'hidden_units': hidden_units,
                'dropout': dropout,
                'epochs': epochs,
                'learning_rate': learning_rate
                }, save_dir)
    
    print("Model has been saved successfully")

  
def load_checkpoint(checkpoint_path):
    model_meta = torch.load(checkpoint_path)
          
    if model_meta['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_meta['arch'] == 'vgg16':    
        model = models.vgg16(pretrained=True)
    elif model_meta['arch'] == 'densenet121':    
        model = models.densenet121(pretrained=True)
    else:    
        print("Sorry {} is not a valid model for this exercise. Please use vgg16, vgg19, or densenet121".format(arch))   
         
    for param in model.parameters():
            param.requires_grad=False    

    model.classifier = model_meta['classifier']
    model.class_to_idx = model_meta['class_to_idx']
    model.state_dict = model_meta['state_dict']
       

    # it is critical to reverse mapping of class_to_idx as i spent hours trying to identify this missing step
    idx_to_class = { v : k for k,v in model_meta['class_to_idx'].items()}
    return model, idx_to_class        