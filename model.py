import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

densenet121 = models.densenet121(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {
    "densenet": {
        "name": densenet121,
        "input": 1024
    },
    "vgg": {
        "name": vgg16,
        "input": 25088
    },
}

def load_model(model_name, checkpoint_path, hidden_units):

    checkpoint = torch.load(checkpoint_path) if checkpoint_path else None

    if checkpoint:
        checkpoint_model = checkpoint['model_name'] 

        model = models[checkpoint_model]["name"] 

        input_layer = models[checkpoint_model]["input"]

        hidden = checkpoint['hidden_layer']
    else :
        model = models[model_name]["name"]

        input_layer = models[model_name]["input"]

        hidden = hidden_units


    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_layer, hidden)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(hidden, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    if checkpoint:
        classifier.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

    model.classifier = classifier

    return model



def save_checkpoint(model_name, model, hidden_units, optimizer, epochs, dataset):
    try:
        checkpoint = {
            'model_name' : model_name,
            'hidden_layer': hidden_units,
            'state_dict': model.classifier.state_dict(),
            'optimizer_state': optimizer.state_dict,
            'epochs': epochs,
            'class_to_idx': dataset.class_to_idx
        }

        torch.save(checkpoint, 'checkpoint.pth')

        print('Model Saved')

    except:
        print('Model is not saved, please try again..')

