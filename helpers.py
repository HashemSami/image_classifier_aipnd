import argparse
import numpy as np
import torch

def get_train_input_args():
    parser = argparse.ArgumentParser()

    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('dataset_path', type = str, 
                        help = 'Path to the dataset folder')

    parser.add_argument('--arch', type = str, default = 'vgg', 
                        help = 'Pretrained model name')

    parser.add_argument('--learning_rate', type = float, default = 0.0003, 
                        help = 'Learning rate value')

    parser.add_argument('--hidden_units', type = int, default = 2048, 
                        help = 'Hidden units value')

    parser.add_argument('--epochs', type = int, default = 1, 
                        help = 'Epochs value')

    parser.add_argument('--checkpoint', type = str, default = None, 
                        help = 'Add checkpoint to retrain the model')

    parser.add_argument('--gpu', type = str, default = 'cuda', 
                        help = 'device type "cpu" or "cuda"')

    

    return parser.parse_args()


def get_predict_input_args():
    parser = argparse.ArgumentParser()

    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('image_path', type = str, 
                        help = 'Path to the image file')
    
    requiredNamed.add_argument('checkpoint', type = str, 
                        help = 'Path to the model\'s ceckpoint file')

    parser.add_argument('--topk', type = int, default = 5, 
                        help = 'Top K values')
    
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'Category names file path')

    parser.add_argument('--gpu', type = str, default = 'cpu', 
                        help = 'device type "cpu" or "cuda"')
    
    return parser.parse_args()



def process_image(im):

    size = 370, 256

    im.thumbnail(size)

    (width, height) = im.size

    y = int((width-224)/2)
    x = int((height-224)/2)

    np_image = np.array(im)

    mean = np_image.mean(axis=(0,1))
    std = np_image.std(axis=(0,1))

    np_cropped = np_image[x:x+224, y:y+224]

    np_cropped = (np_cropped - mean) / std

    t_image = np_cropped.transpose((2, 0, 1))

    return torch.from_numpy(t_image)


