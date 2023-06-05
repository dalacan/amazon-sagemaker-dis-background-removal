import torch
import os

# from skimage import io as skio
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from sagemaker_inference import encoder

from data_loader_cache import normalize, im_preprocess

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from sys import getsizeof

class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5],[1.0, 1.0, 1.0])])


def input_fn(request_body, request_content_type):
    import io
    from PIL import Image

    print("input function")
    f = io.BytesIO(request_body)
    
    print("opening image")
    input_image = Image.open(f).convert("RGB")
    
    print("Converting image to np array")
    # input_image = np.array(input_image)
    
    print("torch.divide")
    im, im_shp = im_preprocess(input_image, [1024, 1024])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    print("input function end")
    
    output = {}
    output['image_tensor'] = transform(im).unsqueeze(0)
    output['orig_size'] = shape.unsqueeze(0)
    return output


def output_fn(prediction, content_type):
    print("output function")
    print("Content type: "+str(content_type))
    print("size of prediction")
    print(getsizeof(prediction))

    
    return encoder.encode(prediction, content_type)

def model_fn(model_dir):
    print("Loading model.")
    net = ISNetDIS()
    print("Model loaded.")
    net.to(device)

    print("Loading model weights.")
    net.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))
    return net.to(device)

def predict_fn(data, model):
    inputs_val = data['image_tensor'].type(torch.FloatTensor)
    
    inputs_val_v = Variable(data['image_tensor'], requires_grad=False).to(device) # wrap inputs in Variable
   
    ds_val = model(inputs_val_v)[0] # list of 6 results
    
    pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(data['orig_size'][0][0],data['orig_size'][0][1]),mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need