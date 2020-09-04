'''
First code written after summer

NEURAL TRANSFER / STYLE TRANSFER

Caveat:
    I installed pylint maybe this will cause an error?  

TODO:
    Loss functions
        Content loss
        style loss
    Get style model and losses

    Run style transfer
'''

from __future__ import print_function

# Torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image handling and plot libraries
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import copy

# Check wether cuda is available (it'd better be)
device = "cuda" if torch.cuda.is_available() else "cpu"

imsize = 512

loader = transforms.Compose([
    transforms.Resize((imsize,imsize)), # Scale the imported image
    transforms.ToTensor()])    # Transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    
    # We need to fake the batch dimension
    image = loader(image).unsqueeze(0) # to make it (1,r,g,b)
    
    return image.to(device)

style_img= image_loader('images/starry night.jpg')
content_img = image_loader('images/drawing_p.jpg')
assert style_img.size() == content_img.size(), \
        "Resize didn't work as expected"

unloader = transforms.ToPILImage() # Reconvert to PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0) # Remove fake batch dimension
    image = unloader(image)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so plots are updated

#plt.figure()
#imshow(style_img, title='Style image')

#plt.figure()
#imshow(content_img, title='Content image')

# Content loss

class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Now we go deep into the Style loss

def gram_matrix(input):
    a,b,c,d = input.size() # a batch size
                           # b =number of feature maps
                           # (c,d) = dimensions of a f. map
    features = input.view(a*b,c*d) # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t()) # Compute the Gram product
    
    # We 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a*b*c*d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Create a module to normalize input image so we can easily put it in a 
# nn.Sequential

class Normalization(nn.Module): # <--- Is this stupid and I can do it differently?

    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean= torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses:
content_layers_default = ['conv_4']
style_layers_default = ['conv_1','conv_2','conv_3','conv_4','conv_5']

def get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, 
                                style_img, content_img, content_layers=content_layers_default,
                                style_layers = style_layers_default):

    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0 # increment every time we see a conv
    for layer in cnn.children():
        name, layer, i = check_layer_type(layer, i)
        model.add_module(name, layer)

        compute_content_loss(name, content_layers, model, content_img, i, content_losses)

        compute_style_loss(name, style_layers, model, style_img, i, style_losses)

            
    for i in range(len(model) -1 ,-1, -1): # Check what this does
        if  isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses

def check_layer_type(layer, i):
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = 'conv_{}'.format(i)
    elif isinstance(layer,nn.ReLU):
        name = 'relu_{}'.format(i)

        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = 'pool_{}'.format(i)
    elif isinstance(layer, nn.BatchNorm2d):
        name = 'bn_{}'.format(i)
    else:
        raise RuntimeError('unrecognized layer: {}'.format(layer.__class__.__name__))
    return name, layer, i

def compute_style_loss(name, style_layers, model, style_img, i, style_losses):
    if name in style_layers:
        # add content loss:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module("style_loss_{}".format(i), style_loss)       
        style_losses.append(style_loss)

def compute_content_loss(name, content_layers, model, content_img, i, content_losses):
    if name in content_layers:
        # add content loss:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module("content_loss_{}".format(i), content_loss)
        content_losses.append(content_loss)

input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
#plt.figure()
#imshow(input_img, title='Input Image')
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=2000,
                       style_weight=200000, content_weight=1):

    
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
                if run[0] % 300 == 0:  
                    imshow(input_img, title='Output Image')
  
                # sphinx_gallery_thumbnail_number = 4
                    plt.ioff()
                    plt.show()
                    import ipdb; ipdb.set_trace()
                    
            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
