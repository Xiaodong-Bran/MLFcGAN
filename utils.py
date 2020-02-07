import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from functools import reduce
import torch.autograd as autograd
# from data import get_training_set_simplified, get_test_set_simplified,get_test_set,get_training_set

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    #print("Image saved as {}".format(filename))

def convert_img_255(image_tensor):

    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2,0)) + 1) / 2.0 * 255.0
    # should be stayed at double format
    image_numpy = image_numpy.clip(0, 255)
    #image_numpy = image_numpy.astype(np.uint8)
    return image_numpy

def batch_image_from_GPU_tensor(tensor):
    """Scales a CxHxW tensor with values in the range [-1, 1] to [0, 255]"""
    image = tensor.cpu()
    N,C,H,W = image.size()
    # to deal with batch images
    image_list =[]
    for idx in range(N):
        one_image = image[idx,:,:,:]
        one_image = torch.squeeze(one_image,dim=0)
        one_image = 0.5 * image + 0.5  # [-1, 1] --> [0, 1]
        one_image = transforms.ToPILImage()(image)  # [0, 1] --> [0, 255]
        image_list.append(one_image)
    return image_list

def one_image_from_GPU_tensor(tensor):
    """Scales a N*CxHxW tensor with values in the range [-1, 1] to [0, 255]"""
    image = tensor.cpu()
    one_image = image[0,:,:,:]
    one_image = torch.squeeze(one_image,dim=0)
    one_image = 0.5 * one_image + 0.5  # [-1, 1] --> [0, 1]
    one_image = transforms.ToPILImage()(one_image)  # [0, 1] --> [0, 255]
    return one_image

def str2int(s):
    def fn(x,y):
        return x*10+y
    def char2num(s):
        return {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}[s]
    return reduce(fn,map(char2num,s))

# ssim loss function

def channel_expanding(tensor_img):
    '''
    convert one channel tensor image into three channels
    for image content extraction.
    :param tensor_img with size (N,1,H,W)
    :return: tensor_img with size (N,3,H,W)
    '''
    channel_b = tensor_img
    channel_c = tensor_img
    return torch.cat((tensor_img,channel_b,channel_c),dim=1)

def content_loss_func(feature_x,feature_y):
    '''

    :param feature_x:
    :param feature_y:
    :return:
    '''
    total_loss = 0
    mse_loss = torch.nn.MSELoss()
    for i in range(len(feature_x)):
        total_loss += mse_loss(feature_x[i],feature_y[i])
    return total_loss

def distance_loss_func(opt):
    '''

    :param opt: options for distance loss
    :return: specified loss function
    '''
    if opt.distance_loss == 'l1':
        print('distance loss method : l1')
        return torch.nn.L1Loss()
    if opt.distance_loss == 'l2':
        print('distance loss method : l2')
        return torch.nn.MSELoss()

def contrast_loss_func(tensor_img_1,tensor_img_2):
    '''
    calcuate the contrast loss between images
    :param tensor_img_1: [-1,1] range
    :param tensor_img_2: [-1,1] range
    :return: r
    '''
    # transfer from [-1,1] to [0,1]
    tensor_img_1 = (tensor_img_1 + 1) /2
    tensor_img_2 = (tensor_img_2 + 1 )/2
    mean_img_1 = torch.mean(torch.mean(tensor_img_1,dim=2),dim=2)
    mean_img_2 = torch.mean(torch.mean(tensor_img_2,dim=2),dim=2)
    ## sqrt(x) when x is approaching to zero will. feedforward can work
    # but when backward, the deriviate of sqrt(x), when x -> zero will d_dsqrt(x)
    # will approach inf and the model will diverge.
    # img_1  =  tensor_img_1 - mean_img_1
    # img_2 = tensor_img_2 - mean_img_2
    # n,c,h,w = img_1.size()
    # img_1_contrast = torch.sqrt(torch.sum(img_1)**2/(h*w))
    # img_2_contrast = torch.sqrt(torch.sum(img_2)**2/(h*w))
    # img_1_contrast = (torch.sum(img_1))**2/(h*w)
    # img_2_contrast = (torch.sum(img_2))**2/(h*w)
    # print(torch.abs(img_1_contrast - img_2_contrast).item())
    # return torch.abs(img_1_contrast - img_2_contrast)
    return mean_img_2 - mean_img_1

def calc_gradient_penalty(netD, real_data, fake_data,opt,device,constant=1.0):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(device) if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(device) if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * opt.lamb
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lamb
    return gradient_penalty

def get_psnr(im1, im2):
    im1 = np.asarray(im1, dtype=np.float64)
    im2 = np.asarray(im2, dtype=np.float64)
    mse = np.mean(np.square(im1 - im2))
    return 10. * np.log10(np.square(255.) / mse)