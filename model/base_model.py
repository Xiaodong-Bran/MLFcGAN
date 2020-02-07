import torch
from .networks import  define_G
from .msssim import ssim
from torchvision import models
from .networks import  Vgg_loss_network
from utils import contrast_loss_func

class BaseModel():
    '''
    the class defines the network
    '''
    def __init__(self,opt):

        self.opt = opt
        self.device = torch.device("cuda:0" if opt.cuda else "cpu")
        # params related to model
        self.model_name = opt.netG
        self.model_input_nc = opt.input_nc
        self.model_output_nc = opt.output_nc
        self.model_ngf = opt.ngf
        # params related to optimized
        # params related to loss function and weights

    def create(self,opt):
        # create net work
        self.net = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.num_res, 'batch', opt.use_dropout, \
                         'normal', 0.02, gpu_id= self.device)
        return self.net

    def feed_input(self,x):

        return self.net(x)

    def loss_func(self,opt):

        if int(opt.ssim_loss_weight):
            # add msssim loss
            criterionSSIM = ssim

        if int(opt.content_loss_weight):
            ## build vgg-content loss // many configuration of vgg can be implemented.
            vgg_model = models.vgg16(pretrained=True)
            if torch.cuda.is_available():
                vgg_model.cuda()
            content_loss_net = Vgg_loss_network(vgg_model)
            content_loss_net.eval()

        if int(opt.contrast_loss_weight):
            criterionContrast = contrast_loss_func

    def save(self, opt):
        pass