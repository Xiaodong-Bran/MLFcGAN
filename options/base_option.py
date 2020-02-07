import argparse
import os
import torch
import numpy

class BaseOptions():
    '''
    This class defines the  options used for during training and test time
    '''
    def __init__(self):
        '''
        Reset the class; indicates the class has not been initailized
        '''
        self.initialized = False
    def initialize(self,parser):
        '''
        Define the common options used for both training and test
        :param parser:
        :return:
                '''
        parser.add_argument('--dataset_path', help='path to load datasets')
        parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
        parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=100,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        parser.add_argument('--cuda', action='store_true', help='use cuda?')
        parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
        parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
        parser.add_argument('--upsample',type=bool,default=True,help='the parameter control deconvolution or upsample')
        parser.add_argument('--UsetensorboardX',type=bool,default=False,help='control whether to use tensorboardX for write and disp train/val loss')
        parser.add_argument('--output_dir', default='./training/', type=str,
                            help='path for output: checkpoints/test images/logs')
        parser.add_argument('--netG', default='unetsimple8Global', type=str, help='ResNet/unet/ICRA_2019/SeNetUnet/unetsimple/unetsimple8/unetsimple8Global/unetsimple8GlobalSeNet')
        parser.add_argument('--netD', default='basic', type=str, help='basic/pixel/n_layers')
        parser.add_argument('--lamb', type=int, default=10, help='gradient penalty')
        parser.add_argument('--lamb_L1', type=int, default=10, help='weight on L1 term in objective')
        parser.add_argument('--lamb_gan',type=float,default=0.1,help='weight on gan term in objective')
        parser.add_argument('--loss_method', default='WGAN-GP', type=str, help='gan/lsgan/wgan/')
        parser.add_argument('--use_dropout',default=False,help='whether to use dropout in NetG')
        parser.add_argument("--num_res",default=9,type=int,help='number of the residual blocks in ResNet')
        parser.add_argument('--resume_netG_path',type=str,help='path to resume training for netG')
        parser.add_argument('--checkpoint',type=str,help='path to load checkpoint for netG')
        parser.add_argument('--ssim_loss_weight',default=1,help='weight factor for ssim loss')
        parser.add_argument('--content_loss_weight',default=0,help='weight factor for vgg content loss,default=0,not use.')
        parser.add_argument('--contrast_loss_weight',default=1000,help='weight factor for contrast loss')
        parser.add_argument('--comment',type=str,default='',help='comment name appear on the folder')
        parser.add_argument('--deterministic',type=str,default=True,help='determinstic|benchmark ')
        parser.add_argument('--test_img_folder', type=str, help='test image folders in test mode')
        parser.add_argument('--test_output_path',type=str,help='test output folders save the result')
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        #################################
        # find CUDA
        ############################
        if opt.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if opt.cuda:
            torch.cuda.manual_seed(opt.seed)

        ############################
        # make output_folders
        #############################
        # output folders /netG/netD/loss method/batch_size
        if opt.isTrain:
            self.result_folder = opt.output_dir + 'loss_method_' + opt.loss_method \
                            + '/netG_' + str(opt.netG) \
                            + '/num_res_' + str(opt.num_res) \
                            + '/batch_size_' + str(opt.batch_size) \
                            + '/iter_' + str(opt.niter + opt.niter_decay)  +'/'
            self.output_folders = {}
            self.output_folders['ckp_folder'] = self.result_folder + 'checkpoint/'
            self.output_folders['img_folder'] = self.result_folder + 'images/'
            self.output_folders['log_folder'] = self.result_folder + 'log/'
            # Only in training model creat these folders:
            for k, v in self.output_folders.items():
                if not os.path.exists(v):
                    os.makedirs(v)
            print('result at: '.format(self.result_folder))

        ################################
        # cuda benchmark or determinstic
        ################################
        # cudnn.benchmark = True
        if opt.deterministic == True:
        # region make cudnn deterministic
            torch.manual_seed(opt.seed)
            numpy.random.seed(opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        # endregion
        # torch.cuda.manual_seed_all(opt.seed)
        self.opt = opt
        return self.opt