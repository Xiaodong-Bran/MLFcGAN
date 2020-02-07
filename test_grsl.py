from __future__ import print_function
import argparse
import json
import os
import time
from math import log10
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from options import train_option
from data.data import get_training_set, get_test_set,get_dataset_loader
from model.networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from utils import *
from model.msssim import ssim
# Default Training settings
# Training settings
options = train_option.TrainOptions()
opt = options.parse()
print(opt)


predict_img_save_folder = opt.test_output_path

if not os.path.exists(predict_img_save_folder):
    os.makedirs(predict_img_save_folder)
print('===> Loading datasets')

training_data_loader,testing_data_loader = get_dataset_loader(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')

net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,0,'batch', False, 'normal', 0.02, gpu_id=device,upsample=opt.upsample)
if opt.loss_method !='WGAN-GP':
    use_sigmoid = True
    norm = 'batch'
else:
    use_sigmoid = False
    norm = 'instance'
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,norm=norm, gpu_id=device,use_sigmoid=use_sigmoid)

criterionGAN = GANLoss(opt.loss_method).to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
# add msssim loss
criterionSSIM = ssim
# SETUP OPTIMISER
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

# so far cpu
G_losses = []
D_losses = []
PSNR_list = []
best_psnr = 0
best_val_loss =1e12

if opt.UsetensorboardX:
    writer = SummaryWriter(comment=opt.comment)


# test at every epoch
if opt.resume_netG_path:
    # resume training
    if os.path.isfile(opt.resume_netG_path):
        print("====>loading checkpoint for netG {}".format(opt.resume_netG_path))
        checkpoint = torch.load(opt.resume_netG_path)
        # opt.start_epoch = checkpoint['epoch']
        # opt.epoch_count = opt.start_epoch
        net_g.load_state_dict(checkpoint['netG_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
    avg_psnr = 0
    avg_mse = 0
    net_g.eval()
    # image_save_flag = True
    testing_data_loader = training_data_loader
    for idx,batch in enumerate(training_data_loader):
        input, target = batch[0].to(device), batch[1].to(device)
        image_name = training_data_loader.dataset.image_filenames[idx]
    # for input,target in testing_data_loader:
        prediction = net_g(input)
        ## for debug/ save input and target image
        # if image_save_flag == True:
        predict_img_save = one_image_from_GPU_tensor(prediction)
        # predict_img_save.save(options.output_folders['img_folder']+'predict_{}.png'.format(epoch))
        predict_img_save.save(os.path.join(predict_img_save_folder,image_name))
        print('[{}/{}]'.format((idx+1), len(training_data_loader)))

            # input_save = one_image_from_GPU_tensor(input)
            # input_save.save(options.output_folders['img_folder']+'input_{}.png'.format(epoch))
            # target_save = one_image_from_GPU_tensor(target)
            # target_save.save(options.output_folders['img_folder']+'target{}.png'.format(epoch))
            # print('saveing.._epoch:{}..prediction'.format(epoch)
            # image_save_flag = False

        mse = criterionMSE(prediction.detach(), target.detach())
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
        avg_mse += mse
    avg_psnr = avg_psnr / len(testing_data_loader)
    avg_mse = avg_mse / len(testing_data_loader)