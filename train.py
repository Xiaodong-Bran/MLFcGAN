from __future__ import print_function
import json
import os
import time
from math import log10
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from options import train_option
from data.data import get_dataset_loader
from model.networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from utils import *
from model.msssim import ssim

options = train_option.TrainOptions()
opt = options.parse()
print(opt)

print('===> Loading datasets')

training_data_loader,val_data_loader = get_dataset_loader(opt)

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
criterionSSIM = ssim
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

if opt.resume_netG_path:
    # resume training
    if os.path.isfile(opt.resume_netG_path):
        print("====>loading checkpoint for netG {}".format(opt.resume_netG_path))
        checkpoint = torch.load(opt.resume_netG_path)
        opt.start_epoch = checkpoint['epoch']
        opt.epoch_count = opt.start_epoch
        net_g.load_state_dict(checkpoint['netG_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        net_g_scheduler.load_state_dict(checkpoint['lr_learning_rate'])
    if os.path.isfile(opt.resume_netD_path):
        print("===>loading checkpoint for netD {}".format(opt.resume_netD_path))
        checkpoint = torch.load(opt.resume_netD_path)
        assert opt.start_epoch == checkpoint['epoch']
        net_d.load_state_dict(checkpoint['netD_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
        net_d_scheduler.load_state_dict(checkpoint['lr_learning_rate'])

# so far cpu
G_losses = []
D_losses = []
PSNR_list = []
best_psnr = 0
best_val_loss =1e12

if opt.UsetensorboardX:
    writer = SummaryWriter(comment=opt.comment)

# TRAINING STARTS HERE
for epoch in range(opt.epoch_count, opt.epoch_count+opt.niter + opt.niter_decay):
    # train
    start_epoch_time = time.time()
    net_g.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        if opt.lamb_gan !=0:
            ######################
            # (1) Update D network
            ######################
            optimizer_d.zero_grad()
            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            if opt.loss_method =='WGAN-GP':
                loss_d_fake = pred_fake.mean().unsqueeze(0)
            else:
                loss_d_fake = criterionGAN(pred_fake, False)
            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            if opt.loss_method == 'WGAN-GP':
                loss_d_real = pred_real.mean().unsqueeze(0)
                gradient_penalty = calc_gradient_penalty(net_d, real_ab.data, fake_ab.data,opt,device)
                gradient_penalty.backward(retain_graph=True)
                #########################################
                ## the weights for GP loss;
                #########################################
                loss_d = -loss_d_real + loss_d_fake
            else:
                loss_d_real = criterionGAN(pred_real, True)
                # Combined D loss
                loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()

            optimizer_d.step()
        ######################
        # (2) Update G network
        ######################
        if opt.lamb_gan != 0:
            optimizer_g.zero_grad()
            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab)
            if opt.loss_method == 'WGAN-GP':
                loss_g_gan = pred_fake.mean().unsqueeze(0)
            else:
                loss_g_gan = criterionGAN(pred_fake, True)
        else:
            loss_g_gan = 0.0
        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb_L1
        loss_g = loss_g_gan*opt.lamb_gan + loss_g_l1
        loss_g.backward()

        optimizer_g.step()

    epoch_cost_time = time.time() - start_epoch_time
    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test for val images at every end of the epoch
    net_g.eval()
    avg_psnr = 0
    avg_mse = 0
    avg_ssim = 0
    image_save_flag = True
    for batch in val_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)
        prediction = net_g(input)
        ## for debug/ save input and target image
        if image_save_flag == True:
            predict_img_save = one_image_from_GPU_tensor(prediction)
            predict_img_save.save(options.output_folders['img_folder']+'predict_{}.png'.format(epoch))
            input_save = one_image_from_GPU_tensor(input)
            input_save.save(options.output_folders['img_folder']+'input_{}.png'.format(epoch))
            target_save = one_image_from_GPU_tensor(target)
            target_save.save(options.output_folders['img_folder']+'target{}.png'.format(epoch))
            print('saveing.._epoch:{}..prediction'.format(epoch))
            image_save_flag = False
        mse = criterionMSE(prediction.detach(), target.detach())
        psnr = 10 * log10(1 / mse.item())
        ssim = criterionSSIM(prediction.detach(), target.detach())
        avg_psnr += psnr
        avg_mse += mse
        avg_ssim += ssim
    avg_psnr = avg_psnr / len(val_data_loader)
    avg_mse = avg_mse / len(val_data_loader)
    avg_ssim = avg_ssim / len(val_data_loader)
    print("===> Avg. PSNR: {:.4f}dB/ ssim {:.4f} at epoch {}, time remaining {:.4f} mins".format(avg_psnr,avg_ssim,epoch,epoch_cost_time/60*(opt.niter + opt.niter_decay -epoch)))
    PSNR_list.append(avg_psnr)

    ### save the best validation model
    if avg_mse < best_val_loss:
        net_g_model_out_path = os.path.join(options.output_folders['ckp_folder'], "netG_best_model.pth")
        torch.save({"epoch":epoch,"netG_state_dict": net_g.state_dict(), "optimizer_state_dict": optimizer_g.state_dict(),
                    "lr_learning_rate": net_g_scheduler.state_dict()},
               net_g_model_out_path)
        print('saving the better model with psnr: {:.4f}dB/ ssim: {:.4f}'.format(avg_psnr,avg_ssim))
        best_val_loss = avg_mse

    #save checkpoint per 50 epochs.
    if epoch %50 ==0:
        net_g_model_out_path = os.path.join(options.output_folders['ckp_folder'],"netG_model_epoch_{}.pth".format(epoch))
        net_d_model_out_path = os.path.join(options.output_folders['ckp_folder'],"netD_model_epoch_{}.pth".format(epoch))
        torch.save({"epoch":epoch,"netG_state_dict":net_g.state_dict(),"optimizer_state_dict":optimizer_g.state_dict(),
                    "lr_learning_rate":net_g_scheduler.state_dict()}, net_g_model_out_path)
        torch.save({"epoch":epoch,"netD_state_dict":net_d.state_dict(),"optimizer_state_dict":optimizer_d.state_dict(),
                    "lr_learning_rate":net_d_scheduler.state_dict()}, net_d_model_out_path)
        print("Checkpoint saved to {}".format(options.output_folders['ckp_folder']))

# save log to disk
log_data= {}
log_data['psnr']=PSNR_list
log_data['Loss_G']= G_losses
log_data['Loss_D'] = D_losses
log_file_path = os.path.join(options.output_folders['log_folder']+'data_log.txt')
with open(log_file_path, 'w') as outfile:
    json.dump(log_data, outfile)
print('saveing psnr for val dataset at {}'.format(log_file_path))