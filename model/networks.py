import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from collections import namedtuple
from enum import Enum

import torch.nn.functional as F

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            # elif init_type == 'kaiming':
            #     init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netg='ResNet',n_blocks = 2,norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0',upsample=False):
    net = None
    print('netG: {}'.format(netg))
    norm_layer = get_norm_layer(norm_type=norm)
    if netg == 'unet':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout) # num of down: 8
    elif netg == 'UnetSimpleGenerator':
        net = UnetSimpleGenerator(upsample=upsample)
    elif netg == 'unetsimple8':
        net = UnetSimple8Generator(upsample=upsample)
    elif netg == 'unetsimple8Global':
        net = UnetSimple8GlobalGenerator(upsample=upsample)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netg)
    return init_net(net, init_type, init_gain, gpu_id)


class UnetSimpleGenerator(nn.Module):
    '''
    The unet-5-layers-encoder-decoder and the last layers output 8*8
    However,our setting is based on 4-layer-encoder-decoder
    Update: modified the structure to 4 layer
    '''
    def __init__(self,upsample=False,norm_method="BatchNorm"):
        super(UnetSimpleGenerator,self).__init__()
        self.ngf = 64
        self.layer_specs = [
            self.ngf,
            self.ngf * 2,
            # encoder_2: [batch, 128, 128, 64] => [batch, 64, 64, ngf * 2]  #number of filter is the output of the conv2d
            self.ngf * 4,  # encoder_3: [batch, 64, 64, 128] => [batch, 32, 32, ngf * 4]
            self.ngf * 8,  # encoder_4: [batch, 32, 32, 256] => [batch, 16, 16, ngf * 8]
            self.ngf * 8,  # encoder_5: [batch, 16, 16, 512] => [batch, 8, 8, ngf * 8]
            self.ngf * 8,  # encoder_6: [batch, 8, 8, 512] => [batch, 4, 4, ngf * 8]
            self.ngf * 8,  # encoder_7: [batch, 4, 4, 512] => [batch, 2, 2, ngf * 8]
            self.ngf * 8,  # encoder_8: [batch, 2, 2, 512] => [batch, 1, 1, ngf * 8]
        ]
        num_down = 8
        if norm_method == "BatchNorm":
            norm_layer = nn.BatchNorm2d
        if norm_method == "LayerNorm":
            norm_layer = nn.LayerNorm

        self.inc = nn.Sequential(
                     nn.Conv2d(in_channels=3, out_channels=self.layer_specs[0], kernel_size=4, stride=2,padding=1),
                     #nn.BatchNorm2d(layer_specs[0]),
                     #nn.LeakyReLU(0.2,True)
        )

        self.down2 = nn.Sequential(
                     nn.LeakyReLU(0.2, True),
                     nn.Conv2d(in_channels=self.layer_specs[0], out_channels=self.layer_specs[1], kernel_size=4, stride=2,padding=1),
                     nn.BatchNorm2d(self.layer_specs[1]),
        )
        self.down3 = nn.Sequential(
                     nn.LeakyReLU(0.2, True),
                     nn.Conv2d(in_channels=self.layer_specs[1], out_channels=self.layer_specs[2], kernel_size=4, stride=2,padding=1),
                     nn.BatchNorm2d(self.layer_specs[2]),

        )
        self.down4 = nn.Sequential(
                     nn.LeakyReLU(0.2, True),
                     nn.Conv2d(in_channels=self.layer_specs[2], out_channels=self.layer_specs[3], kernel_size=4, stride=2,padding=1),
                     nn.BatchNorm2d(self.layer_specs[3]),

        )
        self.down5 = nn.Sequential(
                     nn.LeakyReLU(0.2, True),
                     nn.Conv2d(in_channels=self.layer_specs[3], out_channels=self.layer_specs[4], kernel_size=4, stride=2,padding=1),
                     nn.BatchNorm2d(self.layer_specs[3]),
        )
        self.down6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.layer_specs[4], out_channels=self.layer_specs[5], kernel_size=4, stride=2,
                      padding=1),
            nn.BatchNorm2d(self.layer_specs[5]),
        )
        self.down7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.layer_specs[5], out_channels=self.layer_specs[6], kernel_size=4, stride=2,
                      padding=1),
            nn.BatchNorm2d(self.layer_specs[6]),
        )
        self.down8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.layer_specs[6], out_channels=self.layer_specs[7], kernel_size=4, stride=2,
                      padding=1),
            #nn.BatchNorm2d(self.layer_specs[7]),
        )
        if upsample == True:
            self.up1 =nn.Sequential(
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(512,512,3,stride=1,padding=0),
                nn.BatchNorm2d(512)
                )
            self.up2 =nn.Sequential(
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(1024, 256, 3, stride=1, padding=0),
                nn.BatchNorm2d(256)
                )

            self.up3 =nn.Sequential(
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(512, 128, 3, stride=1, padding=0),
                nn.BatchNorm2d(128)
                )
            self.up4 =nn.Sequential(
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 64, 3, stride=1, padding=0),
                nn.BatchNorm2d(64)
                )

            self.outc =nn.Sequential(
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, 3, 3, stride=1, padding=0),
                # there is no bn the first and last layer
                nn.Tanh()
                )
            '''
            use nn.upsample instead of convTranspose
            '''
        else:
            self.up1 =nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512)
                )
            self.up2 =nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256)
                )

            self.up3 =nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128)
                )
            self.up4 =nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64)
                )
            self.outc =nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                # there is no bn the first and last layer
                nn.Tanh()
                )

    def forward(self,input):
        # encoder part
        out1 = self.inc(input) # 128*128*64
        out2 = self.down2(out1) # 64*64 *128
        out3 = self.down3(out2) # 32*32 * 256
        out4= self.down4(out3) # 16*16 * 512
        out5 = self.down5(out4) # 8*8 * 512
        # decoder part
        deout1 = self.up1(out5) #16*16*512
        deout2 = self.up2(torch.cat((deout1,out4),dim=1))
        deout3 = self.up3(torch.cat((deout2,out3),dim=1))
        deout4 = self.up4(torch.cat((deout3,out2),dim=1))
        deout5 = self.outc(torch.cat((deout4,out1),dim=1))
        return deout5


class UnetSimple8Generator(UnetSimpleGenerator):
        '''
        Now here we stretch the network into 8 downsamples
        '''

        def __init__(self, upsample=False, norm_method="BatchNorm"):
            super(UnetSimple8Generator, self).__init__()

            self.down6 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=self.layer_specs[4], out_channels=self.layer_specs[5], kernel_size=4, stride=2,
                          padding=1),
                nn.BatchNorm2d(self.layer_specs[5]),
            )
            self.down7 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=self.layer_specs[5], out_channels=self.layer_specs[6], kernel_size=4, stride=2,
                          padding=1),
                nn.BatchNorm2d(self.layer_specs[6]),
            )
            self.down8 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=self.layer_specs[6], out_channels=self.layer_specs[7], kernel_size=4, stride=2,
                          padding=1),
                # nn.BatchNorm2d(self.layer_specs[7]),
            )
            if upsample:
                self.up8 = nn.Sequential(
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(512)
                )
                self.up7 = nn.Sequential(
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(512)
                )
                self.up6 = nn.Sequential(
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(512)
                )
                self.up5 = nn.Sequential(
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(512)
                )
                self.up4 = nn.Sequential(
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(256)
                )
                self.up3 = nn.Sequential(
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(128)
                )
                self.up2 = nn.Sequential(
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(64)
                )
                self.outc = nn.Sequential(
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=0),
                    # there is no bn the first and last layer
                    nn.Tanh()
                )
            else:
                self.up8 = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512)
                )
                self.up7 = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512)
                )
                self.up6 = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512)
                )
                self.up5 = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512)
                )
                self.up4 = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(1024, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(256)
                )
                self.up3 = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(128)
                )
                self.up2 = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(64)
                )
                self.outc = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    # there is no bn the first and last layer
                    nn.Tanh()
                )

        def forward(self, input):
            # encoder part
            out1 = self.inc(input)  # 128*128*64
            out2 = self.down2(out1)  # 64*64*128
            out3 = self.down3(out2)  # 32*32*256
            out4 = self.down4(out3)  # 16*16*512
            out5 = self.down5(out4)  # 8*8*512
            out6 = self.down6(out5)  # 4*4*512
            out7 = self.down7(out6)  # 2*2*512
            out8 = self.down8(out7)  # 1*1*512
            # decoder part
            # see what will happen if we detach the output;
            # well the test result : use detach will change the flow diagram
            # so it is not advisiable to remove the detach
            deout8 = self.up8(out8)  # 2*2*512
            deout7 = self.up7(torch.cat((deout8, out7), dim=1))
            deout6 = self.up6(torch.cat((deout7, out6), dim=1))
            deout5 = self.up5(torch.cat((deout6, out5), dim=1))
            deout4 = self.up4(torch.cat((deout5, out4), dim=1))
            deout3 = self.up3(torch.cat((deout4, out3), dim=1))
            deout2 = self.up2(torch.cat((deout3, out2), dim=1))
            deout1 = self.outc(torch.cat((deout2, out1), dim=1))
            return deout1


class UnetSimple8GlobalGenerator(UnetSimple8Generator):
        def __init__(self, upsample=False, norm_method="BatchNorm"):
            super(UnetSimple8GlobalGenerator, self).__init__()
            # here we need to modify param for the upconv
            if upsample:
                self.up8 = nn.Sequential(
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(512, 512, 3, 1, 0),
                    nn.BatchNorm2d(512)
                )
                self.up7 = nn.Sequential(
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(round(1024 * 1.5), 512, 3, 1, 0),
                    nn.BatchNorm2d(512)
                )
                self.up6 = nn.Sequential(
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(round(1024 * 1.5), 512, 3, 1, 0),
                    nn.BatchNorm2d(512)
                )
                self.up5 = nn.Sequential(
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(round(1024 * 1.5), 512, 3, 1, 0),
                    nn.BatchNorm2d(512)
                )
                self.up4 = nn.Sequential(
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(round(1024 * 1.5), 256, 3, 1, 0),
                    nn.BatchNorm2d(256)
                )
                self.up3 = nn.Sequential(
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(round(512 * 1.5), 128, 3, 1, 0),
                    nn.BatchNorm2d(128)
                )
                self.up2 = nn.Sequential(
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(round(256 * 1.5), 64, 3, 1, 0),
                    nn.BatchNorm2d(64)
                )
                self.outc = nn.Sequential(
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(round(128 * 1.5), 3, 3, 1, 0),
                    # there is no bn the first and last layer
                    nn.Tanh()
                )
            else:
                self.up8 = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512)
                )
                self.up7 = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(round(1024 * 1.5), 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                       bias=False),
                    nn.BatchNorm2d(512)
                )
                self.up6 = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(round(1024 * 1.5), 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                       bias=False),
                    nn.BatchNorm2d(512)
                )
                self.up5 = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(round(1024 * 1.5), 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                       bias=False),
                    nn.BatchNorm2d(512)
                )
                self.up4 = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(round(1024 * 1.5), 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                       bias=False),
                    nn.BatchNorm2d(256)
                )
                self.up3 = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(round(512 * 1.5), 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                       bias=False),
                    nn.BatchNorm2d(128)
                )
                self.up2 = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(round(256 * 1.5), 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                       bias=False),
                    nn.BatchNorm2d(64)
                )
                self.outc = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(round(128 * 1.5), 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                       bias=False),
                    # there is no bn the first and last layer
                    nn.Tanh()
                )
            # self.global_copy_layer_2_2_512 = global_copy_layer(2,2)
            # self.global_copy_layer_4_4_512 = global_copy_layer(4,4)
            # self.global_copy_layer_8_8_512 = global_copy_layer(8, 8)
            # self.global_copy_layer_16_16_512 = global_copy_layer(16,16)
            self.global_copy_layer_32_32_256 = global_copy_layer(512, 32, 32, target=256)
            self.global_copy_layer_64_64_128 = global_copy_layer(512, 64, 64, target=128)
            self.global_copy_layer_128_128_64 = global_copy_layer(512, 128, 128, target=64)

        def forward(self, input):
            # encoder part
            out1 = self.inc(input)  # 128*128*64
            out2 = self.down2(out1)  # 64*64*128
            out3 = self.down3(out2)  # 32*32*256
            out4 = self.down4(out3)  # 16*16*512
            out5 = self.down5(out4)  # 8*8*512

            out6 = self.down6(out5)  # 4*4*512
            out7 = self.down7(out6)  # 2*2*512
            out8 = self.down8(out7)  # 1*1*512

            global_feature = out8  # 1*1*512
            # detach can work but no improvements are seen

            # get the gloabl_feature_copy for different layers

            # modify to use CNN for achieve channel modification

            global_feature_copy_2_2_512 = get_global_copy(global_feature, 2, 2)
            global_feature_copy_4_4_512 = get_global_copy(global_feature, 4, 4)
            global_feature_copy_8_8_512 = get_global_copy(global_feature, 8, 8)
            global_feature_copy_16_16_512 = get_global_copy(global_feature, 16, 16)
            # global_feature_copy_32_32_256 = get_global_copy(global_feature,32,32,shrink=256)
            # global_feature_copy_64_64_128 = get_global_copy(global_feature,64,64,shrink=128)
            # global_feature_copy_128_128_3 = get_global_copy(global_feature,128,128,shrink=64)
            # fuse the global and local copies (the skip connection)

            global_feature_copy_32_32_256 = self.global_copy_layer_32_32_256(global_feature)
            global_feature_copy_64_64_128 = self.global_copy_layer_64_64_128(global_feature)
            global_feature_copy_128_128_64 = self.global_copy_layer_128_128_64(global_feature)

            deout8 = self.up8(out8)  # 2*2*512
            deout7 = self.up7(torch.cat((deout8, out7, global_feature_copy_2_2_512), dim=1))
            deout6 = self.up6(torch.cat((deout7, out6, global_feature_copy_4_4_512), dim=1))
            deout5 = self.up5(torch.cat((deout6, out5, global_feature_copy_8_8_512), dim=1))
            deout4 = self.up4(torch.cat((deout5, out4, global_feature_copy_16_16_512), dim=1))
            deout3 = self.up3(torch.cat((deout4, out3, global_feature_copy_32_32_256), dim=1))
            deout2 = self.up2(torch.cat((deout3, out2, global_feature_copy_64_64_128), dim=1))
            deout1 = self.outc(torch.cat((deout2, out1, global_feature_copy_128_128_64), dim=1))
            return deout1

class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Inconv, self).__init__()
        #self.inconv = nn.Sequential(
        #    nn.ReflectionPad2d(3),
        #    nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
        #              bias=use_bias),
        #    norm_layer(out_ch),
        #    nn.ReLU(True)
        #)
        # remove the padding for determinstic performance
        self.inconv = nn.Sequential(
            #nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3,
                      bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x

# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(in_ch, out_ch,
            #           kernel_size=3, stride=1,
            #           padding=1, bias=use_bias),
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
#        self.outconv = nn.Sequential(
#            nn.ReflectionPad2d(3),
#            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
#            nn.Tanh()
#        )
        self.outconv = nn.Sequential(
#            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3),
            # nn.Tanh()
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.outconv(x)
        # print('last layer with {nn.relu()}');
        return x
def define_D(input_nc, ndf, netd,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    print('norm: {}'.format(norm))
    if netd == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        print('netD: PatchGAN')
    elif netd == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netd == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_id)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class GANLoss(nn.Module):
    def __init__(self, loss_method, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if loss_method == 'lsgan':
            self.loss = nn.MSELoss()
            print('loss method: lsgan-loss')
        if loss_method == 'WGAN-GP':
            print('loss method: WGAN-GP')
        else:
            self.loss = nn.BCELoss()
            print('gan-loss')

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        print("num of down is {}".format(num_downs))
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class global_copy_layer(nn.Module):
    def __init__(self,dims,height,width,target):
        super(global_copy_layer,self).__init__()
        self.height = height
        self.width = width
        self.conv = nn.Conv2d(in_channels=dims,out_channels=target,kernel_size=1,stride=1,padding=0)

    def forward(self,tensor):
        tensor_adjust = self.conv(tensor)
        h, w = self.height, self.width
        n,dims,_,_ = tensor.size()
        concat_t = torch.squeeze(tensor_adjust, dim=2).squeeze(dim=2)  # [N,512,1,1] -> [N,512] / (16,512)
        n, dims = concat_t.size()  # [512]
        if n == 1:
            batch_l = concat_t
        else:
            batch_l = torch.unbind(concat_t, dim=0)
        bs = []
        for batch in batch_l:
            batch = batch.unsqueeze(0).repeat(h, w).view(dims, h, w)
            bs.append(batch)
        result = torch.stack(bs).view(-1, dims, h, w)
        return result

def get_global_copy(tensor,height,width,shrink=None):
    '''
    :param tensor: [N,c,1,1] global tensor
    :param shrink the global copy feature to match the result
    :return: tensor [N,c,h,w] tensor
    '''
    h, w = height, width
    concat_t = torch.squeeze(tensor, dim=2).squeeze(dim=2)  # [N,512,1,1] -> [N,512] / (16,512)

    if shrink  == None:
        pass
    else:
        # we need first shrink the tensor
        # choose the maximum number of the shrink target
        sorted,_ = torch.sort(concat_t,descending=True)
        concat_t = torch.narrow(sorted,1,0,shrink)

    n, dims = concat_t.size()  # [512]
    if n==1:
        batch_l = concat_t
    else:
        batch_l = torch.unbind(concat_t, dim=0)
    bs = []
    for batch in batch_l:
        batch=batch.unsqueeze(0).repeat(h,w).view(dims,h,w)
        bs.append(batch)
    result = torch.stack(bs).view(-1,dims,h,w)

    return result

class residual_block(nn.Module):
    def __init__(self, nb_channels_in, nb_channels_out,last=False):
        super(residual_block,self).__init__()
        if last == False:
            self.conv_block_131 = nn.Sequential(
                # 1*1
                nn.Conv2d(in_channels=nb_channels_in, out_channels=nb_channels_in, kernel_size=1, stride=1,padding=0),
                nn.BatchNorm2d(nb_channels_in),
                nn.LeakyReLU(0.2),
                # 3*3
                nn.Conv2d(in_channels=nb_channels_in, out_channels=nb_channels_out, kernel_size=3, stride=2,padding=1),
                nn.BatchNorm2d(nb_channels_out),
                nn.LeakyReLU(0.2),
                # 1*1--stride2
                nn.Conv2d(in_channels=nb_channels_out, out_channels=nb_channels_out, kernel_size=1, stride=1,padding=0),
                nn.BatchNorm2d(nb_channels_out),
                #nn.LeakyReLU(0.2),
            )
            self.conv_block_down = nn.Sequential(
                nn.Conv2d(in_channels=nb_channels_in, out_channels=nb_channels_out, kernel_size=3, stride=2,padding=1),
                nn.BatchNorm2d(nb_channels_out),
                #nn.LeakyReLU(0.2),
            )
        else: # last layer BN can not handle it well
            self.conv_block_131 = nn.Sequential(
                # 1*1
                nn.Conv2d(in_channels=nb_channels_in, out_channels=nb_channels_in, kernel_size=1, stride=1,padding=0),
                #nn.BatchNorm2d(nb_channels_in),
                nn.LeakyReLU(0.2),
                # 3*3
                nn.Conv2d(in_channels=nb_channels_in, out_channels=nb_channels_out, kernel_size=3, stride=2,padding=1),
                #nn.BatchNorm2d(nb_channels_out),
                nn.LeakyReLU(0.2),
                # 1*1
                nn.Conv2d(in_channels=nb_channels_out, out_channels=nb_channels_out, kernel_size=1, stride=1,padding=0),
                #nn.BatchNorm2d(nb_channels_out),
                nn.LeakyReLU(0.2),
            )
            self.conv_block_down = nn.Sequential(
                nn.Conv2d(in_channels=nb_channels_in, out_channels=nb_channels_out, kernel_size=3, stride=2,padding=1),
                #nn.BatchNorm2d(nb_channels_out),
                #nn.LeakyReLU(0.2),
            )
    def forward(self, x):
        # add and relu the output
        out = self.conv_block_down(x) + self.conv_block_131(x)
        return nn.LeakyReLU(0.2)(out)


class Vgg_loss_network(torch.nn.Module):

    def __init__(self,vgg_model):
        super(Vgg_loss_network,self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    def forward(self, x):
        LossOutput =namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)
