from __future__ import print_function
import os
import torch.optim as optim
from options import test_option
from data.data import get_test_images_in_test_mode
from model.networks import define_G, get_scheduler
from utils import *

options = test_option.TestOptions()
opt = options.parse()
# print(opt)

predict_img_save_folder = opt.test_output_path

if not os.path.exists(predict_img_save_folder):
    os.makedirs(predict_img_save_folder)
print('===> Loading datasets')

testing_data_loader = get_test_images_in_test_mode(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,0,'batch', False, 'normal', 0.02, gpu_id=device,upsample=opt.upsample)
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)

if opt.resume_netG_path:
    if os.path.isfile(opt.resume_netG_path):
        print("====>loading checkpoint for netG {}".format(opt.resume_netG_path))
        checkpoint = torch.load(opt.resume_netG_path)
        net_g.load_state_dict(checkpoint['netG_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])

    net_g.eval()
    for idx,batch in enumerate(testing_data_loader):
        input = batch.to(device)
        image_name = testing_data_loader.dataset.image_filenames[idx]
        prediction = net_g(input)
        predict_img_save = one_image_from_GPU_tensor(prediction)
        predict_img_save.save(os.path.join(predict_img_save_folder,image_name))

        print('[{}/{}]'.format((idx+1), len(testing_data_loader)))
    print('*'*20)
    print('result at :{}'.format(predict_img_save_folder))
    print('*'*20)