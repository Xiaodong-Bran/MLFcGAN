from os import listdir
from os.path import join
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    '''
    The class deals with loading images from the dataset loader
    1) a->b or b->a two direction
    2) read RGB images
    3) image was resized to 286/286
    4) image was then random cropped into 256/256 and random clip

    '''
    def __init__(self, image_dir, direction,mode):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.mode= mode
        self.a_path = join(image_dir, "source")
        self.b_path = join(image_dir, "target")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        if (self.mode == 'train'):
            a = a.resize((286, 286), Image.BICUBIC)
            b = b.resize((286, 286), Image.BICUBIC)

            w_offset = np.random.randint(0, max(0, 286 - 256 - 1))
            h_offset = np.random.randint(0, max(0, 286 - 256 - 1))

        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)

        if (self.mode=='train'):
            a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

            #for left-right-flip operation
        if (self.mode=='train'):
            if np.random.random() < 0.5:
                 idx = [i for i in range(a.size(2) - 1, -1, -1)]
                 idx = torch.LongTensor(idx)
                 a = a.index_select(2, idx)
                 b = b.index_select(2, idx)

        if self.direction == "source2target":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolder_simplified(data.Dataset):
    '''
    The class deals with loading images from the dataset loader
    1) a->b or b->a two direction ---> CHANGED from source domain into target domain.
    2) read RGB images ---> CHANGED into read grayscale images: for training
    3) image was resized to 286/286 ---->REMOVE the data agumentation
    4) image was then random cropped into 256/256  random --->REMOVE the data agumentation
    5) image was randomlly left-right flip for data_agumentation-->REMOVE.
    '''

    def __init__(self, image_dir):
        super(DatasetFromFolder_simplified, self).__init__()
        self.source_path = join(image_dir, "source")
        self.target_path = join(image_dir, "target")
        self.image_filenames = [x for x in listdir(self.source_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        source_img = Image.open(join(self.source_path, self.image_filenames[index]))
        target_img = Image.open(join(self.target_path, self.image_filenames[index]))
        # a = a.resize((286, 286), Image.BICUBIC)
        # b = b.resize((286, 286), Image.BICUBIC)
        source_img = transforms.ToTensor()(source_img)
        target_img = transforms.ToTensor()(target_img)

        # w_offset = np.random.randint(0, max(0, 286 - 256 - 1))
        # h_offset = np.random.randint(0, max(0, 286 - 256 - 1))
        #
        # a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        # b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        source_img = transforms.Normalize([0.5], [0.5])(source_img)
        target_img = transforms.Normalize([0.5], [0.5])(target_img)

        # for test images: no flip.
        # for random left-right-flip operation
        # if np.random.random() < 0.5:
        #     idx = [i for i in range(source_img.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     source_img = source_img.index_select(2, idx)
        #     target_img = target_img.index_select(2, idx)
        #
        return source_img, target_img


    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolder_in_test_mode(data.Dataset):
    '''
    The class deals with loading images from the dataset loader
    only return the test image itself for test mode.
    '''

    def __init__(self, image_dir):
        super(DatasetFromFolder_in_test_mode, self).__init__()
        self.img_path = image_dir
        self.image_filenames = [x for x in listdir(self.img_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        test_img = Image.open(join(self.img_path, self.image_filenames[index]))
        test_img = transforms.ToTensor()(test_img)

        test_img = transforms.Normalize([0.5], [0.5])(test_img)
        return test_img


    def __len__(self):
        return len(self.image_filenames)