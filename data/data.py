from os.path import join

from .dataset import DatasetFromFolder, DatasetFromFolder_simplified, DatasetFromFolder_in_test_mode
from torch.utils.data import DataLoader

def get_training_set(root_dir, direction):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir, direction,'train')


def get_test_set(root_dir, direction):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir, direction,'test')

def get_training_set_simplified(root_dir):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder_simplified(train_dir)


def get_test_set_simplified(root_dir):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder_simplified(test_dir)

def get_test_images_in_test_mode(opt):
    '''
    Here only give the path to the test imgs. No need to maunally pair the image
    into dataset format.
    :param opt:
    :return: test_data_loader.
    '''
    test_set = DatasetFromFolder_in_test_mode(opt.test_img_folder)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads,
                                     batch_size=opt.batch_size, shuffle=False)
    print('Validation images: {}'.format(len(test_set)))
    return testing_data_loader

def get_dataset_loader(opt):

    if opt.input_nc == 1:
        print('GRAY imgs')
        train_set = get_training_set_simplified(opt.dataset_path)
        test_set = get_test_set_simplified(opt.dataset_path)
    elif opt.input_nc == 3:
        print('RGB imgs')
        train_set = get_training_set(opt.dataset_path, 'source2target')
        test_set = get_test_set(opt.dataset_path, 'source2target')

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                      batch_size=opt.batch_size, shuffle=False)

    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads,
                                     batch_size=opt.batch_size, shuffle=False)

    print('Training images: {}'.format(len(train_set)))
    print('Validation images: {}'.format(len(test_set)))

    return training_data_loader,testing_data_loader