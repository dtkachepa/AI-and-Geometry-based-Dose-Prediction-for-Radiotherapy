import numpy as np
from torch.nn.functional import interpolate
import pdb

from utils.e_metrics import *

from utils.loss import *

def minmax_normalize(img_npy):
    '''
    img_npy: ndarray
    '''
    min_value = np.min(img_npy)
    max_value = np.max(img_npy)
    return (img_npy - min_value)/(max_value - min_value)

def calc_param_size(model):
    '''
    Show the memory cost of model.parameters, in MB.
    '''
    return np.sum(np.fromiter((np.prod(v.size()) for v in model.parameters()), dtype=np.int64)) * 4e-6


def dim_assert(t_list):
    '''
    To make sure that all the tensors in t_list has the same dims.
    '''
    dims = tuple(np.max([t.size() for t in t_list], axis=0)[-3:])
    for i in range(len(t_list)):
        if tuple(t_list[i].shape[-3:]) != dims:
            print_red('inconsistent dim: i')
            t_list[i] = interpolate(t_list[i], dims)
    return t_list


def print_red(something):
    print("\033[1;31m{}\033[0m".format(something))

# -*- encoding: utf-8 -*-
import torch.nn as nn


    
# -*- encoding: utf-8 -*-
import numpy as np
import random
import cv2
import torch


# Random flip
def random_flip_3d(list_images, list_axis=(0, 1, 2), p=0.5):
    if random.random() <= p:
        if 0 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, ::-1, :, :]
        if 1 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, ::-1, :]
        if 2 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, :, ::-1]

    return list_images


# Random rotation using OpenCV
def random_rotate_around_z_axis(list_images,
                                list_angles,
                                list_interp,
                                list_boder_value,
                                p=0.5):
    if random.random() <= p:
        # Randomly pick an angle list_angles
        _angle = random.sample(list_angles, 1)[0]
        # Do not use random scaling, set scale factor to 1
        _scale = 1.

        for image_i in range(len(list_images)):
            for chan_i in range(list_images[image_i].shape[0]):
                for slice_i in range(list_images[image_i].shape[1]):
                    rows, cols = list_images[image_i][chan_i, slice_i, :, :].shape
                    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), _angle, scale=_scale)
                    list_images[image_i][chan_i, slice_i, :, :] = \
                        cv2.warpAffine(list_images[image_i][chan_i, slice_i, :, :],
                                       M,
                                       (cols, rows),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=list_boder_value[image_i],
                                       flags=list_interp[image_i])
    return list_images


# Random translation
def random_translate(list_images, roi_mask, p, max_shift, list_pad_value):
    if random.random() <= p:
        exist_mask = np.where(roi_mask > 0)
        if exist_mask[0].size == 0:
            return list_images
        ori_z, ori_h, ori_w = list_images[0].shape[1:]

        bz = min(max_shift - 1, np.min(exist_mask[0]))
        ez = max(ori_z - 1 - max_shift, np.max(exist_mask[0]))
        bh = min(max_shift - 1, np.min(exist_mask[1]))
        eh = max(ori_h - 1 - max_shift, np.max(exist_mask[1]))
        bw = min(max_shift - 1, np.min(exist_mask[2]))
        ew = max(ori_w - 1 - max_shift, np.max(exist_mask[2]))

        for image_i in range(len(list_images)):
            list_images[image_i] = list_images[image_i][:, bz:ez + 1, bh:eh + 1, bw:ew + 1]

        # Pad to original size
        list_images = random_pad_to_size_3d(list_images,
                                            target_size=[ori_z, ori_h, ori_w],
                                            list_pad_value=list_pad_value)
    return list_images


# To tensor, images should be C*Z*H*W
def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images


# Pad
def random_pad_to_size_3d(list_images, target_size, list_pad_value):
    _, ori_z, ori_h, ori_w = list_images[0].shape[:]
    new_z, new_h, new_w = target_size[:]

    pad_z = new_z - ori_z
    pad_h = new_h - ori_h
    pad_w = new_w - ori_w

    pad_z_1 = random.randint(0, pad_z)
    pad_h_1 = random.randint(0, pad_h)
    pad_w_1 = random.randint(0, pad_w)

    pad_z_2 = pad_z - pad_z_1
    pad_h_2 = pad_h - pad_h_1
    pad_w_2 = pad_w - pad_w_1

    output = []
    for image_i in range(len(list_images)):
        _image = list_images[image_i]
        output.append(np.pad(_image,
                             ((0, 0), (pad_z_1, pad_z_2), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                             mode='constant',
                             constant_values=list_pad_value[image_i])
                      )
    return output

# -*- encoding: utf-8 -*-
import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2



"""
images are always C*Z*H*W
"""


def read_data(patient_dir):
    dict_images = {}
    list_structures = ['CT',
                       'PTV70',
                       'PTV63',
                       'PTV56',
                       'possible_dose_mask',
                       'Brainstem',
                       'SpinalCord',
                       'RightParotid',
                       'LeftParotid',
                       'Esophagus',
                       'Larynx',
                       'Mandible',
                       'dose']

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '.nii.gz'

        if structure_name == 'CT':
            dtype = sitk.sitkInt16
        elif structure_name == 'dose':
            dtype = sitk.sitkFloat32
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            # To numpy array (C * Z * H * W)
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    return dict_images


def pre_processing(dict_images):
    # PTVs
    PTVs = 70.0 / 70. * dict_images['PTV70'] \
           + 63.0 / 70. * dict_images['PTV63'] \
           + 56.0 / 70. * dict_images['PTV56']

    # OARs
    list_OAR_names = ['Brainstem',
                      'SpinalCord',
                      'RightParotid',
                      'LeftParotid',
                      'Esophagus',
                      'Larynx',
                      'Mandible']
    OAR_all = np.concatenate([dict_images[OAR_name] for OAR_name in list_OAR_names], axis=0)

    # CT image
    CT = dict_images['CT']
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.

    # Dose image
    dose = dict_images['dose'] / 70.

    # Possible_dose_mask, the region that can receive dose
    possible_dose_mask = dict_images['possible_dose_mask']

    list_images = [np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
                   dose,  # Label
                   possible_dose_mask
                   ]
    return list_images


def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angles=(0, 40, 80, 120, 160, 200, 240, 280, 320),
                                              list_boder_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(list_images,
                                   roi_mask=list_images[2][0, :, :, :],  # the possible dose mask
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0])

    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


class MyDataset(data.Dataset):
    def __init__(self, num_samples_per_epoch, phase):
        # 'train' or 'val
        self.phase = phase
        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {'train': train_transform, 'val': val_transform}

        self.list_case_id = {'train': ['OpenKBP_C3D/pt_' + str(i) for i in range(201)],
                              'val': ['OpenKBP_C3D/pt_' + str(i) for i in range(201,241)]}[phase]
                             

        random.shuffle(self.list_case_id)
        self.sum_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            case_id = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            case_id = self.list_case_id[new_index_]

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images)

        list_images = self.transform[self.phase](list_images)
        return list_images

    def __len__(self):
        return self.num_samples_per_epoch


def get_loader(train_bs=1, val_bs=1, train_num_samples_per_epoch=1, val_num_samples_per_epoch=1, num_works=2):

    train_dataset = MyDataset(num_samples_per_epoch=train_num_samples_per_epoch, phase='train')
    val_dataset = MyDataset(num_samples_per_epoch=val_num_samples_per_epoch, phase='val')

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works,
                                   pin_memory=True, prefetch_factor=2)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=True, prefetch_factor=2)

    return train_loader, val_loader

# -*- encoding: utf-8 -*-
import time

import torch
import torch.nn as nn
from torch import optim
import wandb


class TrainerSetting:
    def __init__(self):
        self.project_name = None
        # Path for saving model and training log
        self.output_dir = None

        # Optional: Google Drive directory to persist checkpoints across Colab sessions.
        # Set this to a mounted Drive path, e.g. '/content/drive/MyDrive/RANDose_checkpoints'
        self.checkpoint_gdrive_dir = None

        # Generally only use one of them
        self.max_iter = 99999999
        self.max_epoch = 99999999

        # Default not use this,
        # because the models of "best_train_loss", "best_val_evaluation_index", "latest" have been saved.
        self.save_per_epoch = 99999999
        self.eps_train_loss = 0.01

        self.network = None
        self.device = None
        self.list_GPU_ids = None

        self.train_loader = None
        self.val_loader = None

        self.optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_type = None

        # Default update learning rate after each epoch
        self.lr_scheduler_update_on_iter = False

        self.loss_function = None

        # If do online evaluation during validation
        self.online_evaluation_function_val = None


class TrainerLog:
    def __init__(self):
        self.iter = -1
        self.epoch = -1

        # Moving average loss, loss is the smaller the better
        self.moving_train_loss = None
        # Average train loss of a epoch
        self.average_train_loss = 99999999.
        self.best_average_train_loss = 99999999.
        # Evaluation index is the higher the better
        self.average_val_index = -99999999.
        self.best_average_val_index = -99999999.

        # Record changes in training loss
        self.list_average_train_loss_associate_iter = []
        # Record changes in validation evaluation index
        self.list_average_val_index_associate_iter = []
        # Record changes in learning rate
        self.list_lr_associate_iter = []

        # Save status of the trainer, eg. best_train_loss, latest, best_val_evaluation_index
        self.save_status = []


class TrainerTime:
    def __init__(self):
        self.train_time_per_epoch = 0.
        # Time for loading data, eg. data precessing, data augmentation and moving tensors from cpu to gpu
        # In fact, most of the time is spent on moving tensors from cpu to gpu, something like doing multi-processing on
        # CUDA tensors cannot succeed in Windows,
        # you may use cuda.Steam to accelerate it. https://github.com/NVIDIA/apex, but it needs larger GPU memory
        self.train_loader_time_per_epoch = 0.

        self.val_time_per_epoch = 0.
        self.val_loader_time_per_epoch = 0.


class NetworkTrainer:
    def __init__(self):
        self.log = TrainerLog()
        self.setting = TrainerSetting()
        self.time = TrainerTime()
        wandb.init(project="VNet")
        wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

    def set_GPU_device(self, list_GPU_ids):
        self.setting.list_GPU_ids = list_GPU_ids
        sum_device = len(list_GPU_ids)
        # cpu only
        if list_GPU_ids[0] == -1:
            self.setting.device = torch.device('cpu')
        # single GPU
        elif sum_device == 1:
            self.setting.device = torch.device('cuda:' + str(list_GPU_ids[0]))
        # multi-GPU
        else:
            self.setting.device = torch.device('cuda:' + str(list_GPU_ids[0]))
            self.setting.network = nn.DataParallel(self.setting.network, device_ids=list_GPU_ids)
        self.setting.network.to(self.setting.device)

    def set_optimizer(self, optimizer_type, args):
        # Sometimes we need set different learning rates for "encoder" and "decoder" separately
        if optimizer_type == 'Adam':
            if hasattr(self.setting.network, 'decoder') and hasattr(self.setting.network, 'encoder'):
                self.setting.optimizer = optim.Adam([
                    {'params': self.setting.network.encoder.parameters(), 'lr': args['lr_encoder']},
                    {'params': self.setting.network.decoder.parameters(), 'lr': args['lr_decoder']}
                ],
                    weight_decay=args['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    amsgrad=True)
            else:
                self.setting.optimizer = optim.Adam(self.setting.network.parameters(),
                                                    lr=args['lr'],
                                                    weight_decay=3e-5,
                                                    betas=(0.9, 0.999),
                                                    eps=1e-08,
                                                    amsgrad=True)

    def set_lr_scheduler(self, lr_scheduler_type, args):
        if lr_scheduler_type == 'step':
            self.setting.lr_scheduler_type = 'step'
            self.setting.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.setting.optimizer,
                                                                       milestones=args['milestones'],
                                                                       gamma=args['gamma'],
                                                                       last_epoch=args['last_epoch']
                                                                       )
        elif lr_scheduler_type == 'cosine':
            self.setting.lr_scheduler_type = 'cosine'
            self.setting.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.setting.optimizer,
                                                                             T_max=args['T_max'],
                                                                             eta_min=args['eta_min'],
                                                                             last_epoch=args['last_epoch']
                                                                             )
        elif lr_scheduler_type == 'ReduceLROnPlateau':
            self.setting.lr_scheduler_type = 'ReduceLROnPlateau'
            self.setting.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.setting.optimizer,
                                                                             mode='min',
                                                                             factor=args['factor'],
                                                                             patience=args['patience'],
                                                                             verbose=True,
                                                                             threshold=args['threshold'],
                                                                             threshold_mode='rel',
                                                                             cooldown=0,
                                                                             min_lr=0,
                                                                             eps=1e-08)

    def update_lr(self):
        # Update learning rate, only 'ReduceLROnPlateau' need use the moving train loss
        if self.setting.lr_scheduler_type == 'ReduceLROnPlateau':
            self.setting.lr_scheduler.step(self.log.moving_train_loss)
        else:
            self.setting.lr_scheduler.step()

    def update_moving_train_loss(self, loss):
        if self.log.moving_train_loss is None:
            self.log.moving_train_loss = loss.item()
        else:
            self.log.moving_train_loss = \
                (1 - self.setting.eps_train_loss) * self.log.moving_train_loss \
                + self.setting.eps_train_loss * loss.item()

    def update_average_statistics(self, loss, phase='train'):
        if phase == 'train':
            self.log.average_train_loss = loss
            if loss < self.log.best_average_train_loss:
                self.log.best_average_train_loss = loss
                self.log.save_status.append('best_train_loss')
            self.log.list_average_train_loss_associate_iter.append([self.log.average_train_loss, self.log.iter])

        elif phase == 'val':
            self.log.average_val_index = loss
            if loss > self.log.best_average_val_index:
                self.log.best_average_val_index = loss
                self.log.save_status.append('best_val_evaluation_index')
            self.log.list_average_val_index_associate_iter.append([self.log.average_val_index, self.log.iter])

    def forward(self, input_, phase):
        time_start_load_data = time.time()
        # To device
        input_ = input_.to(self.setting.device)

        # Record time of moving input from cpu to gpu
        self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

        # Forward
        if phase == 'train':
            self.setting.optimizer.zero_grad()
        output = self.setting.network(input_)

        return output

    def backward(self, output, target, PTV, OAR):
        time_start_load_data = time.time()
        for target_i in range(len(target)):
            target[target_i] = target[target_i].to(self.setting.device)

        # Record time of moving target from cpu to gpu
        self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

        # Optimize
        loss = self.setting.loss_function(output, target, PTV, OAR)
        loss.backward()
        self.setting.optimizer.step()

        return loss

    def train(self):
        time_start_train = time.time()

        self.setting.network.train()
        sum_train_loss = 0.
        count_iter = 0

        time_start_load_data = time.time()
        for batch_idx, list_loader_output in enumerate(self.setting.train_loader):

            if (self.setting.max_iter is not None) and (self.log.iter >= self.setting.max_iter - 1):
                break
            self.log.iter += 1

            # List_loader_output[0] default as the input
            input_ = list_loader_output[0]
            target = list_loader_output[1:]

            
            
            PTV = input_[:, 0:1, :, :, :]#input_[:1, :, :, :]  # PTV (1 channel)  ###
            OAR = input_[:, 1:8, :, :, :]                    ###

            # Record time of preparing data
            self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

            # Forward
            output = self.forward(input_, phase='train')

            # Backward
            loss = self.backward(output, target, PTV, OAR)

            # Used for counting average loss of this epoch
            sum_train_loss += loss.item()
            count_iter += 1

            self.update_moving_train_loss(loss)
            self.update_lr()

            # Print loss during the first epoch
            if self.log.epoch == 0:
                if self.log.iter % 10 == 0:
                    self.print_log_to_file('                Iter %12d       %12.5f\n' %
                                           (self.log.iter, self.log.moving_train_loss), 'a')

            time_start_load_data = time.time()

        if count_iter > 0:
            average_loss = sum_train_loss / count_iter
            self.update_average_statistics(average_loss, phase='train')

        self.time.train_time_per_epoch = time.time() - time_start_train

    def val(self):
        time_start_val = time.time()
        self.setting.network.eval()

        if self.setting.online_evaluation_function_val is None:
            self.print_log_to_file('===============================> No online evaluation method specified ! \n', 'a')
            raise Exception('No online evaluation method specified !')
        else:
            val_index = self.setting.online_evaluation_function_val(self)
            self.update_average_statistics(val_index, phase='val')

        self.time.val_time_per_epoch = time.time() - time_start_val

    def run(self):
        if self.log.iter == -1:
            self.print_log_to_file('Start training !\n', 'w')
        else:
            self.print_log_to_file('Continue training !\n', 'w')
        self.print_log_to_file(time.strftime('Local time: %H:%M:%S\n', time.localtime(time.time())), 'a')

        # Start training
        while (self.log.epoch < self.setting.max_epoch - 1) and (self.log.iter < self.setting.max_iter - 1):
            #
            time_start_this_epoch = time.time()
            self.log.epoch += 1
            # Print current learning rate
            self.print_log_to_file('Epoch: %d, iter: %d\n' % (self.log.epoch, self.log.iter), 'a')
            self.print_log_to_file('    Begin lr is %12.12f, %12.12f\n' % (
                self.setting.optimizer.param_groups[0]['lr'], self.setting.optimizer.param_groups[-1]['lr']), 'a')

            # Record initial learning rate for this epoch
            self.log.list_lr_associate_iter.append([self.setting.optimizer.param_groups[0]['lr'], self.log.iter])

            self.time.__init__()
            self.train()
            self.val()
            wandb.log({'train_loss': self.log.average_train_loss})
            wandb.log({'val dose error': self.log.average_val_index})

            # If update learning rate per epoch
            if not self.setting.lr_scheduler_update_on_iter:
                self.update_lr()

            # Save trainer every "self.setting.save_per_epoch"
            if (self.log.epoch + 1) % self.setting.save_per_epoch == 0:
                self.log.save_status.append('iter_' + str(self.log.iter))
            self.log.save_status.append('latest')

            # Try save trainer
            if len(self.log.save_status) > 0:
                for status in self.log.save_status:
                    self.save_trainer(status=status)
                self.log.save_status = []

            self.print_log_to_file(
                '            Average train loss is             %12.12f,     best is           %12.12f\n' %
                (self.log.average_train_loss, self.log.best_average_train_loss), 'a')
            self.print_log_to_file(
                '            Average val evaluation index is   %12.12f,     best is           %12.12f\n'
                % (self.log.average_val_index, self.log.best_average_val_index), 'a')

            self.print_log_to_file('    Train use time %12.5f\n' % (self.time.train_time_per_epoch), 'a')
            self.print_log_to_file('    Train loader use time %12.5f\n' % (self.time.train_loader_time_per_epoch), 'a')
            self.print_log_to_file('    Val use time %12.5f\n' % (self.time.val_time_per_epoch), 'a')
            self.print_log_to_file('    Total use time %12.5f\n' % (time.time() - time_start_this_epoch), 'a')
            self.print_log_to_file('    End lr is %12.12f, %12.12f\n' % (
                self.setting.optimizer.param_groups[0]['lr'], self.setting.optimizer.param_groups[-1]['lr']), 'a')
            self.print_log_to_file(time.strftime('    time: %H:%M:%S\n', time.localtime(time.time())), 'a')

        self.print_log_to_file('===============================> End successfully\n', 'a')
        self.save_wandb()

    def print_log_to_file(self, txt, mode):
        with open(self.setting.output_dir + '/log.txt', mode) as log_:
            log_.write(txt)

        # Also display log in the terminal
        txt = txt.replace('\n', '')
        print(txt)

    def save_trainer(self, status='latest'):
        if len(self.setting.list_GPU_ids) > 1:
            network_state_dict = self.setting.network.module.state_dict()
        else:
            network_state_dict = self.setting.network.state_dict()

        optimizer_state_dict = self.setting.optimizer.state_dict()
        lr_scheduler_state_dict = self.setting.lr_scheduler.state_dict()

        ckpt = {
            'network_state_dict': network_state_dict,
            'lr_scheduler_state_dict': lr_scheduler_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'log': self.log
        }

        local_path = self.setting.output_dir + '/' + status + '.pkl'
        torch.save(ckpt, local_path)
        self.print_log_to_file('        ==> Saving ' + status + ' model successfully !\n', 'a')

        # Mirror checkpoint to Google Drive for persistent storage across Colab sessions
        if self.setting.checkpoint_gdrive_dir is not None:
            import shutil
            gdrive_path = os.path.join(self.setting.checkpoint_gdrive_dir, status + '.pkl')
            shutil.copy2(local_path, gdrive_path)
            self.print_log_to_file('        ==> Mirrored ' + status + ' to Drive: ' + gdrive_path + '\n', 'a')

    # Default load trainer in cpu, please reset device using the function self.set_GPU_device
    def init_trainer(self, ckpt_file, list_GPU_ids, only_network=True):
        ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)

        self.setting.network.load_state_dict(ckpt['network_state_dict'])

        if not only_network:
            self.setting.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            self.setting.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.log = ckpt['log']

        self.set_GPU_device(list_GPU_ids)

        # If do not do so, the states of optimizer will always in cpu
        # This for Adam
        if type(self.setting.optimizer).__name__ == 'Adam':
            for key in self.setting.optimizer.state.items():
                key[1]['exp_avg'] = key[1]['exp_avg'].to(self.setting.device)
                key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(self.setting.device)
                key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(self.setting.device)

        self.print_log_to_file('==> Init trainer from ' + ckpt_file + ' successfully! \n', 'a')


    def save_wandb(self):
        config = wandb.config
        config.project_name = self.setting.project_name
        config.output_dir = self.setting.output_dir
        config.max_iter = self.setting.max_iter
        config.max_epoch = self.setting.max_epoch
        config.save_per_epoch = self.setting.save_per_epoch


# -*- encoding: utf-8 -*-



def online_evaluation(trainer):
    list_patient_dirs = ['OpenKBP_C3D/pt_' + str(i) for i in range(201, 241)]

    list_Dose_score = []

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in list_patient_dirs:
            patient_name = patient_dir.split('/')[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            gt_dose = list_images[1]
            possible_dose_mask = list_images[2]

            # Forward
            [input_] = val_transform([input_])
            input_ = input_.unsqueeze(0).to(trainer.setting.device)
            prediction_B = trainer.setting.network(input_)
            prediction_B = np.array(prediction_B.cpu().data[0, :, :, :, :])

            # Post processing and evaluation
            prediction_B[np.logical_or(possible_dose_mask < 1, prediction_B < 0)] = 0
            Dose_score = 70. * get_3D_Dose_dif(prediction_B.squeeze(0), gt_dose.squeeze(0),
                                               possible_dose_mask.squeeze(0))
            list_Dose_score.append(Dose_score)

            try:
                trainer.print_log_to_file('========> ' + patient_name + ':  ' + str(Dose_score), 'a')
            except:
                pass

    try:
        trainer.print_log_to_file('===============================================> mean Dose score: '
                                  + str(np.mean(list_Dose_score)), 'a')
    except:
        pass
    # Evaluation score is the higher the better
    return - np.mean(list_Dose_score)