"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import torch
import torchvision.transforms.functional as F
import sys
sys.path.append('/home/batman/Documents/cs282a/proj/test_model/fastMRI')
from fastmri import ifft2c, fft2c


class FastmriDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--no_crop', action='store_true', help='dont crop if specified')
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.root = opt.dataroot
        self.root2 = opt.dataroot2 #kspace directory
        # get the image paths of your dataset;
        self.image_paths = sorted(make_dataset(self.root))  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        self.image_paths2 = sorted(make_dataset(self.root2)) 
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt)

    def __getitem__(self, index):#, opt):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        A_path = self.image_paths[index]    # needs to be a string
        K_path = self.image_paths2[index]   # index here should be the same for both mag directory and kspace directory

        A_tensor = torch.load(A_path)
        K_tensor = torch.load(K_path) # should be same size as A tensor but with 2 channels (real/imag kspace) 

        data_A = 2 * ((A_tensor - A_tensor.min()) / (A_tensor.max() - A_tensor.min())) -1   # needs to be a tensor
        #data_A = A_tensor
        converted_kspace = ifft2c(K_tensor)    # needs to be a tensor
        data_B = 2 * ((converted_kspace - converted_kspace.min()) / (converted_kspace.max() - converted_kspace.min())) - 1
        
        #print('before:', data_A.shape)
        #print('before:',data_B.shape)
        #full_tensor = torch.load(path)
        #split_tensor = torch.split(full_tensor, 2, dim=0)
        #data_A = split_tensor[0]    # needs to be a tensor
        #data_B = split_tensor[1]    # needs to be a tensor
        sz = self.opt.crop_size
        if not self.opt.no_crop:
            #if data_A.dim() > 2:
            #    data_A = torch.movedim(data_A, (2, 2), (0, 1))
            #    data_B = torch.movedim(data_B, (2, 2), (0, 1))
            data_A = data_A.unsqueeze(0)
            data_B = data_B.permute(2, 0, 1)
            #print('lewis:', data_A.shape)
            #print('lewis:', data_B.shape))
            data_A = F.center_crop(data_A, [sz, sz])
            data_B = F.center_crop(data_B, [sz, sz])
            #print('after crop:',data_A.shape)
            #print('after crop:',data_B.shape)
        return {'A': data_A, 'B': data_B, 'A_paths': A_path, 'B_paths': K_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
