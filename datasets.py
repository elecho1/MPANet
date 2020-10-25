import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


from nima.train.utils import SCORE_NAMES


class AVADataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        return x, p.astype('float32')

class AVAPathImageDataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        # return x, p.astype('float32')
        return x, image_id, image_path, p.astype('float32')

class AVAMPAesDataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, crop_num: int, transform, np_seed=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        self.crop_num = crop_num
        self.np_seed = np_seed
        if self.np_seed:
            print("val dataset random crop seed:", self.np_seed)

    def _random_crop(self, x):
        # config
        cropped_size = 224

        # cropping
        height = x.shape[-2]
        width = x.shape[-1]
        x_list = np.random.randint(0, width-cropped_size, size=self.crop_num)
        y_list = np.random.randint(0, height-cropped_size, size=self.crop_num)
        crop_list = [x[:, y_idx:y_idx+cropped_size, x_idx:x_idx+cropped_size] for x_idx, y_idx in zip(x_list, y_list)]
        crop_tensor = torch.stack(crop_list)
        # (crop_tensor).shape = [crops, 3, 224, 224]
        return crop_tensor

    def _random_seed_crop(self, x, random_seed=None):
        # config
        cropped_size = 224

        # cropping
        height = x.shape[-2]
        width = x.shape[-1]
        np.random.seed(seed=random_seed)
        x_list = np.random.randint(0, width-cropped_size, size=self.crop_num)
        np.random.seed(seed=random_seed * 2)
        y_list = np.random.randint(0, height-cropped_size, size=self.crop_num)
        crop_list = [x[:, y_idx:y_idx+cropped_size, x_idx:x_idx+cropped_size] for x_idx, y_idx in zip(x_list, y_list)]
        crop_tensor = torch.stack(crop_list)
        # (crop_tensor).shape = [crops, 3, 299, 299]
        return crop_tensor

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        if self.np_seed:
            x_crops = self._random_seed_crop(x, random_seed = image_id + self.np_seed)
        else:
            x_crops = self._random_crop(x)

        return x_crops, p.astype('float32')

class AVAMPAesInceptionDataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, crop_num: int, transform, np_seed=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        self.crop_num = crop_num
        self.np_seed = np_seed
        if self.np_seed:
            print("val dataset random crop seed:", self.np_seed)

    def _random_crop(self, x):
        # config
        cropped_size = 299

        # cropping
        height = x.shape[-2]
        width = x.shape[-1]
        x_list = np.random.randint(0, width-cropped_size, size=self.crop_num)
        y_list = np.random.randint(0, height-cropped_size, size=self.crop_num)
        crop_list = [x[:, y_idx:y_idx+cropped_size, x_idx:x_idx+cropped_size] for x_idx, y_idx in zip(x_list, y_list)]
        crop_tensor = torch.stack(crop_list)
        # (crop_tensor).shape = [crops, 3, 299, 299]
        return crop_tensor

    def _random_seed_crop(self, x, random_seed=None):
        # config
        cropped_size = 299

        # cropping
        height = x.shape[-2]
        width = x.shape[-1]
        np.random.seed(seed=random_seed)
        x_list = np.random.randint(0, width-cropped_size, size=self.crop_num)
        np.random.seed(seed=random_seed * 2)
        y_list = np.random.randint(0, height-cropped_size, size=self.crop_num)
        crop_list = [x[:, y_idx:y_idx+cropped_size, x_idx:x_idx+cropped_size] for x_idx, y_idx in zip(x_list, y_list)]
        crop_tensor = torch.stack(crop_list)
        # (crop_tensor).shape = [crops, 3, 299, 299]
        return crop_tensor

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        if self.np_seed:
            x_crops = self._random_seed_crop(x, random_seed = image_id + self.np_seed)
        else:
            x_crops = self._random_crop(x)

        return x_crops, p.astype('float32')


class AVAMPAesLocalGlobalInceptionDataset(Dataset):
    """ Dataset class for making crops of 3*3+1 """
    def __init__(self, path_to_csv: str, images_path: str, side_crop_num: int, transform, transform_global):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform_local = transform
        self.transform_global = transform_global
        self.side_crop_num = side_crop_num


    def _crop_local(self, x, side_crop_num:int, ignore_duplicate=False):
        """ [EN] extrach 3*3 patches.  [JP] 元画像から3*3のパッチを抽出．"""
        # config
        cropped_size = 299
        div_num = side_crop_num * 2

        # culculating cropping area
        height = x.shape[-2]
        width = x.shape[-1]

        plain_center_x_list = [width / div_num * i for i in range(1, div_num, 2)]
        plain_center_y_list = [height / div_num * i for i in range(1, div_num, 2)]
        plain_basis_x_list = [round(coord - cropped_size / 2) for coord in plain_center_x_list]
        plain_basis_y_list = [round(coord - cropped_size / 2) for coord in plain_center_y_list]

        # rounding process
        basis_x_list = plain_basis_x_list[:]
        basis_y_list = plain_basis_y_list[:]
        if basis_x_list[0] < 0:
            basis_x_list[0] = 0
        if basis_x_list[-1] > width - 1 - cropped_size:
            basis_x_list[-1] = width - 1 - cropped_size
        if basis_y_list[0] < 0:
            basis_y_list[0] = 0
        if basis_y_list[-1] > height - 1 - cropped_size:
            basis_y_list[-1] = height - 1 - cropped_size

        if not ignore_duplicate:
            assert len(basis_x_list) == len(set(basis_x_list)), "There are duplicated x coordinates. (set `ignore_duplicate = True` if you want to ignore duplication.)"
            assert len(basis_x_list) == len(set(basis_x_list)), "There are duplicated y coordinates. (set `ignore_duplicate = True` if you want to ignore duplication.)"

        # cropping
        basis_coords = [(x_idx, y_idx) for x_idx in basis_x_list for y_idx in basis_y_list]
        crop_list = [x[:, y_idx:y_idx + cropped_size, x_idx:x_idx + cropped_size] for x_idx, y_idx in basis_coords]
        # print("shape", x.shape)
        # print("basis_coords", basis_coords)
        # print("crop_num", len(crop_list))
        crop_tensor = torch.stack(crop_list)
        # (crop_tensor).shape = [crops, 3, 299, 299]
        return crop_tensor

    def _crop_global(self, x):
        """ [EN] crop the center of the original image.   [JP]元画像から中心を抽出．"""
        # config
        cropped_size=299

        # culculating cropping area
        height = x.shape[-2]
        width = x.shape[-1]
        center_point = (round(width/2 - cropped_size/2), round(height/2 - cropped_size/2))
        # print((width / 2 - cropped_size / 2, height / 2 - cropped_size / 2))
        # print(center_point)
        # print(x.shape)

        # rounding_process
        if center_point[0] < 0 or center_point[1] < 0:
            temp_center_x = max(center_point[0], 0)
            temp_center_y = max(center_point[1], 0)
            center_point = (temp_center_x, temp_center_y)

        # print(center_point[0] + cropped_size, width)
        # print(center_point[1] + cropped_size, height)
        assert center_point[0] + cropped_size <= width, "x width error."
        assert center_point[1] + cropped_size <= height, "y width error."

        # cropping
        cropped_image = x[:, center_point[1]:center_point[1] + cropped_size, center_point[0]:center_point[0] + cropped_size]
        crop_tensor = torch.stack([cropped_image])
        # (crop_tensor).shape = [1, 3, 299, 299]
        return crop_tensor

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x_local = self.transform_local(image)
        x_global = self.transform_global(image)
        # if self.np_seed:
        #     x_crops = self._random_seed_crop(x, random_seed = image_id + self.np_seed)
        # else:
        #     x_crops = self._random_crop(x)

        x_local_crops = self._crop_local(x_local, side_crop_num = self.side_crop_num)
        x_global_crops = self._crop_global(x_global)
        x_crops = torch.cat((x_local_crops, x_global_crops), dim=0)

        return x_crops, p.astype('float32')


class AVAMPAesLocalInceptionDataset(Dataset):
    """ Dataset class for making crops of 3*3+1 """
    def __init__(self, path_to_csv: str, images_path: str, side_crop_num: int, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform_local = transform
        # self.transform_global = transform_global
        self.side_crop_num = side_crop_num
        
    

    def _crop_local(self, x, side_crop_num:int, ignore_duplicate=False):
        """ [EN] extrach 3*3 patches.   [JP] 元画像から3*3のパッチを抽出．"""
        # config
        cropped_size = 299
        div_num = side_crop_num * 2

        # culculating cropping area
        height = x.shape[-2]
        width = x.shape[-1]

        plain_center_x_list = [width / div_num * i for i in range(1, div_num, 2)]
        plain_center_y_list = [height / div_num * i for i in range(1, div_num, 2)]
        plain_basis_x_list = [round(coord - cropped_size / 2) for coord in plain_center_x_list]
        plain_basis_y_list = [round(coord - cropped_size / 2) for coord in plain_center_y_list]

        # rounding process
        basis_x_list = plain_basis_x_list[:]
        basis_y_list = plain_basis_y_list[:]
        if basis_x_list[0] < 0:
            basis_x_list[0] = 0
        if basis_x_list[-1] > width - 1 - cropped_size:
            basis_x_list[-1] = width - 1 - cropped_size
        if basis_y_list[0] < 0:
            basis_y_list[0] = 0
        if basis_y_list[-1] > height - 1 - cropped_size:
            basis_y_list[-1] = height - 1 - cropped_size

        if not ignore_duplicate:
            assert len(basis_x_list) == len(set(basis_x_list)), "There are duplicated x coordinates. (set `ignore_duplicate = True` if you want to ignore duplication.)"
            assert len(basis_x_list) == len(set(basis_x_list)), "There are duplicated y coordinates. (set `ignore_duplicate = True` if you want to ignore duplication.)"

        # cropping
        basis_coords = [(x_idx, y_idx) for x_idx in basis_x_list for y_idx in basis_y_list]
        crop_list = [x[:, y_idx:y_idx + cropped_size, x_idx:x_idx + cropped_size] for x_idx, y_idx in basis_coords]
        # print("shape", x.shape)
        # print("basis_coords", basis_coords)
        # print("crop_num", len(crop_list))
        crop_tensor = torch.stack(crop_list)
        # (crop_tensor).shape = [crops, 3, 299, 299]
        return crop_tensor

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x_local = self.transform_local(image)
        # if self.np_seed:
        #     x_crops = self._random_seed_crop(x, random_seed = image_id + self.np_seed)
        # else:
        #     x_crops = self._random_crop(x)

        x_local_crops = self._crop_local(x_local, side_crop_num = self.side_crop_num)
        x_crops = x_local_crops

        return x_crops, p.astype('float32')


""" coord conv common methods. """
def _add_abs_coord(x):
    height = x.shape[-2]
    width = x.shape[-1]

    yy_ones = torch.ones(height, dtype=torch.float)
    yy_cum = yy_ones.cumsum(dim=0)
    yy_shift = yy_cum - ((height + 1) / 2)
    yy_usq = yy_shift.unsqueeze(1)
    yy_tensor = yy_usq.repeat(1, width)
    # print("yy_tensor:", yy_tensor.size())

    xx_ones = torch.ones(width, dtype=torch.float)
    xx_cum = xx_ones.cumsum(dim=0)
    xx_shift = xx_cum - ((width + 1) / 2)
    xx_tensor = xx_shift.repeat(height, 1)
    # print("xx_tensor:", xx_tensor.size())

    x_tensor = torch.cat([x, yy_tensor.unsqueeze(0), xx_tensor.unsqueeze(0)])
    # print("x_tensor:", x_tensor.size())

    return x_tensor


def _add_abs_pseudo_norm_coord(x):
    height = x.shape[-2]
    width = x.shape[-1]
    crit_length = (min(height, width) - 1) / 2 # the shorter edge of dataset transform.

    yy_ones = torch.ones(height, dtype=torch.float)
    yy_cum = yy_ones.cumsum(dim=0)
    yy_shift = yy_cum - ((height + 1) / 2)
    yy_shift = yy_shift / crit_length
    yy_usq = yy_shift.unsqueeze(1)
    yy_tensor = yy_usq.repeat(1, width)
    # print("yy_tensor:", yy_tensor.size())

    xx_ones = torch.ones(width, dtype=torch.float)
    xx_cum = xx_ones.cumsum(dim=0)
    xx_shift = xx_cum - ((width + 1) / 2)
    xx_shift = xx_shift / crit_length
    xx_tensor = xx_shift.repeat(height, 1)
    # print("xx_tensor:", xx_tensor.size())

    x_tensor = torch.cat([x, yy_tensor.unsqueeze(0), xx_tensor.unsqueeze(0)])
    # print("x_tensor:", x_tensor.size())

    return x_tensor


def _add_rel_coord(x):
    height = x.shape[-2]
    width = x.shape[-1]

    yy_ones = torch.ones(height, dtype = torch.float)
    yy_cum = yy_ones.cumsum(dim=0)
    yy_shift = yy_cum - ((height+1) / 2)
    yy_max = torch.max(yy_shift[0].abs(), yy_shift[-1].abs())
    yy_shift = yy_shift / yy_max
    yy_usq = yy_shift.unsqueeze(1)
    yy_tensor = yy_usq.repeat(1, width)
    # print("yy_tensor:", yy_tensor.size())

    xx_ones = torch.ones(width, dtype=torch.float)
    xx_cum = xx_ones.cumsum(dim=0)
    xx_shift = xx_cum - ((width + 1) / 2)
    xx_max = torch.max(xx_shift[0].abs(), xx_shift[-1].abs())
    xx_shift = xx_shift / xx_max
    xx_tensor = xx_shift.repeat(height, 1)
    # print("xx_tensor:", xx_tensor.size())

    x_tensor = torch.cat([x, yy_tensor.unsqueeze(0), xx_tensor.unsqueeze(0)])
    # print("x_tensor:", x_tensor.size())

    return x_tensor


def _random_single_crop(x, cropped_size):
    # config
    # cropped_size = 299
    # cropped_size = self.cropped_size

    # cropping
    height = x.shape[-2]
    width = x.shape[-1]
    x_idx = np.random.randint(0, width - cropped_size, size=None)
    y_idx = np.random.randint(0, height-cropped_size, size=None)
    crop = x[:, y_idx:y_idx+cropped_size, x_idx:x_idx+cropped_size]
    crop_tensor = crop
    # (crop_tensor).shape = [5, 299, 299]
    return crop_tensor


def _random_seed_single_crop(x, cropped_size, random_seed=None):
    # config
    # cropped_size = 299
    # cropped_size = self.cropped_size

    # cropping
    height = x.shape[-2]
    width = x.shape[-1]
    np.random.seed(seed=random_seed)
    x_idx = np.random.randint(0, width - cropped_size, size=None)
    np.random.seed(seed=random_seed * 2)
    y_idx = np.random.randint(0, height - cropped_size, size=None)
    crop = x[:, y_idx:y_idx + cropped_size, x_idx:x_idx + cropped_size]
    crop_tensor = crop
    return crop_tensor

def _random_crop(x, cropped_size, crop_num):
    # config
    # cropped_size = 299
    # cropped_size = self.cropped_size

    # cropping
    height = x.shape[-2]
    width = x.shape[-1]
    x_list = np.random.randint(0, width-cropped_size, size=crop_num)
    y_list = np.random.randint(0, height-cropped_size, size=crop_num)
    crop_list = [x[:, y_idx:y_idx+cropped_size, x_idx:x_idx+cropped_size] for x_idx, y_idx in zip(x_list, y_list)]
    crop_tensor = torch.stack(crop_list)
    # (crop_tensor).shape = [crops, 5, 299, 299]
    return crop_tensor

def _random_seed_crop(x, cropped_size, crop_num, random_seed=None):
    # config
    # cropped_size = 299
    # cropped_size = self.cropped_size

    # cropping
    height = x.shape[-2]
    width = x.shape[-1]
    np.random.seed(seed=random_seed)
    x_list = np.random.randint(0, width-cropped_size, size=crop_num)
    np.random.seed(seed=random_seed * 2)
    y_list = np.random.randint(0, height-cropped_size, size=crop_num)
    crop_list = [x[:, y_idx:y_idx+cropped_size, x_idx:x_idx+cropped_size] for x_idx, y_idx in zip(x_list, y_list)]
    crop_tensor = torch.stack(crop_list)
    # (crop_tensor).shape = [crops, 5, 299, 299]
    return crop_tensor


def _crop_local(x, cropped_size, side_crop_num:int, ignore_duplicate=False):
    """ [EN] extract 3*3 patches.  [JP] 元画像から3*3のパッチを抽出．"""
    # config
    # cropped_size = self.cropped_size
    div_num = side_crop_num * 2

    # culculating cropping area
    height = x.shape[-2]
    width = x.shape[-1]

    plain_center_x_list = [width / div_num * i for i in range(1, div_num, 2)]
    plain_center_y_list = [height / div_num * i for i in range(1, div_num, 2)]
    plain_basis_x_list = [round(coord - cropped_size / 2) for coord in plain_center_x_list]
    plain_basis_y_list = [round(coord - cropped_size / 2) for coord in plain_center_y_list]

    # rounding process
    basis_x_list = plain_basis_x_list[:]
    basis_y_list = plain_basis_y_list[:]
    if basis_x_list[0] < 0:
        basis_x_list[0] = 0
    if basis_x_list[-1] > width - 1 - cropped_size:
        basis_x_list[-1] = width - 1 - cropped_size
    if basis_y_list[0] < 0:
        basis_y_list[0] = 0
    if basis_y_list[-1] > height - 1 - cropped_size:
        basis_y_list[-1] = height - 1 - cropped_size

    if not ignore_duplicate:
        assert len(basis_x_list) == len(set(basis_x_list)), "There are duplicated x coordinates. (set `ignore_duplicate = True` if you want to ignore duplication.)"
        assert len(basis_x_list) == len(set(basis_x_list)), "There are duplicated y coordinates. (set `ignore_duplicate = True` if you want to ignore duplication.)"

    # cropping
    basis_coords = [(x_idx, y_idx) for x_idx in basis_x_list for y_idx in basis_y_list]
    crop_list = [x[:, y_idx:y_idx + cropped_size, x_idx:x_idx + cropped_size] for x_idx, y_idx in basis_coords]
    crop_tensor = torch.stack(crop_list)
    # (crop_tensor).shape = [crops, 5, 299, 299]
    return crop_tensor


def _crop_global_center(x, cropped_size):
    """ [EN] crop the center.   [JP] 元画像から中心を抽出．"""
    # config
    # cropped_size=299

    # culculating cropping area
    height = x.shape[-2]
    width = x.shape[-1]
    center_point = (round(width/2 - cropped_size/2), round(height/2 - cropped_size/2))
    # print((width / 2 - cropped_size / 2, height / 2 - cropped_size / 2))
    # print(center_point)
    # print(x.shape)

    # rounding_process
    if center_point[0] < 0 or center_point[1] < 0:
        temp_center_x = max(center_point[0], 0)
        temp_center_y = max(center_point[1], 0)
        center_point = (temp_center_x, temp_center_y)

    # print(center_point[0] + cropped_size, width)
    # print(center_point[1] + cropped_size, height)
    assert center_point[0] + cropped_size <= width, "x width error."
    assert center_point[1] + cropped_size <= height, "y width error."

    # cropping
    cropped_image = x[:, center_point[1]:center_point[1] + cropped_size, center_point[0]:center_point[0] + cropped_size]
    crop_tensor = torch.stack([cropped_image])
    # (crop_tensor).shape = [1, 3, 299, 299]
    return crop_tensor


class AVACoordShiftAbsDataset(Dataset):
    # def __init__(self, path_to_csv: str, images_path: str, crop_num: int, transform, np_seed=None):
    def __init__(self, path_to_csv: str, images_path: str, transform, cropped_size=None, np_seed=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        # self.crop_num = crop_num
        self.cropped_size = cropped_size
        self.np_seed = np_seed
        if self.np_seed:
            print("val dataset random crop seed:", self.np_seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        x = _add_abs_coord(x)
        if self.np_seed:
            x_crop = _random_seed_single_crop(x, cropped_size=self.cropped_size, random_seed = image_id + self.np_seed)
        else:
            # x_crop = self._random_single_crop(x)
            x_crop = _random_single_crop(x, cropped_size=self.cropped_size)

        return x_crop, p.astype('float32')


class AVAMPCoordShiftAbsDataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform, crop_num: int, cropped_size=None, np_seed=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        self.crop_num = crop_num
        self.cropped_size = cropped_size
        self.np_seed = np_seed
        if self.np_seed:
            print("val dataset random crop seed:", self.np_seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        x = _add_abs_coord(x)
        if self.np_seed:
            x_crops = _random_seed_crop(x, cropped_size=self.cropped_size, crop_num=self.crop_num, random_seed = image_id + self.np_seed)
        else:
            # x_crops = self._random_crop(x)
            x_crops = _random_crop(x, cropped_size=self.cropped_size, crop_num=self.crop_num)

        return x_crops, p.astype('float32')


class AVALocalCoordShiftAbsDataset(Dataset):
    """ Dataset class for making crops of 3*3 """
    def __init__(self, path_to_csv: str, images_path: str, transform, side_crop_num: int, cropped_size=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform_local = transform
        self.side_crop_num = side_crop_num
        self.cropped_size = cropped_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x_local = self.transform_local(image)
        x_local = _add_abs_coord(x_local)
        
        x_local_crops = _crop_local(x_local, cropped_size=self.cropped_size, side_crop_num=self.side_crop_num)
        x_crops = x_local_crops

        return x_crops, p.astype('float32')


class AVACoordShiftRelDataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform, cropped_size=None, np_seed=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        # self.crop_num = crop_num
        self.cropped_size = cropped_size
        self.np_seed = np_seed
        if self.np_seed:
            print("val dataset random crop seed:", self.np_seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        x = _add_rel_coord(x)
        if self.np_seed:
            x_crop = _random_seed_single_crop(x, cropped_size=self.cropped_size, random_seed = image_id + self.np_seed)
        else:
            x_crop = _random_single_crop(x, cropped_size=self.cropped_size)

        return x_crop, p.astype('float32')


class AVAMPCoordShiftRelDataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform, crop_num: int, cropped_size=None, np_seed=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        self.crop_num = crop_num
        self.cropped_size = cropped_size
        self.np_seed = np_seed
        if self.np_seed:
            print("val dataset random crop seed:", self.np_seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        x = _add_rel_coord(x)
        if self.np_seed:
            x_crops = _random_seed_crop(x, cropped_size=self.cropped_size, crop_num=self.crop_num, random_seed = image_id + self.np_seed)
        else:
            x_crops = _random_crop(x, cropped_size=self.cropped_size, crop_num=self.crop_num)

        return x_crops, p.astype('float32')


class AVALocalCoordShiftRelDataset(Dataset):
    """ Dataset class for making crops of 3*3 """
    def __init__(self, path_to_csv: str, images_path: str, transform, side_crop_num: int, cropped_size=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform_local = transform
        self.side_crop_num = side_crop_num
        self.cropped_size = cropped_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x_local = self.transform_local(image)
        x_local = _add_rel_coord(x_local)
        
        x_local_crops = _crop_local(x_local, cropped_size=self.cropped_size, side_crop_num=self.side_crop_num)
        x_crops = x_local_crops

        return x_crops, p.astype('float32')


class AVALocalGlobalCoordShiftRelDataset(Dataset):
    """ Dataset class for making crops of 3*3 """

    def __init__(self, path_to_csv: str, images_path: str, transform, transform_global, side_crop_num: int, cropped_size=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform_local = transform
        self.transform_global = transform_global
        self.side_crop_num = side_crop_num
        self.cropped_size = cropped_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x_local = self.transform_local(image)
        x_local = _add_rel_coord(x_local)
        x_global = self.transform_global(image)
        x_global = _add_rel_coord(x_global)
        
        x_local_crops = _crop_local(x_local, cropped_size=self.cropped_size, side_crop_num=self.side_crop_num)
        x_global_crops = _crop_global_center(x_global, cropped_size=self.cropped_size)
        x_crops = torch.cat((x_local_crops, x_global_crops), dim=0)
        
        return x_crops, p.astype('float32')


class AVACoordShiftAbsPseudoNormDataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform, cropped_size=None, np_seed=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        # self.crop_num = crop_num
        self.cropped_size = cropped_size
        self.np_seed = np_seed
        if self.np_seed:
            print("val dataset random crop seed:", self.np_seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        x = _add_abs_pseudo_norm_coord(x)
        if self.np_seed:
            x_crop = _random_seed_single_crop(x, cropped_size=self.cropped_size, random_seed = image_id + self.np_seed)
        else:
            # x_crop = self._random_single_crop(x)
            x_crop = _random_single_crop(x, cropped_size=self.cropped_size)

        return x_crop, p.astype('float32')


class AVAMPCoordShiftAbsPseudoNormDataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform, crop_num: int, cropped_size=None, np_seed=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        self.crop_num = crop_num
        self.cropped_size = cropped_size
        self.np_seed = np_seed
        if self.np_seed:
            print("val dataset random crop seed:", self.np_seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        x = _add_abs_pseudo_norm_coord(x)
        if self.np_seed:
            x_crops = _random_seed_crop(x, cropped_size=self.cropped_size, crop_num=self.crop_num, random_seed = image_id + self.np_seed)
        else:
            # x_crops = self._random_crop(x)
            x_crops = _random_crop(x, cropped_size=self.cropped_size, crop_num=self.crop_num)

        return x_crops, p.astype('float32')


class AVALocalCoordShiftAbsPseudoNormDataset(Dataset):
    """ Dataset class for making crops of 3*3 """
    def __init__(self, path_to_csv: str, images_path: str, transform, side_crop_num: int, cropped_size=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform_local = transform
        self.side_crop_num = side_crop_num
        self.cropped_size = cropped_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x_local = self.transform_local(image)
        x_local = _add_abs_pseudo_norm_coord(x_local)
        x_local_crops = _crop_local(x_local, cropped_size=self.cropped_size, side_crop_num=self.side_crop_num)
        x_crops = x_local_crops

        return x_crops, p.astype('float32')


class AVALocalGlobalCoordShiftAbsPseudoNormDataset(Dataset):
    """ Dataset class for making crops of 3*3 """
    def __init__(self, path_to_csv: str, images_path: str, transform, transform_global, side_crop_num: int, cropped_size=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform_local = transform
        self.transform_global = transform_global
        self.side_crop_num = side_crop_num
        self.cropped_size = cropped_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, str(image_id) + '.jpg')
        image = default_loader(image_path)
        x_local = self.transform_local(image)
        x_local = _add_abs_pseudo_norm_coord(x_local)
        x_global = self.transform_global(image)
        x_global = _add_abs_pseudo_norm_coord(x_global)
        
        x_local_crops = _crop_local(x_local, cropped_size=self.cropped_size, side_crop_num=self.side_crop_num)
        x_global_crops = _crop_global_center(x_global, cropped_size=self.cropped_size)
        x_crops = torch.cat((x_local_crops, x_global_crops), dim=0)
        # x_crops = x_local_crops

        return x_crops, p.astype('float32')

