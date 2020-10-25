import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
# from segnet import segnetLike
import copy
import random
import numpy as np
import pandas as pd
from nima.model import NIMA
from nima.mobile_net_v2 import mobile_net_v2
from inception_v3_custom import inception_v3, coord_inception_v3, coord_comp_inception_v3
from resnet_custom import resnet18_custom as resnet18
from utils import get_mean_score, get_std_score

""" Pr functions below. """

class Pr2clsResNet18(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(Pr2clsResNet18, self).__init__()
        base_model = resnet18(pretrained=pretrained_base_model)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.base_model = base_model

        self.head = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        # x: [Pr_bad, pr_good]
        return x


class Pr2clsInceptionV3(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(Pr2clsInceptionV3, self).__init__()
        base_model = inception_v3(pretrained=pretrained_base_model, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        # x: [Pr_bad, pr_good]
        return x


class Pr2clsMobileNetV2(nn.Module):
    """
    [EN] same as nima.model.NIMA  [JP] nima.model.NIMAと同一
    """
    def __init__(self, pretrained_base_model=True):
        super(Pr2clsMobileNetV2, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


""" NIMA functions below. """

class NIMAMobileNet(nn.Module):
    """
    [EN] same as nima.model.NIMA  [JP] nima.model.NIMAと同一
    """
    def __init__(self, pretrained_base_model=True):
        super(NIMAMobileNet, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class NIMAVGG16(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMAVGG16, self).__init__()
        base_model = models.vgg16(pretrained=pretrained_base_model)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])
        base_model.classifier = nn.Sequential(*list(base_model.classifier.children())[:-2])

        self.base_model = base_model

        self.head = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class NIMAInceptionV3(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMAInceptionV3, self).__init__()
        base_model = inception_v3(pretrained=pretrained_base_model, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class NIMACoordInceptionV3(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMACoordInceptionV3, self).__init__()
        base_model = coord_inception_v3(pretrained=pretrained_base_model, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class NIMACoordCompInceptionV3(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMACoordCompInceptionV3, self).__init__()
        base_model = coord_comp_inception_v3(pretrained=pretrained_base_model, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class NIMAImportanceInceptionV3(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMAImportanceInceptionV3, self).__init__()
        base_model = inception_v3(pretrained=pretrained_base_model, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

        self.selector = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(500, 1)
        )

    def forward(self, x):
        x = self.base_model(x)
        # x = x.view(x.size(0), -1)
        feature = x.view(x.size(0), -1)
        # x = self.head(x)
        x = self.head(feature)
        return x, feature

    def selector_feature(self, feature):
        pred_EMD_loss = self.selector(feature) # (x).shape = [1]
        return pred_EMD_loss.mean(), pred_EMD_loss


class NIMAResNet18(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMAResNet18, self).__init__()
        base_model = resnet18(pretrained=pretrained_base_model)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.base_model = base_model

        self.head = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        # x: [Pr_bad, pr_good]
        return x


class Softplusmax(nn.Module):
    def __init__(self, dim=None, beta=1, threshold=20):
        super(Softplusmax, self).__init__()
        self.dim = dim
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x):
        x = self.softplus(x)
        x = x / x.sum(dim=self.dim, keepdim=True)
        return x


class NIMASoftplusmaxMobileNet(nn.Module):
    """
    [EN] same as nima.model.NIMA  [JP] nima.model.NIMAと同一
    """
    def __init__(self, pretrained_base_model=True):
        super(NIMASoftplusmaxMobileNet, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            # nn.Softmax(dim=1)
            Softplusmax(dim=1, beta=1, threshold=20)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class NIMASoftplusmaxVGG16(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMASoftplusmaxVGG16, self).__init__()
        base_model = models.vgg16(pretrained=pretrained_base_model)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])
        base_model.classifier = nn.Sequential(*list(base_model.classifier.children())[:-2])

        self.base_model = base_model

        self.head = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(4096, 10),
            # nn.Softmax(dim=1)
            Softplusmax(dim=1, beta=1, threshold=20)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class NIMASoftplusmaxInceptionV3(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMASoftplusmaxInceptionV3, self).__init__()
        base_model = inception_v3(pretrained=pretrained_base_model, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 10),
            # nn.Softmax(dim=1)
            Softplusmax(dim=1, beta=1, threshold=20)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


""" MP score models below. """

# class MPAesScore(nn.Module):
#     """
#     deprecated
#     class MPCroppedAesScore()にほぼ引き継ぎ
#     """
#     def __init__(self, nima_pretrained=False, path_to_nima="nima/pretrain-model.pth", crop_num = 32):
#         """
#         :param nima_pretrained: 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
#         """
#         super(MPAesScore, self).__init__()
#         self.crop_num = crop_num
#         self.nima = NIMA(pretrained_base_model=False)
#         # init nima weights
#         if nima_pretrained: # True at train
#             if path_to_nima is None:
#                 raise ValueError
#             state_dict = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
#             self.nima.load_state_dict(state_dict)
#         self.network_size = 224
#
#     def forward(self, img):
#         # img = img.squeeze(dim=0)
#         output_list = []
#         for one_img in img:
#             height = one_img.shape[-2]
#             width = one_img.shape[-1]
#             x_list = np.random.randint(0, width-self.network_size, size=self.crop_num)
#             y_list = np.random.randint(0, height-self.network_size, size=self.crop_num)
#             crop_list = [one_img[:, y:y+self.network_size, x:x+self.network_size] for x, y in zip(x_list, y_list)]
#             crop_tensor = torch.stack(crop_list)
#             # (crop_tensor).shape = [crops, 3, 224, 224]
#             temp_output = self.nima(crop_tensor)
#             # (temp_output).shape = [crops, 10]
#             output_list.append(temp_output)
#
#         output = torch.stack(output_list)
#         # (output).shape = [N, crops, 10]
#         return output


class MPCroppedAesScore(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScore, self).__init__()
        self.nima = NIMA(pretrained_base_model=False)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        return output


class MPCroppedAesScoreNIMAVGG16(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScoreNIMAVGG16, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = NIMAVGG16(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        return output


class MPCroppedAesScoreNIMAInceptionV3(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScoreNIMAInceptionV3, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = NIMAInceptionV3(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        return output


class MPCroppedAesScoreNIMACoordInceptionV3(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScoreNIMACoordInceptionV3, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = NIMACoordInceptionV3(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        return output


class MPCroppedAesScoreNIMACoordCompInceptionV3(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScoreNIMACoordCompInceptionV3, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = NIMACoordCompInceptionV3(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        return output


class MPCroppedAesScoreNIMAImportanceInceptionV3(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScoreNIMAImportanceInceptionV3, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = NIMAImportanceInceptionV3(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output, temp_feature = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        # (temp_feature).shape = [N*crops, 2048]
        _, temp_pred_loss = self.nima.selector_feature(temp_feature)
        # (temp_pred_loss).shape = [N*crops, 1]
        pred_EMD_loss = temp_pred_loss.view(orig_shape[0], orig_shape[1], -1)
        # (pred_loss).shape = [N, crops. 1]
        importance = F.softmax(-pred_EMD_loss, dim=1)
        # (importance).shape = [N, crops, 1]
        output_raw = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        output = output_raw * importance * orig_shape[1] # orig_shape[1]はクロップ数

        return output


class MPCroppedAesScoreNIMASoftplusmaxMobileNetV2(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScoreNIMASoftplusmaxMobileNetV2, self).__init__()
        self.nima = NIMASoftplusmaxMobileNet(pretrained_base_model=False)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        return output


class MPCroppedAesScoreNIMASoftplusmaxVGG16(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScoreNIMASoftplusmaxVGG16, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = NIMASoftplusmaxVGG16(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        return output


class MPCroppedAesScoreNIMASoftplusmaxInceptionV3(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedAesScoreNIMASoftplusmaxInceptionV3, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = NIMASoftplusmaxInceptionV3(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained: # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location = lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 10]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 10]
        return output


""" MP score models (with attention) below """
class MPAttentionFeatureInceptionV3(nn.Module):
    def __init__(self, crop_num:int, imagenet_pretrained=False, nima_pretrained=True, path_to_nima=None):
        super(MPAttentionFeatureInceptionV3, self).__init__()
        base_model = inception_v3(pretrained=imagenet_pretrained, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(2048*crop_num, crop_num),
            nn.Softmax(dim=1)
        )

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

        if nima_pretrained:
            checkpoint = torch.load(path_to_nima)
            state_dict = checkpoint["state_dict"]
            each_state_dict = dict()
            each_state_dict["base_model"] = OrderedDict([(key.replace("base_model.", ""), value) for key, value in state_dict.items() if key.startswith("base_model.")])
            each_state_dict["head"] = OrderedDict(
                [(key.replace("head.", ""), value) for key, value in state_dict.items() if
                 key.startswith("head.")])
            self.base_model.load_state_dict(each_state_dict["base_model"])
            self.head.load_state_dict(each_state_dict["head"])

    def forward(self, x):
        # (x).shape = [N, crop_num, Channel, height, width]
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1]) # (x).shape = [N*crops, Channel, height, width]
        x = self.base_model(x) # (x).shape = [N*crop_num, ...]
        # print(x.shape)
        feature_shape = x.shape
        x = x.view(orig_shape[0], orig_shape[1], *list(feature_shape)[1:])  # (x).shape = [N, crop_num, ...]

        # culculate attention
        y = x.view(orig_shape[0], -1) # (y).shape = [N, 2048*crop_num]
        y = self.attention(y) # (y).shape = [N, crop_num]
        y = y.view(y.shape[0], y.shape[1], *([1]*(len(x.shape)-2))) # (y).shape = [N, crop_num, 1, 1, 1]
        # culculate attention end

        # print(x.shape, y.shape)
        # exit()

        x = x*y                     # (x).shape = [N, crop_num, ...]
        x = x.sum(dim=1)            # (x).shape = [N, ...]
        x = x.view(x.size(0), -1)   # (x).shape = [N, 2048]
        # print(x.shape)
        x = self.head(x)
        return x


class MPAttentionDistributionInceptionV3(nn.Module):
    def __init__(self, crop_num:int, imagenet_pretrained=False, nima_pretrained=True, path_to_nima=None):
        super(MPAttentionDistributionInceptionV3, self).__init__()
        base_model = inception_v3(pretrained=imagenet_pretrained, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.attention = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.75),
            nn.Linear(2048*crop_num, crop_num),
            nn.Softmax(dim=1)
        )

        self.head = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

        if nima_pretrained:
            checkpoint = torch.load(path_to_nima)
            state_dict = checkpoint["state_dict"]
            each_state_dict = dict()
            each_state_dict["base_model"] = OrderedDict([(key.replace("base_model.", ""), value) for key, value in state_dict.items() if key.startswith("base_model.")])
            each_state_dict["head"] = OrderedDict(
                [(key.replace("head.", ""), value) for key, value in state_dict.items() if
                 key.startswith("head.")])
            self.base_model.load_state_dict(each_state_dict["base_model"])
            self.head.load_state_dict(each_state_dict["head"])

    def forward(self, x):
        # (x).shape = [N, crop_num, Channel, height, width]
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1]) # (x).shape = [N*crops, Channel, height, width]
        x = self.base_model(x) # (x).shape = [N*crop_num, ...]

        # culculate attention
        feature_shape = x.shape
        y = x.view(orig_shape[0], orig_shape[1], *list(feature_shape)[1:])  # (x).shape = [N, crop_num, ...] # maybe not necessary
        y = y.view(orig_shape[0], -1) # (y).shape = [N, 2048*crop_num]
        y = self.attention(y) # (y).shape = [N, crop_num]
        # culculate attention end

        # print(x.shape, y.shape)
        # exit()

        x = x.view(x.size(0), -1)   # (x).shape = [N*crop_num, 2048]
        # print(x.shape)
        x = self.head(x)            # (x).shape = [N*crop_num, 10]

        # apply attention
        feature_shape = x.shape
        x = x.view(orig_shape[0], orig_shape[1], *list(feature_shape)[1:])  # (x).shape = [N, crop_num, 10]
        y = y.view(y.shape[0], y.shape[1], *([1] * (len(x.shape) - 2)))  # (y).shape = [N, crop_num, 1]
        x = x * y  # (x).shape = [N, crop_num, 10]
        x = x.sum(dim=1)  # (x).shape = [N, 10]

        return x


# TODO: implementation
class MPSingleAttentionDistributionInceptionV3(nn.Module):
    def __init__(self, imagenet_pretrained=False, nima_pretrained=True, path_to_nima=None):
        super(MPSingleAttentionDistributionInceptionV3, self).__init__()
        base_model = inception_v3(pretrained=imagenet_pretrained, aux_logits=False)
        # base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.attention = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.75),
            # nn.Linear(2048*crop_num, crop_num),
            nn.Linear(2048, 1),
            # nn.Softmax(dim=1)
        )

        self.head = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.75),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

        if nima_pretrained:
            checkpoint = torch.load(path_to_nima)
            state_dict = checkpoint["state_dict"]
            each_state_dict = dict()
            each_state_dict["base_model"] = OrderedDict([(key.replace("base_model.", ""), value) for key, value in state_dict.items() if key.startswith("base_model.")])
            each_state_dict["head"] = OrderedDict(
                [(key.replace("head.", ""), value) for key, value in state_dict.items() if
                 key.startswith("head.")])
            self.base_model.load_state_dict(each_state_dict["base_model"])
            self.head.load_state_dict(each_state_dict["head"])

    def forward(self, x):
        # (x).shape = [N, crop_num, Channel, height, width]
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1]) # (x).shape = [N*crops, Channel, height, width]
        x = self.base_model(x) # (x).shape = [N*crop_num, ...]

        # calculate attention
        feature_shape = x.shape
        # y = x.view(orig_shape[0], orig_shape[1], *list(feature_shape)[1:])  # (x).shape = [N, crop_num, ...] # maybe not necessary

        # y = y.view(orig_shape[0], -1) # (y).shape = [N, 2048*crop_num]
        y = x.view(feature_shape[0], -1) # (y).shape = [N*crop_num, 2048]
        y = self.attention(y) # (y).shape = [N*crop_num, 1]
        feature_shape = y.shape
        y = y.view(orig_shape[0], orig_shape[1], *list(feature_shape)[1:]) # (y).shape = [N, crops, 1]
        y = F.softmax(y, dim=1) # (y).shape = [N, crops, 1]
        feature_shape = y.shape
        y = y.view(orig_shape[0]*orig_shape[1], *list(feature_shape)[2:]) # (y).shape = [N*crops, 1]

        # print("y:", y)


        # culculate attention end

        # print(x.shape, y.shape)
        # exit()

        x = x.view(x.size(0), -1)   # (x).shape = [N*crop_num, 2048]
        # print(x.shape)
        x = self.head(x)            # (x).shape = [N*crop_num, 10]
        # print("x:", x)

        # apply attention
        x = x*y # (x).shape = [N*crop_num, 10]
        # print("out:", x)
        feature_shape = x.shape
        x = x.view(orig_shape[0], orig_shape[1], *list(feature_shape)[1:]) # (x).shape = [N, crop_num, 10]

        # print(x.size())

        """
        feature_shape = x.shape
        x = x.view(orig_shape[0], orig_shape[1], *list(feature_shape)[1:])  # (x).shape = [N, crop_num, 10]
        y = y.view(y.shape[0], y.shape[1], *([1] * (len(x.shape) - 2)))  # (y).shape = [N, crop_num, 1]
        x = x * y  # (x).shape = [N, crop_num, 10]
        x = x.sum(dim=1)  # (x).shape = [N, 10]
        """

        x = x.sum(dim=1) # (x).shape = [N, 10]
        # print(x.size())
        # print(x)
        # print(x.sum(dim=1)) # x.sum(dim=1).shape = [N]

        return x


""" MP pr models below. """

class MPCroppedPr2clsResNet18(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedPr2clsResNet18, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = Pr2clsResNet18(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained:  # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location=lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape  # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 2]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 2]
        return output


class MPCroppedPr2clsInceptionV3(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedPr2clsInceptionV3, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = Pr2clsInceptionV3(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained:  # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location=lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape  # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 2]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 2]
        return output


class MPCroppedPr2clsMobileNetV2(nn.Module):
    def __init__(self, nima_pretrained=False, path_to_nima=None):
        """
        :param nima_pretrained: [EN] Specify false if load a model at evaluation. Please set true for training since the code use a NIMA pretrained model.   [JP] 後からevaluation時にモデルをロードするならFalse, train時はNIMA pretrainedを使うためTrue．
        """
        super(MPCroppedPr2clsMobileNetV2, self).__init__()
        # self.nima = NIMA(pretrained_base_model=False)
        self.nima = Pr2clsMobileNetV2(pretrained_base_model=True)
        # init nima weights
        if nima_pretrained:  # True at train
            if path_to_nima is None:
                raise ValueError
            checkpoint = torch.load(path_to_nima, map_location=lambda storage, loc: storage)
            self.nima.load_state_dict(checkpoint["state_dict"])

    def forward(self, img):
        # img = img.squeeze(dim=0)
        orig_shape = img.shape  # [N, crops, Channel, height, width]
        crop_tensor = img.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
        # (crop_tensor).shape = [N*crops, Chennel, height, width]
        temp_output = self.nima(crop_tensor)
        # (temp_output).shape = [N*crops, 2]
        output = temp_output.view(orig_shape[0], orig_shape[1], -1)
        # (output).shape = [N, crops, 2]
        return output


""" loss score fuctions below. """

class MPAdaEMDLossSigmoid(nn.Module):
    def __init__(self, beta, pr_weight):
        super(MPAdaEMDLossSigmoid, self).__init__()
        self.emd_loss = EMDLoss()
        self.beta = beta
        self.pr_weight = pr_weight

    def forward(self, output, target):
        #! new part
        # (output, target).shape = [N, crops, 10], [N,10]
        N, crops, scores = output.shape
        target_expand = target.view(N, 1, scores).repeat(1, crops, 1) # [N, crops, 10]
        target_rs = target_expand.view(-1, scores) # [N*crops, 10]
        output_rs = output.view(-1, scores) # [N*crops, 10]
        pr_rs = self.emd_loss(output_rs, target_rs)[1] # [N*crops]
        pr = pr_rs.view(N, crops) # [N, crops]
        #! new part (end)

        pr = 1 - torch.sigmoid(pr * self.pr_weight).clamp(0.000001, 1.)
        # print("pr:", "(mean)", "{:.10f}".format(pr.mean()), "\t(min)", "{:.10f}".format(pr.min()), "\t(max)", "{:.10f}".format(pr.max()))
        wb = 1 - torch.pow(pr, self.beta)
        product = wb * torch.log(pr)
        # to maximize product = to minimize -1*product
        loss = -product
        return loss.mean(), wb.mean(), loss.mean(dim=1), wb


class MPAdaEMDLossAdv(nn.Module):
    def __init__(self, beta, pr_weight):
        super(MPAdaEMDLossAdv, self).__init__()
        self.emd_loss = EMDLoss()
        self.beta = beta
        self.pr_weight = pr_weight

    def forward(self, output, target):
        # (output, target).shape = [N, crops, 10], [N,10]
        N, crops, scores = output.shape
        target_expand = target.view(N, 1, scores).repeat(1, crops, 1) # [N, crops, 10]
        target_rs = target_expand.view(-1, scores) # [N*crops, 10]
        output_rs = output.view(-1, scores) # [N*crops, 10]
        pr_rs = self.emd_loss(output_rs, target_rs)[1] # [N*crops]
        pr = pr_rs.view(N, crops) # [N, crops]
        
        pr = (1-pr*self.pr_weight).clamp(0.00001, 1.)
        # print("pr:", "(mean)", "{:.10f}".format(pr.mean()), "\t(min)", "{:.10f}".format(pr.min()), "\t(max)", "{:.10f}".format(pr.max()))
        wb = 1 - torch.pow(pr, self.beta)
        product = wb * torch.log(pr)
        # to maximize product = to minimize -1*product
        loss = -product
        return loss.mean(), wb.mean(), loss.mean(dim=1), wb


class MPAvgEMDLoss(nn.Module):
    # def __init__(self, beta, pr_weight):
    def __init__(self, pr_weight):
        super(MPAvgEMDLoss, self).__init__()
        self.emd_loss = EMDLoss()
        # self.beta = beta
        self.pr_weight = pr_weight

    def forward(self, output, target):
        # (output, target).shape = [N, crops, 10], [N,10]
        N, crops, scores = output.shape
        target_expand = target.view(N, 1, scores).repeat(1, crops, 1) # [N, crops, 10]
        target_rs = target_expand.view(-1, scores) # [N*crops, 10]
        output_rs = output.view(-1, scores) # [N*crops, 10]
        pr_rs = self.emd_loss(output_rs, target_rs)[1] # [N*crops]
        pr = pr_rs.view(N, crops) # [N, crops]
        
        pr = (1-pr*self.pr_weight).clamp(0.00001, 1.)
        # print("pr:", "(mean)", "{:.10f}".format(pr.mean()), "\t(min)", "{:.10f}".format(pr.min()), "\t(max)", "{:.10f}".format(pr.max()))
        # wb = 1 - torch.pow(pr, self.beta)
        # product = wb * torch.log(pr)
        product = torch.log(pr)
        # to maximize product = to minimize -1*product
        loss = -product
        # return loss.mean(), wb.mean(), loss.mean(dim=1), wb
        return loss.mean(), torch.Tensor([1.], device=loss.device), loss.mean(dim=1), torch.ones((loss.mean(dim=1)).shape, device=loss.device)


class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()
    def forward(self, output, target):
        assert output.shape == target.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_output = torch.cumsum(output, dim=1)
        cdf_diff = cdf_output - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        return samplewise_emd.mean(), samplewise_emd

class WeightedEMDLoss(nn.Module):
    def __init__(self, beta, pr_weight):
        super(WeightedEMDLoss, self).__init__()
        self.emd_loss = EMDLoss()
        self.beta = beta
        self.pr_weight = pr_weight

    def forward(self, output, target):
        assert output.shape == target.shape

        # samplewise_emd_mean, samplewise_emd = self.emd_loss(output, target)
        _, samplewise_emd = self.emd_loss(output, target)

        pr = (1 - samplewise_emd * self.pr_weight).clamp(0.000001, 1.)
        # print("pr:", "(mean)", "{:.10f}".format(pr.mean()), "\t(min)", "{:.10f}".format(pr.min()), "\t(max)", "{:.10f}".format(pr.max()))
        wb = 1 - torch.pow(pr, self.beta)
        loss = wb * samplewise_emd

        return loss.mean(), loss


class AdaEMDLoss(nn.Module):
    def __init__(self, beta, pr_weight):
        super(AdaEMDLoss, self).__init__()
        self.emd_loss = EMDLoss()
        self.beta = beta
        self.pr_weight = pr_weight

    def forward(self, output, target):
        assert output.shape == target.shape

        # samplewise_emd_mean, samplewise_emd = self.emd_loss(output, target)
        _, samplewise_emd = self.emd_loss(output, target)

        pr = (1 - samplewise_emd * self.pr_weight).clamp(0.000001, 1.)
        # print("pr:", "(mean)", "{:.10f}".format(pr.mean()), "\t(min)", "{:.10f}".format(pr.min()), "\t(max)", "{:.10f}".format(pr.max()))
        wb = 1 - torch.pow(pr, self.beta)
        # samplewise_emd = wb * samplewise_emd
        product = wb * torch.log(pr)
        # to maximize product = to minimize -1*product
        loss = -product
        # loss.shape = (N) (probably)

        # return samplewise_emd.mean(), samplewise_emd
        return loss.mean(), loss, samplewise_emd.mean(), samplewise_emd

class AvgEMDLoss(nn.Module):
    def __init__(self, pr_weight):
        super(AvgEMDLoss, self).__init__()
        self.emd_loss = EMDLoss()
        self.pr_weight = pr_weight

    def forward(self, output, target):
        assert output.shape == target.shape

        # samplewise_emd_mean, samplewise_emd = self.emd_loss(output, target)
        _, samplewise_emd = self.emd_loss(output, target)

        pr = (1 - samplewise_emd * self.pr_weight).clamp(0.000001, 1.)
        # print("pr:", "(mean)", "{:.10f}".format(pr.mean()), "\t(min)", "{:.10f}".format(pr.min()), "\t(max)", "{:.10f}".format(pr.max()))
        # wb = 1 - torch.pow(pr, self.beta)
        # samplewise_emd = wb * samplewise_emd
        # product = wb * torch.log(pr)
        product = torch.log(pr)
        # to maximize product = to minimize -1*product
        loss = -product
        # loss.shape = (N) (probably)

        # return samplewise_emd.mean(), samplewise_emd
        return loss.mean(), loss, samplewise_emd.mean(), samplewise_emd


class MPEMDLoss(nn.Module):
    def __init__(self):
        super(MPEMDLoss, self).__init__()
    def forward(self, output, target):
        # (output) = (N, crops, 10)
        # calculate mean of patches
        output = output.mean(dim=1)
        # (output) = (N, 10)
        assert output.shape == target.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_output = torch.cumsum(output, dim=1)
        cdf_diff = cdf_output - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        return samplewise_emd.mean(), samplewise_emd


class MPWeightedEMDLoss(nn.Module):
    def __init__(self, beta, pr_weight):
        super(MPWeightedEMDLoss, self).__init__()
        self.emd_loss = EMDLoss()
        self.beta = beta
        self.pr_weight = pr_weight

    def forward(self, output, target):
        # assert output.shape == target.shape
        # (output, target).shape = [N, crops, 10], [N,10]
        #! calculate EMD
        N, crops, scores = output.shape
        target_expand = target.view(N, 1, scores).repeat(1, crops, 1)  # [N, crops, 10]
        target_rs = target_expand.view(-1, scores)  # [N*crops, 10]
        output_rs = output.view(-1, scores)  # [N*crops, 10]
        samplewise_emd_rs = self.emd_loss(output_rs, target_rs)[1]  # [N*crops]
        samplewise_emd = samplewise_emd_rs.view(N, crops) # [N, crops]
        #! calculate EMD (end)

        pr = (1 - samplewise_emd * self.pr_weight).clamp(0.000001, 1.)
        # print("pr:", "(mean)", "{:.10f}".format(pr.mean()), "\t(min)", "{:.10f}".format(pr.min()), "\t(max)", "{:.10f}".format(pr.max()))
        wb = 1 - torch.pow(pr, self.beta)
        loss = wb * samplewise_emd

        return loss.mean(), wb.mean(), loss.mean(dim=1), wb


class CDFLayer(nn.Module):
    def __init__(self):
        super(CDFLayer, self).__init__()

    def forward(self, p:torch.Tensor):
        cdf_p = torch.cumsum(p, dim=1)
        return cdf_p


""" loss pr functions below. """
class MPAdaPrLoss(nn.Module):
    def __init__(self, beta):
        super(MPAdaPrLoss, self).__init__()
        self.beta = beta

    def forward(self, output, target):
        #! new part
        # (output, target).shape = [N, crops, 2], [N,10]
        N, crops, scores = output.shape
        target_np = target.cpu().numpy()
        # (target_np).shape = [N, 10]
        target_mean_scores = np.array([get_mean_score(row) for row in target_np])
        # (target_mean_scores).shape = [N]
        target_bin_np = target_mean_scores > 5 # [EN] set a flag as 1 for aesthetic images   [JP] aesthetic imageだったらflag=1を立てる．
        # (target_bin_np).shape = [N]
        target_bin = torch.from_numpy(target_bin_np.astype(np.float32)).to(target.device)
        target_bin = target_bin.view(-1, 1, 1)
        # (target_bin).shape = [N, 1, 1]
        target_bin = target_bin.repeat(1, crops, 2)
        # (target_bin).shape = [N, crops, 2]
        target_bin[:, :, 0] = 1 - target_bin[:, :, 0] # [EN] the order of [:, :, 2] is [bad, good].   [JP] [:, :, 2]の2の部分が[bad, good]の順番になるようにする．
        # (target_bin).shape = [N, crops, 2]


        """ new in pr2cls. """
        pr_rs = output * target_bin
        # (pr_rs).shape = [N, crops, 2]
        pr = pr_rs.sum(dim=2)
        # (pr).shape = [N, crops]
        wb = 1 - torch.pow(pr, self.beta)
        product = wb * torch.log(pr)
        # to maximize product = to minimize -1*product
        loss = -product
        return loss.mean(), wb.mean(), loss.mean(dim=1), wb


class MPAvgPrLoss(nn.Module):
    def __init__(self):
        super(MPAvgPrLoss, self).__init__()
        # self.beta = beta

    def forward(self, output, target):
        #! new part
        # (output, target).shape = [N, crops, 2], [N,10]
        N, crops, scores = output.shape
        target_np = target.cpu().numpy()
        # (target_np).shape = [N, 10]
        target_mean_scores = np.array([get_mean_score(row) for row in target_np])
        # (target_mean_scores).shape = [N]
        target_bin_np = target_mean_scores > 5 # [EN] set a flag as 1 for aesthetic images   [JP] aesthetic imageだったらflag=1を立てる．
        # (target_bin_np).shape = [N]
        target_bin = torch.from_numpy(target_bin_np.astype(np.float32)).to(target.device)
        target_bin = target_bin.view(-1, 1, 1)
        # (target_bin).shape = [N, 1, 1]
        target_bin = target_bin.repeat(1, crops, 2)
        # (target_bin).shape = [N, crops, 2]
        target_bin[:, :, 0] = 1 - target_bin[:, :, 0] # [EN] the order of [:, :, 2] is [bad, good].  [JP] [:, :, 2]の2の部分が[bad, good]の順番になるようにする．
        # (target_bin).shape = [N, crops, 2]


        """ new in pr2cls. """
        pr_rs = output * target_bin
        # (pr_rs).shape = [N, crops, 2]
        pr = pr_rs.sum(dim=2)
        # (pr).shape = [N, crops]
        # wb = 1 - torch.pow(pr, self.beta)
        # product = wb * torch.log(pr)
        product = torch.log(pr)
        # to maximize product = to minimize -1*product
        loss = -product
        return loss.mean(), torch.Tensor([1.], device=loss.device), loss.mean(dim=1), torch.ones((loss.mean(dim=1)).shape, device=loss.device)


class MPPrLoss(nn.Module):
    def __init__(self, beta):
        super(MPPrLoss, self).__init__()
        self.beta = beta

    def forward(self, output, target):
        #! new part
        # (output, target).shape = [N, crops, 2], [N,10]
        N, crops, scores = output.shape
        target_np = target.cpu().numpy()
        # (target_np).shape = [N, 10]
        target_mean_scores = np.array([get_mean_score(row) for row in target_np])
        # (target_mean_scores).shape = [N]
        target_bin_np = target_mean_scores > 5 #  [EN] set a flag as 1 for aesthetic images   [JP] aesthetic imageだったらflag=1を立てる．
        # (target_bin_np).shape = [N]
        target_bin = torch.from_numpy(target_bin_np.astype(np.float32)).to(target.device)
        target_bin = target_bin.view(-1, 1, 1)
        # (target_bin).shape = [N, 1, 1]
        target_bin = target_bin.repeat(1, crops, 2)
        # (target_bin).shape = [N, crops, 2]
        target_bin[:, :, 0] = 1 - target_bin[:, :, 0] # [EN] the order of [:, :, 2] is [bad, good].   [JP] [:, :, 2]の2の部分が[bad, good]の順番になるようにする．
        # (target_bin).shape = [N, crops, 2]

        pr_rs = output * target_bin
        # (pr_rs).shape = [N, crops, 2]
        pr = pr_rs.sum(dim=2)
        # (pr).shape = [N, crops]
        # wb = 1 - torch.pow(pr, self.beta)
        log_pr = torch.log(pr)
        # (log_pr).shape = [N, crops]
        # to maximize product = to minimize -1*product
        loss = -log_pr
        return loss.mean(), pr.mean(), loss.mean(dim=1), pr.mean(dim=1)


class PrLoss(nn.Module):
    def __init__(self):
        super(PrLoss, self).__init__()

    def forward(self, output, target):
        # (output, target).shape = [N, 2], [N, 10]
        N, scores = output.shape
        target_np = target.cpu().numpy()
        # (target_np).shape = [N, 10]
        target_mean_scores = np.array([get_mean_score(row) for row in target_np])
        # (target_mean_scores).shape = [N]
        target_bin_np = target_mean_scores > 5  #  [EN] set a flag as 1 for aesthetic images   [JP] aesthetic imageだったらflag=1を立てる．
        # (target_bin_np).shape = [N]
        target_bin = torch.from_numpy(target_bin_np.astype(np.float32)).to(target.device)
        target_bin = target_bin.view(-1, 1)
        # (target_bin).shape = [N, 1]
        target_bin = target_bin.repeat(1, 2)
        # (target_bin).shape = [N, 2]
        target_bin[:, 0] = 1 - target_bin[:, 0]  # [EN] the order of [:, :, 2] is [bad, good].   [JP] [:, 2]の2の部分が[bad, good]の順番になるようにする．
        # (target_bin).shape = [N, 2]

        pr_rs = output * target_bin
        # (pr_rs).shape = [N, 2]
        pr = pr_rs.sum(dim=1)
        # (pr).shape = [N]
        # wb = 1 - torch.pow(pr, self.beta)
        log_pr = torch.log(pr)
        loss = -log_pr
        return loss.mean(), pr.mean(), loss, pr

# class RMSEAttentionLayer(nn.Module):
#     def __init__(self):
#         super(RMSEAttentionLayer, self).__init__()
#
#     def forward(self,):
#         pass