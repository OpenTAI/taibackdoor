import torch
import numpy as np
from torchvision import datasets
from glob import glob

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class IBAImageNetClean(datasets.folder.ImageFolder):
    def __init__(self, root, transform=None, mode=None, **kwargs):
        super().__init__(root=root, transform=transform)
        targets = np.array([item[1] for item in self.samples])
        cidx = [np.where(targets == i)[0] for i in range(10)]
        new_samples = []
        for idx in cidx:
            for i in idx:
                new_samples.append(self.samples[i])
        self.samples = new_samples
        print('clean_sample_count', len(self))


class IBAImageNet(datasets.folder.ImageFolder):
    def __init__(self, root, transform=None, mode=None, **kwargs):
        super().__init__(root=root, transform=transform)
        clean_samples = self.__len__()
        print(root, mode)
        if 'backdoor_path' in kwargs:
            backdoor_path = kwargs['backdoor_path']
            target_label = kwargs['target_label']
            bd_ratio = kwargs['bd_ratio']
            bd_list = glob(backdoor_path + '/' + mode + '/*_hidden*')[:]
            n = int(len(bd_list)*bd_ratio)
            self.poison_idx = np.array(range(len(self), len(self)+n))
            if mode == 'train':
                bd_list = bd_list[:n]
            new_targets = [target_label] * len(bd_list)

            if mode == 'train':
                self.samples += list(zip(bd_list, new_targets))
                self.imgs = self.samples
            else:
                self.samples = list(zip(bd_list, new_targets))
                self.imgs = self.samples
            backdoor_samples = len(bd_list)

        if mode == 'train':
            print('clean_sample_count', clean_samples)
        print('backdoor_samples_count', backdoor_samples)
        print('total', self.__len__())
