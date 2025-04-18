import torch
import torchvision
import dataset_folders2 as dataset_folders


class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain

        if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013')| (dataset == 'kadid-10k') | (dataset == 'livec'):
            # Train transforms
            if istrain:
                transforms = torchvision.transforms.Compose([
                    # torchvision.transforms.RandomHorizontalFlip(),
                    # torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
            # Test transforms
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
        elif dataset == 'livemd':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((640, 360)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((640, 360)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])


        if dataset == 'live':
            self.data = dataset_folders.LIVEFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'kadid-10k':
            self.data = dataset_folders.Kadid_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'livec':
            self.data = dataset_folders.LIVEChallengeFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'livemd':
            self.data = dataset_folders.LIVEMDFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'csiq':
            self.data = dataset_folders.CSIQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'tid2013':
            self.data = dataset_folders.TID2013Folder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=16, shuffle=False)
        return dataloader


