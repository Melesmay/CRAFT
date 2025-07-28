from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import cv2
#import invert
import os

class DIREData(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        # self.root = root
        # self.transform = transform
        # self.image_paths = []
        # self.labels = []

        # for _,dataset  in enumerate(os.listdir(root)):
        #     #print(dataset)
        #     path = os.path.join(root, dataset)
        #     for n_class in os.listdir(path):
        #         if '.' not in n_class:
        #             for r, d, f in os.walk(os.path.join(path, n_class)):
        #                 for file in f:
        #                     file_pth = os.path.join(r, file).replace('/','\\')
        #                     self.image_paths.append(file_pth)
        #                     self.labels.append(n_class)
        super().__init__(root)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # img_path = self.image_paths[index]
        # label = self.labels[index]
        # print(img_path)
        # image = Image.fromarray(cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB))
        # if self.transform:
        #     image = self.transform(izmage)

        return path

loader = DIREData('E:/Datasets/DiffusionForensics/image/train')
print(loader.__getitem__(100))