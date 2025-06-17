import glob
import random
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from Functions import resize_image_with_proportion


ROOT=r'F:\datasets\Facades'
MIN_SIZE=256
BS=8


def get_jpg_files_glob(folder_path):
    """使用通配符匹配所有.jpg文件"""
    pattern = os.path.join(folder_path, "*.jpg")
    return [ os.path.splitext(os.path.basename(f))[0]  for f in glob.glob(pattern)]



tr=transforms.Compose([
    # transforms.Resize((MIN_SIZE, MIN_SIZE)),       # 调整图像尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class FacadesDataset(Dataset):
    def __init__(self, root=ROOT, t1=tr, mode="train"):
        '''
        :param root:
            F:\datasets\Facades
            |----------train/
            |       |--------A/
            |              |-----xxx.jpg
            |       |--------B/
            |              |-----xxx.jpg
            |----------test/
            |       |--------A/
            |              |-----xxx.jpg
            |       |--------B/
            |              |-----xxx.jpg
            |......
        :param t1:
        :param mode: train / test
        '''
        self.root=root
        self.mode = mode
        self.t1 = t1
        self.t2 = transforms.PILToTensor()

        self.img_fielname = None  # ['100_A', '101_A', '102_A', '103_A', '104_A', '105_A', '106_A',...]
        if mode=='train':
            self.img_fielname=get_jpg_files_glob(root + "\\train\\A")
        elif mode=='test':
            self.img_fielname = get_jpg_files_glob(root + "\\test\\A")
        else:
            print('mode只能为train或test')
            exit(-1)

    def __getitem__(self, index):
        if self.mode=='train':
            num=self.img_fielname[index].split('_')[0]
            img_path=self.root+'\\train\\A\\'+num+'_A.jpg'
            lab_path=self.root+'\\train\\B\\'+num+'_B.jpg'
            img = Image.open(img_path)
            lab = Image.open(lab_path)
            if self.t1 is not None:
                img = self.t1(img)
            return img,self.t2(lab).type(torch.float32)
        elif self.mode=='test':
            num = self.img_fielname[index].split('_')[0]
            img_path = self.root + '\\test\\A\\' + num + '.jpg'
            lab_path = self.root + '\\test\\B\\' + num + '.jpg'
            img = Image.open(img_path)
            lab = Image.open(lab_path)
            if self.t1 is not None:
                img = self.t1(img)
            return img, self.t2(lab).type(torch.float32)
        else:
            exit(-1)

    def __len__(self):
        return len(self.img_fielname)



data_train=FacadesDataset(mode='train')
data_test=FacadesDataset(mode='test')

loader_train = DataLoader(
    dataset=data_train,
    batch_size=BS,
    shuffle=True
)
loader_test = DataLoader(
    dataset=data_test,
    batch_size=BS,
    shuffle=True
)


if __name__ == '__main__':

    img,lab=data_train[0]
    print(img,img.shape)
    print(lab,lab.shape)
