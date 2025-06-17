import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import time
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
plt.switch_backend('tkagg')
from torchvision.utils import make_grid
from torch.optim import Adam
from torchsummary import summary
from Pix2Pix import Discriminator,GeneratorUNet,weights_init
from datasets_Facades import loader_train,loader_test,data_train,data_test,BS



'''
《Image-to-Image Translation with Conditional Adversarial Networks》
'''


ROOT=r'F:\NeuralNetworkModel\Pix2Pix_Facades\RUN_1'
# Number of training epochs
NUM_EPOCHS = 300
# Learning rate for optimizers
LR = 2e-4
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')







if __name__ == '__main__':

    netG = GeneratorUNet(in_channels=3,out_channels=3).to(device)
    netG.load_state_dict(torch.load(r'F:\NeuralNetworkModel\Pix2Pix_Facades\RUN_1\G\dict_epoch_72.pth'))
    # netG.apply(weights_init)

    netD = Discriminator(in_channels=6,use_sigmoid=True).to(device)
    netD.load_state_dict(torch.load(r'F:\NeuralNetworkModel\Pix2Pix_Facades\RUN_1\D\dict_epoch_72.pth'))
    # netD.apply(weights_init)

    criterion = nn.MSELoss().to(device)
    criterion_pixelwise = nn.L1Loss().to(device)

    optimizerD = Adam(netD.parameters(), lr=LR, betas=(beta1, 0.999))
    optimizerG = Adam(netG.parameters(), lr=LR, betas=(beta1, 0.999))

    for epoch in range(73,NUM_EPOCHS+1):
        tm_start = time.time()

        # 训练
        for image,annotation in tqdm(loader_train,desc='epoch={}/{}'.format(epoch,NUM_EPOCHS)):
            # torch.Size([64, 3, 256, 256])  torch.Size([64, 3, 256, 256])

            # 真实/假标签（PatchGAN输出为30x30）
            gt = torch.ones((BS, 1, 30, 30), requires_grad=False).to(device)
            fa = torch.zeros((BS, 1, 30, 30), requires_grad=False).to(device)

            # TODO 向鉴别器展示一个真实的数据样本，z告诉它该样本的分类应该是1.0。
            optimizerD.zero_grad()
            real_img = image.to(device)
            real_anno=annotation.to(device)
            output = netD(
                real_img,
                real_anno
            )
            loss1 = criterion(output, gt)

            # TODO 向鉴别器显示一个生成器的输出，告诉它该样本的分类应该是0.0。
            fake = netG(real_anno) # torch.Size([32, 1, 64, 64])
            output = netD(
                fake.detach(),
                real_anno
            )
            loss2 = criterion(output, fa)
            loss_D=(loss1+loss2)*0.5
            loss_D.backward()
            optimizerD.step()

            # TODO 向鉴别器显示一个生成器的输出，告诉生成器结果应该是1.0。
            optimizerG.zero_grad()
            output = netD(
                fake.detach(),
                real_anno
            )
            loss4 = criterion(output, gt)
            loss5 = criterion_pixelwise(fake, real_img)
            loss_G = loss4 + 100 * loss5
            loss_G.backward()
            optimizerG.step()

        # 打印日志
        tm_end = time.time()
        str_train = 'epoch={} lr={:.8f} D_loss={:.3f} G_loss={:.3f} cost_time_m={:.3f}\n'.format(
            epoch,
            optimizerD.param_groups[0]['lr'],
            loss_D.item(),
            loss_G.item(),
            (tm_end - tm_start) / 60,
        )
        print(str_train, end='')

        # 保存日志+模型
        with open(ROOT+"\\INFO.txt", "a", encoding="utf-8") as f:
            f.write(str_train)  # 格式化字符串
        torch.save(netD.state_dict(), ROOT+'\\D\\dict_epoch_{}.pth'.format(epoch))
        torch.save(netG.state_dict(), ROOT+'\\G\\dict_epoch_{}.pth'.format(epoch))

        # 保存虚假图像
        with torch.no_grad():
            i=1
            stake=None
            for image, annotation in loader_test:
                fakes = netG(annotation.to(device)).detach().cpu()
                if stake==None:
                    stake=torch.cat([image,fakes],dim=0)
                else:
                    stake=torch.cat([stake,image,fakes],dim=0)
                i+=1
                if i>=3:
                    break
        grid = make_grid(
            stake*0.5+0.5,
            nrow=BS,
        )
        plt.figure(figsize=(20,27))
        plt.imshow(grid.permute(1,2,0).numpy())
        plt.savefig(
            ROOT+'\\fake_imgs\\epoch_{}.jpg'.format(epoch),
            bbox_inches='tight',
            pad_inches=0.1)
        plt.close()

