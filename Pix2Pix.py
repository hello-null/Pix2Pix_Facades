import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
from torchsummary import summary


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        '''
        根据pix2pix论文中的描述，对于facades数据集（CMP Facades），输入图像的分辨率是​​256×256像素​​。
        :param in_size: 输入通道数
        :param out_size: 输出通道数
        :param normalize: 是否执行BatchNorm2d
        :param dropout: 随机丢弃率
        '''
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)





class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        '''
        :param in_size: 输入通道数
        :param out_size: 输出通道数
        :param dropout: 随机丢弃率
        '''
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        out = self.model(x)
        out = torch.cat((out, skip_input), 1)
        return out


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        '''
        :param in_channels: 输入通道数，image annotation的通道
        :param out_channels: 生成图像的通道数
        '''
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False) # 64*128*128
        self.down2 = UNetDown(64, 128) # 128*64*64
        self.down3 = UNetDown(128, 256) # 256*32*32
        self.down4 = UNetDown(256, 512, dropout=0.5) # 512*16*16
        self.down5 = UNetDown(512, 512, dropout=0.5) # 512*8*8
        self.down6 = UNetDown(512, 512, dropout=0.5) # 512*4*4
        self.down7 = UNetDown(512, 512, dropout=0.5) # 512*2*2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5) # 512*1*1

        self.up1 = UNetUp(512, 512, dropout=0.5) # 1024*2*2
        self.up2 = UNetUp(1024, 512, dropout=0.5) # 1024*4*4
        self.up3 = UNetUp(1024, 512, dropout=0.5) # 1024*8*8
        self.up4 = UNetUp(1024, 512, dropout=0.5) # 1024*16*16
        self.up5 = UNetUp(1024, 256) # 512*32*32
        self.up6 = UNetUp(512, 128) # 256*64*64
        self.up7 = UNetUp(256, 64) # 128*128*128

        # 修复：最后一层不使用UNetUp而是直接上采样
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 修复：添加1×1卷积输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, anno):
        '''
        :param x: 输入x为源 域图像（如边缘图、语义分割图、灰度图等）作为条件输入，
                  通过U-Net架构将其转换为目标域图像（如上色图、真实照片等）
        :return:
        '''
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(anno)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)  # 输入256, 输出64, 拼接后128 * 128 * 128
        # 修复：最后一层上采样 (128 * 128 * 128 -> 64 * 256 * 256)
        u8 = self.final_up(u7)
        # 输出层 (64 * 256 * 256 -> out_channels*256 * 256)
        out = self.final_conv(u8)
        return self.final_act(out)



##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, use_sigmoid=True):
        """
            Pix2Pix的判别器（PatchGAN）实现

            根据论文描述：
            - 使用70×70 PatchGAN架构
            - 输入为生成图像或真实图像与条件图像的拼接
            - 输出为N×N的特征图，每个元素对应输入图像的一个局部区域（感受野70×70）

            架构细节（论文附录6.1.2）：
            C64-C128-C256-C512（每个Ck表示4×4卷积、批归一化、LeakyReLU的模块）

            参数：
            - in_channels: 输入通道数（条件图像+目标图像的通道总和）
            - use_sigmoid: 是否在输出层添加sigmoid激活（通常BCEWithLogitsLoss中不需要）
        """
        super(Discriminator, self).__init__()
        # 创建4个卷积模块
        self.layers = nn.Sequential(
            # 第1层：无批归一化（论文要求）
            # 输入：in_channels x 256 x 256, 输出：64 x 128 x 128
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 第2层：128通道
            # 输入：64 x 128 x 128, 输出：128 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 第3层：256通道
            # 输入：128 x 64 x 64, 输出：256 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 第4层：512通道（步长改为1以获得更大的感受野）
            # 输入：256 x 32 x 32, 输出：512 x 31 x 31
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层：1通道（每个像素对应一个70×70的patch）
            # 输入：512 x 31 x 31, 输出：1 x 30 x 30
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
        # 可选的sigmoid激活
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, img, condition):
        """
        输入是通道维度的拼接结果：torch.cat([源域图像, 目标域图像], dim=1)，形成6通道张量（假设RGB图像）
        参数：
        - img: 待判别图像：3通道（RGB照片）（生成或真实图像） 3*256*256
        - condition: 条件图像：（灰度图、边缘图、语义标签图） 3*256*256
        返回：
        - 判别器输出一个​​N×N的矩阵​​（如30×30），每个元素对应输入图像的一个局部区域（如70×70像素块）的真假概率，而非单值判断
        """
        # 拼接条件图像和待判别图像（沿通道维度）
        x = torch.cat([img,condition], dim=1)
        # 通过卷积层
        x = self.layers(x)
        # 应用sigmoid（如果启用）
        if self.use_sigmoid:
            return self.sigmoid(x)
        return x


if __name__ == '__main__':
    d=Discriminator(6)
    a=torch.randn((64,3,256,256))
    b=torch.randn((64,3,256,256))
    o=d(a,b)
    print(o,o.shape)

    g=GeneratorUNet(3,3)
    a=torch.randn((64,3,256,256))
    o=g(a)
    print(o,o.shape)