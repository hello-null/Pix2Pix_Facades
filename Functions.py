import numpy as np
import torch
from thop import profile
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import trange,tqdm
from PIL import Image
from sklearn.metrics import accuracy_score


'''
获取独热编码
'''
def get_one_hot(lab,n_class,low=0.05):
    '''
    :param lab: tensor 真实标签,torch.Size([64]),表示64个样本类别编号，例如[0,1,5,2,3,....,9]
    :param n_class: int 总类别个数
    :param low: 用于软化标签，不成立时的值，例如太阳从西边升起是不可能事件，概率应该为0，但软化后这里是设置为low
    :return: [n_sample,n_class]
    '''
    assert len(lab.shape)==1,'ERROR-get_one_hot'
    a=torch.zeros(size=(lab.shape[0],n_class),dtype=torch.float32)
    a=torch.fill_(a,low)
    for i in range(lab.shape[0]):
        a[i][int(lab[i].item())]=1-low
    return a


'''
通道交换
'''
def exchange_channles(img):
    '''
    :param img: tensor 图像 [C H W]
    :return np:[H W C] 0~1
    '''
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)
    return np.transpose(npimg, (1, 2, 0)) # H W C


'''
plt显示图像
'''
def show_img(img):
    '''
    先调用denormalize_tensor(img,[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    :param img: tensor 图像
    :return: None
    '''
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # H W C
    plt.show()



'''
逆标准化
'''
def denormalize_tensor(tensor, mean, std):
    """
    复原一个经过Z-score标准化处理的tensor。
    参数:
    tensor (torch.Tensor): 标准化后的tensor。
    mean (torch.Tensor or list or tuple): 每个通道的均值。
    std (torch.Tensor or list or tuple): 每个通道的标准差。
    返回:
    torch.Tensor: 复原后的tensor。
    """
    # 确保mean和std是tensor类型，并且它们的形状与tensor的通道数相匹配
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    # 广播机制应用于每个通道
    return tensor * std[:, None, None] + mean[:, None, None]




'''
生成类激活图'''
def generate_cam(feature_map, class_weights, class_idx, target_size=None, apply_relu=True):
    """
    生成类激活图（CAM）
    Args:
        feature_map (torch.Tensor): 卷积层输出，形状为 [C, H, W]（例如 [10, 8, 8]）
        class_weights (torch.Tensor): 全连接层的权重矩阵，形状为 [num_classes, C]
        class_idx (int): 目标类别索引
        target_size (tuple, optional): 上采样目标尺寸，例如 (224, 224)
        apply_relu (bool): 是否对 CAM 应用 ReLU（默认 True，过滤负贡献）
    Returns:
        cam (np.ndarray): 归一化后的热力图，形状为 [H, W] 或 target_size
    """
    # 检查输入合法性
    assert len(feature_map.shape) == 3, f"特征图形状应为 [C, H, W]，实际输入 {feature_map.shape}"
    C, H, W = feature_map.shape
    assert class_weights.dim() == 2, f"权重形状应为 [num_classes, C]，实际输入 {class_weights.shape}"
    assert class_idx < class_weights.shape[0], f"类别索引 {class_idx} 超出范围 [0, {class_weights.shape[0] - 1}]"
    # 提取目标类别的权重 [C]
    weights = class_weights[class_idx]
    # 计算 CAM：加权求和
    cam = torch.sum(weights[:, None, None] * feature_map, dim=0)  # [H, W]
    # 应用 ReLU（过滤负贡献）
    if apply_relu:
        cam = torch.relu(cam)
    # 归一化到 [0, 1]
    if cam.max() == cam.min():
        cam = torch.zeros_like(cam)  # 全零处理
    else:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # 转换为 CPU 上的 NumPy 数组
    cam = cam.cpu().detach().numpy()
    # 上采样到目标尺寸
    if target_size is not None:
        # 使用双线性插值
        cam = cam[np.newaxis, np.newaxis, ...]  # [1, 1, H, W]
        cam = torch.from_numpy(cam)
        cam_upsampled = interpolate(
            cam, size=target_size, mode="bilinear", align_corners=False
        )
        cam = cam_upsampled[0, 0].numpy()
    return cam



'''
显示类激活图
'''
def plot_cam(cam, original_image=None, alpha=0.5, cmap="jet"):
    """
    可视化 CAM，先调用generate_cam()
    Args:
        cam (np.ndarray): 热力图，形状 [H, W]
        original_image (np.ndarray, optional): 原始图像，形状 [H, W, 3]（值范围 [0,1]）
        alpha (float): 热力图透明度（默认 0.5）
        cmap (str): 热力图颜色图（默认 "jet"）
    """
    plt.figure(figsize=(10, 5))

    if original_image is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.axis("off")
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(original_image)
        plt.imshow(cam, cmap=cmap, alpha=alpha)
        plt.axis("off")
        plt.title("CAM Overlay")
    else:
        plt.imshow(cam, cmap=cmap)
        plt.axis("off")
        plt.title("CAM Heatmap")

    plt.show()


def model_performance_info(model,shape=(1,3,32,32)):
    '''
    显示模型性能信息
    :param model: 实例化后的模型
    :param shape: 模型输入形状，例如(1,3,32,32) (1,3,224,224)
    :return: FLOPs , 参数数量
    '''
    a=torch.randn(size=shape)
    flops, params = profile(model, inputs=(a,))
    return flops,params


def mode_avg_performance(net,loader_test,cnt_sample,PTH,n=10):
    '''
    获取模型运n次的平均性能
    :param net:网络，必须是已经实例化后的
    :param loader_test:测试集加载器，尽量是数据打乱的加载器
    :param cnt_sample:测试样本个数
    :param PTH:权重路径
    :param n:模型运行次数
    :return:
    '''
    model = net.cuda()
    model.load_state_dict(torch.load(PTH))
    model.eval()
    with torch.no_grad():
        arr_top1=[]
        arr_top5=[]
        for _ in trange(n):
            i_correct_top1 = 0
            i_correct_top5 = 0
            for data in loader_test:
                img, lab = data  # torch.Size([64, 3, 32, 32]) torch.Size([64])
                img = img.cuda()
                # ===================forward=====================
                output = model(img)  # torch.Size([64, N_CLASS])
                output = F.softmax(output, dim=1)
                # ===================Top-1 Top-5=====================
                output_cpu = output.detach().cpu()
                _, top5_indices = torch.topk(output_cpu, 5, dim=1)
                # 检查Top-1准确率 预测正确的条件是：最大概率对应的索引与真实标签相等
                pred_top1 = torch.argmax(output_cpu, dim=1)
                i_correct_top1 += (pred_top1 == lab.cpu()).sum().item()
                # 检查Top-5准确率 预测正确的条件是：真实标签在Top-5索引中
                i_correct_top5 += (top5_indices == lab.unsqueeze(1).expand_as(top5_indices)).any(dim=1).sum().item()
            # ==================print log========================
            total_samples = cnt_sample  # 获取验证集的总样本数
            accuracy_top1 = i_correct_top1 / total_samples
            accuracy_top5 = i_correct_top5 / total_samples
            arr_top1.append(accuracy_top1)
            arr_top5.append(accuracy_top5)
        return {
            'top-1':{
                'run':arr_top1,
                'avg':np.average(arr_top1)
            },
            'top-5':{
                'run': arr_top5,
                'avg': np.average(arr_top5)
            }
        }



def read_img_and_transform(path='F:/data/1.jpg',tra=None):
    '''
    读取图像并且转换
    :param path: 图像路径
    :param tra: from torchvision import transforms
    :return:
    '''
    image = Image.open(path)
    image_tensor = tra(image)
    return image_tensor


def keep_aspect_ratio(path, size=(448, 448),rgb=(0,0,0)):
    '''
    调整图像尺寸并保持其完整内容，同时确保输出图像为指定尺寸。
    会先按比例缩放图像，再将其居中粘贴到目标尺寸画布，避免变形和黑边残留。
    :param path:
    :param size:
    :return:
    '''
    img = Image.open(path)
    img.thumbnail(size)  # 保持比例缩放至不超过目标尺寸
    bg = Image.new('RGB', size, rgb)  # 创建目标尺寸画布
    bg.paste(img, ((size[0]-img.width)//2, (size[1]-img.height)//2))  # 居中粘贴
    return bg



def read_txt_lines(file_path):
    '''
    读取一个文本文件（txt）中的每一行内容，并将其返回为一个字符串列表
    :param file_path:
    :return:
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 去除每行末尾的换行符
    lines = [line.strip() for line in lines]
    return lines



def modify_image(image, a, b, c=1):
    """
    将张量中不等于a的值全部设置为b,等于a的值全部设置为c
    参数:
    image (torch.Tensor): 单通道图像，形状为[448, 448]
    a (int or float): 要排除的值
    b (int or float): 设置的新值
    c (int or float): 设置的新值
    返回:
    torch.Tensor: 修改后的图像
    """
    # 确保输入是 PyTorch 的 Tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch Tensor")
    # 确保输入张量是二维的（单通道图像）
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D tensor (single-channel image)")
    # 创建一个与输入张量相同形状的张量用于存储结果
    modified_image = torch.zeros_like(image, dtype=image.dtype)
    # 使用条件索引进行赋值
    modified_image[image == a] = c
    modified_image[image != a] = b
    return modified_image



def threshold_tensor(tensor, a, v1, v2):
    """
    将张量中小于a的值设置为v1，大于等于a的值设置为v2
    参数:
    tensor (torch.Tensor): 输入的张量，形状为[-1, 21, 448, 448]
    a (float): 阈值
    v1 (float): 小于a时的设置值
    v2 (float): 大于等于a时的设置值
    返回:
    torch.Tensor: 修改后的张量
    """
    # 使用布尔索引找到小于a的位置，并将这些位置的值设置为v1
    # 使用布尔索引找到大于等于a的位置，并将这些位置的值设置为v2
    modified_tensor = tensor.clone()  # 创建副本以避免修改原始张量
    modified_tensor[modified_tensor < a] = v1
    modified_tensor[modified_tensor >= a] = v2
    return modified_tensor


def calculate_segmentation_metrics(lab, pred_map,nc=21):
    """
    计算语义分割指标
    :param lab: 真实标签 [B, H, W] LongTensor (值域0~20)
    :param pred_map: 预测结果 [B, H, W] LongTensor (值域0~20)
    :return: (PA, mIoU, Dice)
    """
    # 展平张量
    lab_flat = lab.view(-1)
    pred_flat = pred_map.view(-1)
    # 计算Pixel Accuracy
    correct = (pred_flat == lab_flat).sum().item()
    total = lab_flat.numel()
    pa = correct / total if total > 0 else 0.0

    iou_list = []
    dice_list = []

    # 遍历所有21个类别
    for cls in range(nc):
        true_cls = (lab_flat == cls)
        pred_cls = (pred_flat == cls)

        tp = (true_cls & pred_cls).sum().item()
        fp = (pred_cls & ~true_cls).sum().item()
        fn = (true_cls & ~pred_cls).sum().item()

        # 计算IoU（处理边界情况）
        if tp + fp + fn == 0:
            iou = 1.0  # 无正负样本时视为完全匹配
        else:
            iou = tp / (tp + fp + fn + 1e-6)

        # 计算Dice（处理边界情况）
        if 2 * tp + fp + fn == 0:
            dice = 1.0
        else:
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

        iou_list.append(iou)
        dice_list.append(dice)

    # 计算平均IoU和Dice
    miou = np.sum(iou_list) / len(iou_list)
    dice = np.sum(dice_list) / len(dice_list)

    return pa, miou, dice


def dynamic_regularization(
        iteration,
        min_size, max_size,
        min_dropout, max_dropout,
        total_iterations,
        min_wd,max_wd,
):
    """
    根据当前的迭代次数，计算当前的图像缩放尺寸和dropout率。 动态正则化
    from paper《EfficientNetV2: Smaller Models and Faster Training》 Algorithm 1 Progressive learning with adaptive regularization.

    # 示例用法
    iteration = 100
    min_size = 64
    max_size = 128
    min_dropout = 0.1
    max_dropout = 0.5
    total_iterations = 100
    current_size, current_dropout = dynamic_regularization(iteration, min_size, max_size, min_dropout, max_dropout, total_iterations)
    print(f"Iteration {iteration}: Current size = {current_size:.2f}, Current dropout = {current_dropout:.2f}")

    参数:
    iteration (int): 当前的迭代次数，从1开始。
    min_size (int): 最小尺寸图像。
    max_size (int): 最大尺寸图像。
    min_dropout (float): 最小的dropout率。
    max_dropout (float): 最大的dropout率。
    total_iterations (int): 总迭代次数，用于计算阶段索引。

    返回:
    tuple: 当前的图像缩放尺寸和dropout率 (current_size, current_dropout)。
    """
    # 计算阶段索引，由于迭代次数从1开始，所以阶段索引为 iteration - 1
    stage_index = iteration - 1
    # 计算当前图像尺寸
    sz = min_size + (max_size - min_size) * (stage_index / (total_iterations - 1))
    # 计算当前dropout率
    dr = min_dropout + (max_dropout - min_dropout) * (stage_index / (total_iterations - 1))
    # 计算当前权重衰减
    wd = min_wd + (max_wd - min_wd) * (stage_index / (total_iterations - 1))

    return int(sz),dr,wd


def resize_image_with_proportion(img_path, target_min=256):
    """
    等比例缩放图像，确保最小边≥256像素，使用双线性插值
    :param img_path: 输入图像路径
    :param target_min: 目标最小边长（默认256）
    """
    try:
        with Image.open(img_path) as img:
            # 获取原始尺寸
            orig_width, orig_height = img.size
            min_side = min(orig_width, orig_height)

            # 计算缩放比例（确保最小边≥256）
            # scale = max(target_min / min_side, 1.0)  # 仅当原始尺寸<256时放大
            scale = target_min / min_side  # 仅当原始尺寸<256时放大

            # 计算新尺寸（保持宽高比）
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            new_size = (new_width, new_height)

            # 双线性插值缩放
            resized_img = img.resize(new_size, resample=Image.BILINEAR)
            return resized_img

    except Exception as e:
        print(f"处理 {img_path} 失败: {str(e)}")