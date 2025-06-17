import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

ROOT=r'F:\NeuralNetworkModel\Pix2Pix_Facades\RUN_1\INFO.txt'

# 读取TXT文件并解析数据
data = []
with open(ROOT, 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        epoch = int(parts[0].split('=')[-1])
        lr = float(parts[1].split('=')[-1])
        d_loss = float(parts[2].split('=')[-1])
        g_loss = float(parts[3].split('=')[-1])
        cost_time = float(parts[4].split('=')[-1])
        data.append([epoch, lr, d_loss,g_loss, cost_time])

# 提取数据列
epochs = [d[0] for d in data]
lrs = [d[1] for d in data]
d_losses = [d[2] for d in data]
g_losses = [d[3] for d in data]
cost_times = [d[4] for d in data]




plt.plot(epochs, lrs, marker='o', label='Learning Rate',linewidth=0.5,markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate vs Epoch')
plt.grid(True)
plt.legend()
plt.show()




plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(d_losses, label="D")
plt.plot(g_losses, label="G")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()






