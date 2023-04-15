# 整理的DDPM代码

import PIL
import torch
import torch.nn as nn
import time
import sys
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader

sys.path.append("/home/deng/dzn/l/Gener") # 添加当前路径到系统路径中
from utils import *

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# 设置超参数
with torch.no_grad():
    num_steps = 500
    batch_size = 128
    num_epoch = 2000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas)
    betas = betas * (5e-3 - 1e-5) + 1e-5  # 将betas张量中的每个值缩放到一个范围，这个范围是从1e-5到5e-3。

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)  # 计算累乘Cumulative product
    alphas_prod_p = torch.cat([torch.tensor([1.0]), alphas_prod[:-1]], 0)  # p=previous

    alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
    betas = betas.to(device)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod).to(device)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)

## 生成数据集
img = PIL.Image.open("/home/deng/dzn/l/Gener/DDPM/onePic/000007.jpg")
img = img.convert("L")  # 转换为灰度图
img = img.resize((512, 512))      # 调整图像大小
# img.save("灰度图.png")
img = np.array(img, dtype=np.float32) / 255.0    # 将像素值缩放到0到1之间的范围内
# img = img.transpose((2, 0, 1))  # 转换为通道在前的形式，shape为[C, H, W]
img = torch.from_numpy(img)     # 将图像数据转换为tensor
dataset = data.TensorDataset(img.repeat(batch_size, 1, 1))

now_time = time.strftime("%m-%d-%H_%M_%S", time.localtime(time.time()))

def q_x(x_0, t, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt, alphas_bar_sqrt=alphas_bar_sqrt):
    """
    扩散过程,基于x0计算出任意时刻的采样x[t]
    """
    noise = torch.randn_like(x_0)  # 生成与x0同形状的随机张量(服从正态分布)
    return alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * noise, noise


class MLPDiffusion(nn.Module):
    # 基于MLP的多层感知机

    def __init__(self, n_steps, num_units=128) -> None:
        super().__init__()

        self.linears = nn.ModuleList(
            [
                
                nn.Linear(2, num_units),
                nn.ReLU(),
                
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                
                nn.Linear(num_units, 2)
            ]
        )

        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x_0, t):
        x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t进行采样计算"""

    batch_size = x_0.shape[0]

    # 对一个batchsize的样本生成随机的时刻t
    t = torch.randint(0, n_steps, size=(batch_size // 2,))  # 先生成一个batchsize/2的随机数
    y = n_steps - 1 - t
    t = torch.cat([t, y], dim=0)  # 将t与n_steps-1-t拼接起来 目的是为了尽可能覆盖到更多的t shape= [batch_size]
    t = t.unsqueeze(-1)  # shape = [batch_size,1] 升维操作
    t = t.to(device)

    # 采样两个batch_size维的超参数
    a = alphas_bar_sqrt[t].squeeze(-1).to(device)
    am1 = one_minus_alphas_bar_sqrt[t].to(device)

    e = torch.randn_like(x_0).to(device)  # 生成与x0同形状的随机张量(服从正态分布) batch_size,512,512
    x = x_0 * a
    x +=  e * am1

    out = model(x, t.squeeze(-1))

    return (e - out).pow(2).mean()


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t]).to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t])) * (x - coeff * eps_theta)
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)


def p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt):
    """从x[T]中恢复x[T-1],x[T-2]...x[0]"""
    cur_x = torch.randn(shape).to(device) # X[T]
    x_seq = [cur_x] # x_seq 是一个list,里面存放的是x[T],x[T-1],x[T-2]...x[0]
    for i in reversed(range(num_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)

    return x_seq




def train():
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MLPDiffusion(num_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_info(img.shape,num_steps,num_epoch,f"MLPddpm/{now_time}",'train_log.txt')
    
    for epoch in range(num_epoch):
        start_time = time.time()
        for idx, data in enumerate(dataloader):
            data = data[0].to(device)
            loss = diffusion_loss_fn(model, data, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 用于解决梯度爆炸的问题,也就是梯度爆炸的解决方法.
            optimizer.step()
            end_time = time.time()
        log = 'epoch:{}/{},loss:{:.3f},time:{:.2f}'.format(epoch, num_epoch, loss.item(), end_time - start_time)
        print(log)
        save_log(log,f"MLPddpm/{now_time}",'train_log.txt')

        if epoch % 50 == 0:
            x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)
            cur_x = x_seq[num_steps].detach() 
            save_scatter(cur_x.T.cpu(), dirname=f"MLPddpm/{now_time}/Pic", epoch=epoch)
        if epoch % 500 == 0:
            save_model(model, dirname=f"MLPddpm/{now_time}/Model", epoch=epoch)
    save_log('finished'+time.strftime("%m-%d-%H_%M_%S", time.localtime(time.time())),f"MLPddpm/{now_time}",'train_log.txt')

if __name__ == "__main__":
    train()