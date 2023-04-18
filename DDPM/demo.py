# DDPM 使用多层感知机的demo 数据集是s-curve
from sklearn.datasets import make_s_curve
import torch
import torch.nn as nn
import time
import sys
sys.path.append("/home/deng/dzn/l/Gener") # 添加当前路径到系统路径中
from utils import *
from tqdm import tqdm

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

## 生成数据集
s_curve, _ = make_s_curve(10 ** 4, noise=0.1)
s_curve = s_curve[:, [0, 2]] / 10.0

data = s_curve.T  # [2,10000]
dataset = torch.Tensor(s_curve).float()

now_time = time.strftime("%m-%d-%H_%M_%S", time.localtime(time.time()))

# 设置超参数
with torch.no_grad():
    num_steps = 500
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


def q_x(x_0, t, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt, alphas_bar_sqrt=alphas_bar_sqrt):
    """
    扩散过程,基于x0计算出任意时刻的采样x[t]
    """
    noise = torch.randn_like(x_0)  # 生成与x0同形状的随机张量(服从正态分布)
    return alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * noise, noise


class MLPDiffusion(nn.Module):
    # 基于MLP的多层感知机
    # 用于逆扩散过程，从给定的时间t和输入x[t]预测噪声分布的均值。

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

    def forward(self, x, t):
        """ x: [batch_size, 2] 表示x[t]
            t: [batch_size]
            这是逆扩散过程,也就是输入x[t]和time_embedding预测其噪声分布 均值和方差
        """
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x 


def diffusion_loss_fn(model, batch_data, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):

    batch_size = batch_data.shape[0]

    # 对一个batchsize的样本生成随机的时刻t
    t = torch.randint(0, n_steps, size=(batch_size // 2,))  # 先生成一个batchsize/2的随机数
    y = n_steps - 1 - t
    t = torch.cat([t, y], dim=0)  # 将t与n_steps-1`-t`拼接起来 目的是为了尽可能覆盖到更多的t shape= [batch_size]
    
    # 如果t<200,则t=t+200
    # t = torch.where(t < 200, t + 200, t)  # 将t中小于200的数加200
    t = torch.where(t > 1800, t - 200, t)
    # print(t)
    t = t.unsqueeze(-1)  # shape = [batch_size,1] 升维操作
    t = t.to(device)

    a = alphas_bar_sqrt[t].to(device)
    am1 = one_minus_alphas_bar_sqrt[t].to(device)

    z = torch.randn_like(batch_data).to(device)  # 生成与x0同形状的随机张量(服从正态分布) 
    x_t = batch_data * a + z * am1 # 对batch_data进行扩散 t时刻不相同
    # 128,512,512  512,512 t
    e = model(x_t, t.squeeze(-1)) # 送入模型，得到t时刻的不同维度随机噪声均值预测值
    '''
    x_t = batch_data * a + z * am1 是前向扩散的过程，用于计算t时刻的采样x[t]，其中z是服从标准正态分布的随机张量。
    模型接受x[t]和t作为输入，预测出x[t]时刻的噪声分布，即用于计算逆扩散的均值和方差。
    这个预测噪声分布需要和标准的正态分布进行比较，因为我们的目标是通过学习样本数据的分布来生成新的数据，而!!样本数据的分布假定为标准的正态分布!!。
    因此，我们需要尽可能地接近标准正态分布，从而提高生成数据的质量。

    实际上，我们假设逆扩散过程中的噪声分布是标准的正态分布，但是这个假设并不总是成立的。
    如果我们能够知道逆扩散过程中的噪声分布，那么我们可以使用更适合实际分布的损失函数来计算损失。
    不过，通常情况下我们无法得知实际分布，所以使用标准正态分布作为假设是一种比较常见和可行的方法。
    在DDPM中，较小的t时刻表示的是生成过程中的较早时刻，即生成数据的开始阶段。在这个阶段，我们可能无法得到与真实数据分布相匹配的噪声分布，因为我们只有初始采样x[0]和一些噪声，没有其他数据点可以提供分布信息。

    消融试验点1:
    在正向训练时，我们使用随机的噪声分布作为假设，即标准正态分布，来生成x[t]的预测值。
    虽然在较小的t时刻可能会与真实分布存在偏差，但是由于我们是通过学习样本数据的分布来生成新的数据，因此我们更关注的是整个数据集的分布，而不是单个数据点的分布。
    因此，即使在较小的t时刻存在一些误差，也不会对整个模型的训练效果造成太大影响。而在逆扩散过程中，我们是通过模型预测得到噪声分布的均值和方差，因此相对来说更加准确。
    通过忽略前20%的t取值，可能会减少对噪声分布的误差，从而提高模型的训练效果和收敛速度。
    不过，这种方法的有效性需要根据具体情况来判断，因为取哪个t的区间可能会受到数据集的特点和模型的复杂度等因素的影响。
    从算法创新的角度来看，这种想法可能算是一种探索性的尝试。
    因为DDPM是一个相对较新的生成模型，仍有很多可以探索的方向。在实践中，通过不断地尝试和实验，我们可以发现一些新的技巧和方法，从而提高模型的性能和效率。
    '''
    return (z - e).pow(2).mean()


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t]).to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t] 
    eps_theta = model(x, t) # 得到噪声分布的均值
    mean = (1 / (1 - betas[t])) * (x - coeff * eps_theta) # 
    
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt() # sigma_t  imporoved DDPM 训练得到了sigma_t
    sample = mean + sigma_t * z # 重参数化技巧 
    return (sample)


def p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt):
    """从x[T]中恢复x[T-1],x[T-2]...x[0]"""
    cur_x = torch.randn(shape).to(device) # X[T]
    x_seq = [cur_x] # x_seq 是一个list,里面存放的是x[T],x[T-1],x[T-2]...x[0]
    for t in reversed(range(num_steps)):
        cur_x = p_sample(model, cur_x, t, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)

    return x_seq




def train():
    batch_size = 128

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoch = 20000
    model = MLPDiffusion(num_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_info(dataset.shape,num_steps,num_epoch,f"MLPddpm/{now_time}",'train_log.txt')
    save_log('train start at '+time.strftime("%m.%d,%H:%M:%S", time.localtime(time.time())),f"MLPddpm/{now_time}",'train_log.txt')

    for epoch in tqdm(range(num_epoch)):
        lossList = []
        lastLoss = 0
        start_time = time.time()
        for idx, data in enumerate(dataloader):
            data = data.to(device)
            loss = diffusion_loss_fn(model, data, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            lossList.append(loss.item())
 
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 用于解决梯度爆炸的问题,也就是梯度爆炸的解决方法.
            optimizer.step()
            end_time = time.time()
            meanLoss = sum(lossList)/len(lossList)
        log = 'epoch:{}/{},meanloss:{:.3f},time:{:.2f}'.format(epoch, num_epoch, meanLoss, end_time - start_time)
        
        if (meanLoss - lastLoss) < 0.0005:
            break
        lastLoss = meanLoss
        # 采样展示
        if epoch % 50 == 0:
            x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)
            cur_x = x_seq[num_steps].detach() 
            save_scatter(cur_x.T.cpu(), dirname=f"MLPddpm/{now_time}/Pic", epoch=epoch)
            tqdm.write(log)
            save_log(log,f"MLPddpm/{now_time}",'train_log.txt')

        if epoch % 500 == 0:
            save_model(model, dirname=f"MLPddpm/{now_time}/Model", epoch=epoch)
            
    save_log('train finished at '+time.strftime("%m.%d,%H:%M:%S", time.localtime(time.time())),f"MLPddpm/{now_time}",'train_log.txt')

if __name__ == "__main__":
    train()