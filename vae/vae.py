import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time,os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


now_time = time.strftime("%m-%d-%H_%M_%S", time.localtime(time.time()))
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

# 定义变分自编码器模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2) # 输出均值和方差 所以最后一个全连接层共40个神经元 前20个是均值 后20个是方差
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid() # 输出在[0,1]范围内的概率
        )

    # 从潜在变量z中采样,重参数?
    def sample_z(self, mu, log_var):
        eps = torch.randn(mu.size(0), self.latent_dim).to(device) # 从标准高斯分布中采样一个eps shape=[128,20] 即采样128个标准高斯分布的样本
        sigma = torch.exp(log_var / 2)
        return mu + sigma * eps # z = mu + sigma * eps
        '''
        在 VAE 中，我们使用编码器网络对输入数据进行编码，得到均值向量 mu 和对数方差向量 log_var，
        其中 log_var 是一个向量，其每个元素对应一个潜在变量的对数方差。
        为了得到标准差矩阵 sigma，我们需要对 log_var 进行指数操作，并将结果除以 2，即 sigma = exp(log_var / 2)。
        
        在VAE中，编码器网络的输出是隐变量的均值和方差，其中方差必须为正，但是在神经网络中，我们通常使用线性层进行预测，这样就有可能输出负数，
        为了使方差始终为正，我们可以对其进行指数操作，从而得到正数的方差。
        同时，在进行解码器的重参数化时，需要使用到方差的平方根，为了简化计算，我们可以将标准差（方差的平方根）直接计算出来，然后用它来重参数化潜在向量 z。
        因此，我们需要将方差除以 2 取指数，从而得到标准差。(除以2就是开根号)
        具体来说，我们使用标准正态分布 N(0,1) 中的样本进行重参数化，先从该分布中采样一个样本 eps，然后计算出潜在变量 z = mu + sigma * eps，其中 mu 是隐变量的均值，sigma 是标准差。这样，我们就可以使用反向传播算法对整个 VAE 模型进行训练。
        '''
    # 编码函数
    def encode(self, x):
        h = self.encoder(x)
        # print(h.shape) # 128,40 128个样本 每个样本40个特征(均值和方差)
        mu, log_var = torch.chunk(h, 2, dim=1) # 将输出分成均值和方差两部分
        return mu, log_var

    # 解码函数
    def decode(self, z):
        return self.decoder(z)

    # 前向传播
    def forward(self, x):
        mu, log_var = self.encode(x) # 编码
        z = self.sample_z(mu, log_var) # 重参数化进行采样 # 128,20
        x_hat = self.decode(z)# 128,784
        return x_hat, mu, log_var

# 定义损失函数
def loss_fn(x, x_hat, mu, log_var):
    """
    VAE的损失函数由两部分构成,
    一个是重构误差(Reconstruction Loss),用来衡量生成的样本和原始样本之间的差距,
    另一个则是KL散度(Kullback-Leibler Divergence),用来衡量模型生成的潜在变量分布于预设分布之间的差距.
    """
    
    # 重构误差(图片相似度)
    recon_loss = F.binary_cross_entropy(x_hat,x,reduction='sum')

    # KL散度 正则项
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss , kl_loss

# 加载数据
transform = transforms.Compose([
    transforms.ToTensor(), # 转tensor
    transforms.Normalize((0.1307,), (0.3081,)), #归一化
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# 定义模型和优化器
model = VAE(input_dim=784, latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
def train(num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        model.train()
        train_recon_loss = 0
        train_kl_loss = 0
        train_loss = 0
        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(x)
            recon_loss,kl_loss = loss_fn(x, x_hat, mu, log_var)
            loss = recon_loss + kl_loss
            loss.backward()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            
            optimizer.step()
        print('Loss: {:.2f}'.format(train_loss / len(train_loader.dataset)))
        print('Recon Loss: {:.2f}'.format(train_recon_loss / len(train_loader.dataset)))
        print('KL Loss: {:.2f}'.format(train_kl_loss / len(train_loader.dataset)))
        
        
        if (epoch+1) % 5 == 0 or epoch == 1:
            generate_image(model,epoch+1)# 生成临时图片 
            torch.save(model.state_dict(), 'vae/vae_params.pt')# 保存模型


def generate_image(model,epoch):
    # 生成图片
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, model.latent_dim).to(device)
    out = model.decode(z).view(-1, 1, 28, 28).cpu()
    
    path = 'vae/output/'+now_time
    if not os.path.exists(path):
        os.makedirs(path)
    save_image(out, path+'/image_epoch_{}.png'.format(epoch))


def main():
    num_epochs = 100
    train(num_epochs)
    
    # 创建新的VAE模型，并加载模型参数
    # new_vae = VAE().to(device)
    # new_vae.load_state_dict(torch.load('vae_params.pt'))
    # new_vae.eval()
        




if __name__ == "__main__":
    main()
    