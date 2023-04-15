import os
import torchvision
import torch
import matplotlib.pyplot as plt

def save_image(x, dirname, epoch,fmt="png" ):
    # 生成图片
    path = os.path.join("output", dirname)
    if not os.path.exists(path):
        os.makedirs(path)

    torchvision.utils.save_image(x,path + '/image_epoch_{}.{}'.format(epoch,fmt))


def save_scatter(x, dirname, epoch,fmt="png"):
    # 生成散点图片
    path = os.path.join("output", dirname)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.figure(figsize=(10,10),dpi=100) #设置画布大小，像素
    plt.scatter(x[0],x[1]) #画散点图并指定图片标签
    plt.savefig(path + '/image_epoch_{}.{}'.format(epoch,fmt))#保存图片


    
def save_model(model, dirname, epoch):
    # 保存模型
    path = os.path.join("output", dirname)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path + '/MLPddpm-epoch{}.pth'.format(epoch))



def save_info(shape,num_steps,num_epoch,dirname,filename):
    # 保存超参数信息
    path = os.path.join("output", dirname)
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    with open(file_path,'w') as f:
        f.write('dataset: {}\n'.format(shape))
        f.write('num_steps: {}\n'.format(num_steps))
        f.write('num_epoch: {}\n'.format(num_epoch))

def save_log(log, path, filename):
    path = os.path.join("output", path)
    with open(os.path.join(path, filename), 'a') as f:
        f.write(log + '\n')