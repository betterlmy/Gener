import os
import torchvision
import time
import torch

def save_image(x,dirname,epoch,now_time):
    # 生成图片
    path = os.path.join("output",dirname,now_time)
    if not os.path.exists(path):
        os.makedirs(path)
    
    torchvision.utils.save_image(x, path+'/image_epoch_{}.png'.format(epoch))

def save_model(model,dirname,epoch,now_time):
    # 保存模型
    path = os.path.join("output",dirname,now_time)
    if not os.path.exists(path):
        os.makedirs(path)
        
    torch.save(model.state_dict(),path+'/MLPddpm-epoch{}.pth'.format(epoch))

