from torch import optim,nn
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data import MyDataset
from net import Unet
import os
import time

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print("used device now is",device)
weight_path = r'params/unet.pth'
data_dir = r'/Users/zane/Desktop/VOCdevkit/VOC2007' # 数据集使用VOC2007
out_dir = r'output/'


if __name__ == '__main__':
    dataset = MyDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    net = Unet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("load weight success")
    else:
        print("load weight failed")
    
    opt = optim.Adam(net.parameters(),lr=1e-3)
    
    loss_fn = nn.BCELoss()
    num_epoch = 10
    min_loss = 99999.0
    for epoch in range(num_epoch):
        start_time = time.time()

        for i,(img,seg_img) in enumerate(dataloader):
            img,seg_img = img.to(device),seg_img.to(device)
            
            out = net(img)
            train_loss = loss_fn(out,seg_img)
            
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            if (i+1)%10 == 0:
                print("epoch:{},step:{},loss======>{}".format(epoch,i+1,train_loss.item()))
                end_time = time.time()
                print("time cost:{:.2f}".format(end_time-start_time))
                _image = img[0]
                _segment_image = seg_img[0]
                _out_image = out[0]
                img = torch.stack([_image,_segment_image,_out_image],dim=0)
                save_image(img,out_dir+"out.png")
            
            
        if epoch % 3 == 0:
            if train_loss.item() < min_loss:
                min_loss = train_loss.item()
                torch.save(net.state_dict(),weight_path)
                print("save weight success")
            
            

    

