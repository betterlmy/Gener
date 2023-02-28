from torch import optim,nn
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data import MyDataset
from net import Unet
import os
import time


if __name__ == '__main__':
        
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("used device now is",device)
    weight_path = r'params'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    weight_path = os.path.join(weight_path,"unet.pth")
    out_dir = r'output/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    data_dir = r'/root/lmy/data/VOCdevkit/VOC2007/' # 数据集使用VOC2007
    dataset = MyDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=28, shuffle=True)
    
    net = Unet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("load weight success")
    else:
        print("load weight failed")
    
    opt = optim.Adam(net.parameters(),lr=1e-3)
    
    loss_fn = nn.BCELoss()
    num_epoch = 100
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
                print("epoch:{},step:{},loss======>{:.4f}".format(epoch,i+1,train_loss.item()))

                _image = img[0]
                _segment_image = seg_img[0]
                _out_image = out[0]
                img = torch.stack([_image,_segment_image,_out_image],dim=0)
                save_image(img,out_dir+f"out{epoch+1}.png")
            
        end_time = time.time()
        print("time cost:{:.2f}/epoch".format(end_time-start_time))
        if epoch % 3 == 0:
            if train_loss.item() < min_loss:
                min_loss = train_loss.item()
                torch.save(net.state_dict(),weight_path)
                print("save weight success")
            
            

    

