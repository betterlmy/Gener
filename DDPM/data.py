import os
from torch.utils.data import Dataset
from util import keep_image_size_open
from torchvision import transforms



class MyDataset(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.path = path
        
        self.name = []
        files = os.listdir(os.path.join(path,'SegmentationClass'))
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                self.name.append(filename)
                
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        segname = self.name[index]
        segpath = os.path.join(self.path,'SegmentationClass',segname)
        imgpath = os.path.join(self.path,'JPEGImages',segname.replace('png','jpg'))
        seg_img = self.transform(keep_image_size_open(segpath))
        img = self.transform(keep_image_size_open(imgpath))
        return img,seg_img

if __name__  == '__main__':
    data = MyDataset('/Users/zane/Desktop/VOCdevkit/VOC2007') # 数据集使用VOC2007
    
    print(data[0][0].shape)
    print(data[0][1].shape)