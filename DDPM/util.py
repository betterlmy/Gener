from PIL import Image

def keep_image_size_open(path,size=(256,256)):
    img = Image.open(path)
    tmp = max(img.size)
    mask = Image.new('RGB',(tmp,tmp),(0,0,0))
    mask.paste(img,(0,0))
    mask = mask.resize(size)
    return mask