# flake8: noqa
from mmedit.apis import init_model, sample_unconditional_model
from torchvision.utils import save_image

config_file = 'mmediting/configs/styleganv2/stylegan2_c2_8xb4-800kiters_lsun-church-256x256.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth'

device = "cuda:5"
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_unconditional_model(model, 40)
save_image(fake_imgs, "output/fake_imgs.png")
print("Done")

