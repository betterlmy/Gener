from mmgen.apis import init_model, sample_unconditional_model

config_file = 'mmgeneration/configs/styleganv2/stylegan2_c2_8xb4-800kiters_lsun-church-256x256.py'
checkpoint_file = 'https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth'
device = "cpu"

model = init_model(config_file, checkpoint_file, device=device)

fake_img = sample_unconditional_model(model, num_samples=4)