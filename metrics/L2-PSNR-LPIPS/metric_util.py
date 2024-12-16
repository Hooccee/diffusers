import torch
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn
import lpips

dev = 'cuda'
to_tensor_transform = transforms.Compose([transforms.ToTensor()])

mse_loss = nn.MSELoss()
# loss_fn = lpips.LPIPS(net='alex',model_path="/home/lyw/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth").to('cuda')
# loss_fn = lpips.LPIPS(net='vgg',model_path="/home/lyw/.cache/torch/hub/checkpoints/vgg16-397923af.pth").to('cuda')
loss_fn = lpips.LPIPS(net='vgg').to('cuda')
def calculate_l2_difference(image1, image2, device = 'cuda'):
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)

    mse = mse_loss(image1, image2).item()
    return mse

def calculate_psnr(image1, image2, device = 'cuda'):
    max_value = 1.0
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)
    
    mse = mse_loss(image1, image2)
    psnr = 10 * torch.log10(max_value**2 / mse).item()
    return psnr




def calculate_lpips(image1, image2, device = 'cuda'):
    if isinstance(image1, Image.Image):
        image1_tensor = to_tensor_transform(image1)
        # 手动归一化
        mean = image1_tensor.mean(dim=(1, 2))
        std = image1_tensor.std(dim=(1, 2))
        image1_normalized = (image1_tensor - mean[:, None, None]) / std[:, None, None]
        image1 = image1_normalized.to(device)
    if isinstance(image2, Image.Image):
        image2_tensor = to_tensor_transform(image2)
        mean = image2_tensor.mean(dim=(1, 2))
        std = image2_tensor.std(dim=(1, 2))
        image2_normalized = (image2_tensor - mean[:, None, None]) / std[:, None, None]
        image2 = image2_normalized.to(device)
    # loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
    loss = loss_fn(image1, image2).item()
    return loss

def calculate_metrics(image1, image2, device = 'cuda', size=(512, 512)):
    if isinstance(image1, Image.Image):
        image1 = image1.resize(size)
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = image2.resize(size)
        image2 = to_tensor_transform(image2).to(device)

    l2 = calculate_l2_difference(image1, image2, device)
    psnr = calculate_psnr(image1, image2, device)
    lpips = calculate_lpips(image1, image2, device)
    return {"l2": l2, "psnr": psnr, "lpips": lpips}

