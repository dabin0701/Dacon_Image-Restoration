# Dacon_Image-Restoration

## < 이미지 복원을 위한 학습 및 모델 설계 전략 >

1. 시간 절약을 위한 데이터 분할 학습 : 데이터를 분할 학습하여 변경한 모델구조와 파라미터가 잘 적용되는지 확인

2. Validation : 과적합 방지를 위해 분할 학습

4. 데이터 증강 & Resize 조정
   
5. 모델 구조 변경(Residual, Resnet)

6. 스케줄러 변경 : 학습 안정화를 위해 변경

7. 하이퍼 파라미터 조정

8. G&D 학습 불균형 조정 : G와 D 학습 주기를 다르게 하고, 자동 조정하게 하여 균형을 이루도록 조정

---

## < 모델별 특징 >

| model | 장단점 | 특징 |
|:------------:|:---------------------:|:---------:|
| Unet | 적은 메모리와 간단한 구조로 빠른 학습 가능 / 깊은 특징 학습에 한계 | 기본적인 이미지 복원 작업에 적합. 세부 디테일 복원은 부족할 수 있음. |
| Residual + Unet | Residual Block 추가로 학습 효율 증가, Dropout으로 과적합 방지 가능 / Resnet + Unet 보다 성능이 낮을 수 있음 | 깊이 있는 복원 가능, 복잡한 이미지에서도 안정적으로 동작 |
| Resnet + Unet | 깊은 네트워크 학습이 가능하며, Resnet skip connection으로 학습 안정성 증가 / 구조가 복잡하여 학습 시간이 길고 메모리 사용량 증가 | 복잡한 이미지 세부 정보를 잘 복원하지만 학습 비용이 높음 |

---
## DACON_image.ipynb

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import zipfile
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection for dimensionality adjustment
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)

# UNet with ResNet Blocks
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = ResNetBlock(3, 64)
        self.enc2 = ResNetBlock(64, 128)
        self.enc3 = ResNetBlock(128, 256)
        self.enc4 = ResNetBlock(256, 512)
        self.enc5 = ResNetBlock(512, 1024)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResNetBlock(1024 + 512, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ResNetBlock(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ResNetBlock(256 + 128, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ResNetBlock(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        e5 = self.enc5(nn.MaxPool2d(2)(e4))

        # Decoder
        d1 = self.dec1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return torch.sigmoid(self.final(d4))

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])
        input_image = cv2.imread(input_path)
        gt_image = cv2.imread(gt_path)
        # Resize input and GT images to 256x256
        input_image = cv2.resize(input_image, (256, 256))  # 리사이즈
        gt_image = cv2.resize(gt_image, (256, 256))        # 리사이즈
        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
        return (
            torch.tensor(input_image).permute(2, 0, 1).float() / 255.0,
            torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0
    
        )

generator = UNet().to(device)
discriminator = PatchGANDiscriminator().to(device)

generator = generator.to(device)
discriminator = discriminator.to(device)

adversarial_loss = nn.BCELoss()  
pixel_loss = nn.MSELoss()  

optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_G = optim.AdamW(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))


train_dataset = ImageDataset("/home/work/Dacon/dataset/train_input", "/home/work/Dacon/dataset/train_gt")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True) ####################################################배치사이즈 설정

epochs = 100
result_dir = "/home/work/Dacon/dataset/result"
os.makedirs(result_dir, exist_ok=True)
checkpoint_path = "/home/work/Dacon/dataset/checkpoint/checkpoint.pth"


for epoch in range(epochs):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0
    n_critic = 2  # 기본 값

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for input_images, gt_images in train_loader:
            input_images, gt_images = input_images.to(device), gt_images.to(device)

            # 1. Discriminator 업데이트
            optimizer_D.zero_grad()
            fake_images = generator(input_images).detach()
            pred_real = discriminator(gt_images)
            pred_fake = discriminator(fake_images)
            
            loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
            loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            running_loss_D += d_loss.item()

            # 2. `n_critic` 동적 조정
            if d_loss.item() > 1.5:  # D의 손실이 크면
                n_critic = 5  # D를 더 자주 업데이트
            elif d_loss.item() < 0.7:  # D의 손실이 너무 작으면
                n_critic = 1  # G를 더 자주 업데이트
            else:
                n_critic = 2  # 기본 값

            # 3. Generator 업데이트
            if pbar.n % n_critic == 0:  # `n_critic` 주기에 맞춰 Generator 업데이트
                optimizer_G.zero_grad()
                fake_images = generator(input_images)
                pred_fake = discriminator(fake_images)

                g_loss_adv = adversarial_loss(pred_fake, torch.ones_like(pred_fake))
                g_loss_pixel = pixel_loss(fake_images, gt_images)
                g_loss = g_loss_adv + 100 * g_loss_pixel
                g_loss.backward()
                optimizer_G.step()
                running_loss_G += g_loss.item()

            # Progress bar 업데이트
            pbar.set_postfix(generator_loss=running_loss_G / max(1, (pbar.n // n_critic + 1)),
                             discriminator_loss=running_loss_D / (pbar.n + 1))
            pbar.update(1)

    # Epoch 결과 출력
    print(f"Epoch [{epoch+1}/{epochs}] - Generator Loss: {running_loss_G / len(train_loader):.4f}, Discriminator Loss: {running_loss_D / len(train_loader):.4f}")

    test_input_dir = "test_input"
    output_dir = f"output_images_epoch_{epoch+1}"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for img_name in sorted(os.listdir(test_input_dir)):
            img_path = os.path.join(test_input_dir, img_name)
            img = cv2.imread(img_path)
            input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            output = generator(input_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
            output = output.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, img_name), output)

    zip_filename = os.path.join(result_dir, f"epoch_{epoch+1}.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_name in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, img_name), arcname=img_name)
    print(f"Epoch {epoch+1} results saved to {zip_filename}")

    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_path)

generator.train()  
discriminator.train()  
```

---

## Loss Graph

<p align="center">
<img src="https://github.com/dabin0701/Dacon_Image-Restoration/blob/f8ae4764d356de2df2292da81f5e49e5ccca782b/epochs%2058.png"  width="500" height="300"/>
</p>

## Test_input

<p align="center">
<img src="https://github.com/dabin0701/Dacon_Image-Restoration/blob/ad137c190c745c86e3dc74500730727e5a006c82/TEST_001%20(1).png"  width="500" height="300"/>
</p>

## Test_result

<p align="center">
<img src="https://github.com/dabin0701/Dacon_Image-Restoration/blob/ad137c190c745c86e3dc74500730727e5a006c82/TEST_001.png"  width="500" height="300"/>
</p>
