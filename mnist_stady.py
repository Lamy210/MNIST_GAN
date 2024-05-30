import torch
import numpy as np
from PIL import Image
import os
from PIL import Image
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ハイパーパラメータ
batch_size = 100
lr = 0.0002
n_epoch = 20


save_dir = './fake_images'
save_format = 'png'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# MNISTデータセットのダウンロード
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 生成器
G = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh()
).to(device)

# 識別器
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,100),
    nn.Sigmoid()
).to(device)

# 損失関数と最適化手法の定義
criterion = nn.BCELoss()
G_solver = torch.optim.Adam(G.parameters(), lr=lr)
D_solver = torch.optim.Adam(D.parameters(), lr=lr)

# 学習ループ
for epoch in range(n_epoch):
    for i, (images, _) in enumerate(train_loader):

        # 実データをVectorに変換してGPUへ転送
        images = images.reshape(batch_size, 784).to(device)
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size,1).to(device)

        # 識別器の訓練
        outputs = D(images)
        D_loss_real = criterion(outputs, real_label)
        real_score = outputs
        
        z = torch.randn(batch_size, 100, device=device)
        fake_images = G(z)
        outputs = D(fake_images)
        D_loss_fake = criterion(outputs, fake_label)
        fake_score = outputs
        
        D_loss = criterion(outputs, real_label)
        D.zero_grad()
        D_loss.backward()
        D_solver.step()

        # 生成器の訓練
        z = torch.randn(batch_size, 100, device=device)


        print(f'********{i}********')
        filename = f'epoch_{epoch+1}_step_{i+1}.{save_format}'
        print("*******************")
        fake_images_np = fake_images.cpu().detach().numpy()

        # ピクセル値を0-255の範囲に正規化
        fake_images_np = (fake_images_np * 255).astype(np.uint8)

        Image.fromarray(fake_images_np).save(os.path.join(save_dir, filename))
        with open(os.path.join(save_dir, filename.replace(save_format, 'txt')), 'w') as f:
            f.write(str(real_score.data.mean().item()))

        fake_images = G(z)
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)

        outputs = D(fake_images)

        
        G_loss = criterion(outputs, real_label)
        G.zero_grad()
        G_loss.backward(retain_graph=True)
        G_solver.step()
        
        # ログ
        if (i+1) % 200 == 0:
                print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f, ' 
                            'D(x): %.2f, D(G(z)): %.2f'
                            %(epoch+1, n_epoch, i+1, len(train_loader), 
                                D_loss.item(), G_loss.item(),
                                real_score.data.mean(), fake_score.data.mean()))
    # モデルの保存
    torch.save(G.state_dict(), './G.pth')
    torch.save(D.state_dict(), './D.pth')
    print(f'Epoch {epoch+1} finished')
print('Finished Training')
