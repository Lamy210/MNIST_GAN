import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import csv

def save_generated_images(images, path):
    images = images.detach().cpu().numpy()
    n_images = images.shape[0]
    rows = 4
    cols = n_images // rows if n_images % rows == 0 else n_images // rows + 1
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    for i in range(n_images):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(images[i, 0], cmap='gray')
        axes[row, col].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(path, bbox_inches='tight')
    plt.close()




# 画像の保存先ディレクトリの作成
image_dir = 'generated_images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

fakeimage_dir = 'fake_images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

train_data = MNIST("./data", 
                   train=True, 
                   download=True, 
                   transform=transforms.ToTensor())

train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
print("train data size: ",len(train_data))
print("train iteration number: ",len(train_data)//BATCH_SIZE)


images, labels = next(iter(train_loader))
print("images_size:",images.size())
print("label:",labels)

image_numpy = images.detach().numpy().copy()
plt.imshow(image_numpy[0,0,:,:], cmap='gray')


class TwoConvBlock_2D(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding="same")
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.rl = nn.LeakyReLU()
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding="same")
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.rl(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.rl(x)
    return x

class Discriminator(nn.Module):   #識別器
  def __init__(self):
    super().__init__()
    self.conv1 = TwoConvBlock_2D(1,64)
    self.conv2 = TwoConvBlock_2D(64, 128)
    self.conv3 = TwoConvBlock_2D(128, 256)

    self.maxpool_2D = nn.AvgPool2d(2, stride = 2)

    self.l1 = nn.Linear(2304, 100)
    self.l2 = nn.Linear(100, 1)
    self.relu = nn.LeakyReLU()
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool_2D(x)
    x = self.conv2(x)
    x = self.maxpool_2D(x)
    x = self.conv3(x)
    x = self.maxpool_2D(x)
    x = x.view(-1, 2304)
    x = self.dropout(x)
    x = self.l1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.l2(x)
    x = torch.sigmoid(x)
    return x

class Generator(nn.Module):   #生成器
  def __init__(self):
    super().__init__()
    self.TCB1 = TwoConvBlock_2D(1,64)
    self.TCB2 = TwoConvBlock_2D(64,128)
    self.TCB3 = TwoConvBlock_2D(128,256)
    self.UC1 = nn.ConvTranspose2d(64, 64, kernel_size =2, stride = 2)
    self.UC2 = nn.ConvTranspose2d(128, 128, kernel_size =2, stride = 2)
    self.conv1 = nn.Conv2d(256, 1, kernel_size = 2, padding="same")
  
  def forward(self, x):
    x = self.TCB1(x)
    x = self.UC1(x)
    x = self.TCB2(x)
    x = self.UC2(x)
    x = self.TCB3(x)
    x = self.conv1(x)
    x = torch.sigmoid(x)
    return x
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_D = Discriminator().to(device)
model_G = Generator().to(device)

one_labels = torch.ones(BATCH_SIZE).reshape(BATCH_SIZE, 1).to(device)
zero_labels = torch.zeros(BATCH_SIZE).reshape(BATCH_SIZE, 1).to(device)

criterion = nn.BCELoss() 

optimizer_D = optim.Adam(model_D.parameters(), lr=0.00001)   #GANの学習率は低いことが多いです
optimizer_G = optim.Adam(model_G.parameters(), lr=0.00001)

epoch_num = 30
print_coef = 10
G_train_ratio = 2   #1epoch当たり何回生成器の学習を行うか
train_length = len(train_data)

history = {"train_loss_D": [], "train_loss_G": []}
n = 0
for epoch in range(epoch_num):
  train_loss_D = 0
  train_loss_G = 0

  model_D.train()
  model_G.train()
  for i, data in enumerate(train_loader):
    
    #識別器の学習（1）
    optimizer_D.zero_grad()
    inputs = data[0].to(device)
    outputs = model_D(inputs)
    loss_real = criterion(outputs, one_labels)   #本物のデータは1（本物）と判定してほしいので1のラベルを使用します

    #識別器の学習（2）
    noise = torch.randn((BATCH_SIZE, 1, 7, 7), dtype=torch.float32).to(device)   #ランダム配列の生成
    inputs_fake = model_G(noise)   #偽物データの生成
    outputs_fake = model_D(inputs_fake.detach())   #.detach()を使用して生成器が学習しないようにします
    loss_fake = criterion(outputs_fake, zero_labels)   #偽物のデータは0（偽物）と判定してほしいので0のラベルを使用します
    loss_D = loss_real + loss_fake   #識別器の学習（1）と（2）の損失を合算
    loss_D.backward()
    optimizer_D.step()

    #生成器の学習
    for j in range(G_train_ratio):
      optimizer_G.zero_grad()
      noise = torch.randn((BATCH_SIZE, 1, 7, 7), dtype=torch.float32).to(device)   #ランダム配列の生成
      inputs_fake = model_G(noise)
      outputs_fake = model_D(inputs_fake)
      loss_G = criterion(outputs_fake, one_labels)   #偽物のデータを1（本物）と判定する方向に学習したいので1のラベルを使用します
      loss_G.backward()
      optimizer_G.step()


      if j==0:
        image_filename = f'epoch_{epoch+1}_batch_{i+1}.png'
        image_path = os.path.join(image_dir, image_filename)
        save_generated_images(inputs_fake, image_path)


    #学習経過の保存
    train_loss_D += loss_D.item()
    train_loss_G += loss_G.item()
    n += 1
    history["train_loss_D"].append(loss_D.item())
    history["train_loss_G"].append(loss_G.item())

    if i % ((train_length//BATCH_SIZE)//print_coef) == (train_length//BATCH_SIZE)//print_coef - 1:
      print(f"epoch:{epoch+1}  index:{i+1}  train_loss_D:{train_loss_D/n:.10f}  train_loss_G:{train_loss_G/(n*BATCH_SIZE):.10f}")
      n = 0
      train_loss_D = 0
      train_loss_G = 0

print("finish training")

model_G.to("cpu")
with torch.no_grad():
  noise = torch.randn((BATCH_SIZE, 1, 7, 7), dtype=torch.float32)
  syn_image = model_G(noise)

  plt.figure()
  fig, ax = plt.subplots(2, BATCH_SIZE//2, figsize=(15,3))
  for i in range(BATCH_SIZE//2):
      ax[0, i].imshow(syn_image.detach().numpy().copy()[i,0,:,:], cmap='gray')
      ax[1, i].imshow(syn_image.detach().numpy().copy()[i + BATCH_SIZE//2,0,:,:], cmap='gray')
      ax[0, i].axis("off")
      ax[1, i].axis("off")
plt.savefig(os.path.join(image_dir, 'generated_images.png'), bbox_inches='tight')
plt.close()

# train_lossのCSV保存
train_loss_file = 'train_loss.csv'
with open(train_loss_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'train_loss_D', 'train_loss_G'])
    for epoch, (loss_D, loss_G) in enumerate(zip(history["train_loss_D"], history["train_loss_G"])):
        writer.writerow([epoch+1, loss_D, loss_G])