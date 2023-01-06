import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pylab as plt
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 100
batch_size = 100

sample_dir = "gan_sample"
if not os.path.exists(sample_dir):
  os.makedirs(sample_dir)

transform = transforms.Compose([
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.ToTensor()
            ])

mnist = torchvision.datasets.MNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

data_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

def denorm(x):
  out = (x+1)/2
  return out.clamp(0,1) # normalize the data into the range between 0 and 1.

def reset_grad():
  d_optimizer.zero_grad()
  g_optimizer.zero_grad()

# define the discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)

# define the generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Sigmoid()
)

D = D.to(device)
G = G.to(device)

criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

# Start training GAN
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # print(images.size())
        images = images.reshape(batch_size, -1).to(device)
        # print(images.shape)
        # Create the labels which are later used as input
        real_labels = torch.ones(batch_size, 1).to(device)
        # print(real_labels)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        ### 1. Discrimnator Training starts ###
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        # print(images.size())
        outputs = D(images)
        # print(outputs)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize

        d_loss = d_loss_real + d_loss_fake  # combine loss of discriminator
        reset_grad()  # reset the gradient
        d_loss.backward()  # back propagate the loss
        d_optimizer.step()

        ### 2. Generator Training starts ###
        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print out your loss, score for both out your discriminator and generator per epoch
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

    # Save real images
    if (epoch + 1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

