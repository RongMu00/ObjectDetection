import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# transform = transforms.Compose(
#     [transforms.ToTensor(),    # transform numpy image to tensor image
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
#for images, labels in dataiter:
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

class VGG16Net(nn.Module):

  def __init__(self):
    super(VGG16Net, self).__init__()
    self.input_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.hidden_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128, momentum=0.9),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    )
    self.hidden_layer2 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, momentum=0.9),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    )
    self.hidden_layer3 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512, momentum=0.9),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    )
    self.hidden_layer4 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512, momentum=0.9),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    )
    self.hidden_layer5 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512, momentum=0.9),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    )
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(512*2*2, 1*1*4096)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(1*1*4096, 1*1*4096)
    self.dropout2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(1*1*4096, 1*1*1000)

  def forward(self, x):
    out = self.input_layer(x)
    out = self.hidden_layer1(out)
    out = self.hidden_layer2(out)
    out = self.hidden_layer3(out)
    out = self.hidden_layer4(out)
    out = self.hidden_layer5(out)
    #print(out.shape)
    out = self.flatten(out)
    #print(out.shape)
    out = self.fc1(out)
    out = self.dropout1(out)
    out = self.fc2(out)
    out = self.dropout2(out)
    out = self.fc3(out)
    return out

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

batch_size = 256
momentum = 0.9
learning_rate = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = VGG16Net().to(device)
#model = VGG('VGG16').to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum)

print(model)

def train_one_epoch():
    running_loss = 0
    last_loss = 0
    for i, (images, label) in enumerate(trainloader):
        images = images.to(device)
        #print(images.shape)
        label = label.to(device)
        #print(label)
        # zero gradients for every batch
        optimizer.zero_grad()
        # predict on every batch
        outputs = model(images)
        # compute the loss and its gradients
        loss = criterion(outputs, label)
        loss.backward()
        #print("training loss: {}".format(loss.grad))
        # adjust learning weights
        optimizer.step()
        # gather data and report
        running_loss += loss.item()
        #print(running_loss)
        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print(' batch {} loss: {}'.format(i+1, last_loss))
            running_loss = 0
    # print(f'epoch {epoch+1}:', loss.item())
    return last_loss

epoches = 10
best_vloss = 1_000_000.

for epoch in range(epoches):
    print('EPOCH {}:'.format(epoch + 1))

    # make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch()

    running_loss_test = 0
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, label) in enumerate(testloader):
            images = images.to(device)
            label = label.to(device)
            outputs = model(images)
            loss = criterion(outputs, label)
            #print("loss: ", loss)
            #print('loss grad: {}'.format(loss.grad))
            running_loss_test += loss
            ind, y_pred = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (y_pred==label).sum().item()
        avg_loss_test = running_loss_test / (i+1)
        print('LOSS train {} test {}'.format(avg_loss, avg_loss_test))
        accuracy = correct / total
        print('Accuracy: ', accuracy)

        if avg_loss_test < best_vloss:
            best_vloss = avg_loss_test
            model_path = 'model_{}'.format(epoch+1)
            torch.save(model.state_dict(), model_path)

# model.eval()
# with torch.no_grad():
#   correct = 0
#   total = 0
#   for images, labels in testloader:
#     images = images.to(device)
#     labels = labels.to(device)
#     output = model(images)
#     ind, y_pred = torch.max(output.data, 1)
#     total += labels.size(0)
#     correct += (y_pred==labels).sum().item()
#   accuracy = correct/total
#   print('Accuracy: ', accuracy)