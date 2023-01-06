import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import copy

epsilons = [0, .05, .1, .15, .2, .25, .3]
use_cuda = True
pretrained_model = 'model_9'
batch_size = 256
CIFAR10_RESNET_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_RESNET_STD = (0.2023, 0.1994, 0.2010)

transform_test = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
# MNIST Test dataset and dataloader declaration
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# get some random training images
dataiter = iter(testloader)
images, labels = dataiter.next()
#for images, labels in dataiter:
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# model
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

# Initialize the network
model = VGG16Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def fgsm_attack(image, epsilon, data_grad, image_to_perturb):
     original_image = copy.copy(image)
     sign_data_grad = data_grad.sign()
     perturbed_image = torch.as_tensor(image)
     # print(image.shape)
     if len(image) < batch_size:
         for i in range(len(image)):
             if i in image_to_perturb:
                 image[i] = image[i] - epsilon * sign_data_grad[i]
                 perturbed_image[i] = torch.clamp(image[i], min=0, max=1)
     else:
         for i in range(batch_size):
             if i in image_to_perturb:
                 image[i] = image[i] - epsilon*sign_data_grad[i]
                 perturbed_image[i] = torch.clamp(image[i], min=0, max=1)
     image = copy.copy(original_image)
     return image, perturbed_image

def find_target(i):
    image_to_perturb = []
    for k, (data, target) in enumerate(testloader):
        if k == i:
            if len(target) < batch_size:
                for j in range(len(target)):
                    if target[j] == classes.index('plane'):
                        image_to_perturb.append(j)
            else:
                for j in range(batch_size):
                    #print(f'target[j]: {target[j]}')
                    #print(classes.index('plane'))
                    if target[j] == classes.index('plane'):
                        image_to_perturb.append(j)
            break
    return image_to_perturb

def cifar10_resnet_normalize(t, mean=None, std=None):
    if mean is None:
        mean = CIFAR10_RESNET_MEAN
    if std is None:
        std = CIFAR10_RESNET_STD

    ts = []
    for i in range(3):
        ts.append(torch.unsqueeze((t[:, i] - mean[i]) / std[i], 1))
    return torch.cat(ts, dim=1)

def test(model, device, testloader, epsilon):
    correct = 0
    cnt = 0
    adv_examples = []
    bs = 0
    for i, (data, target) in enumerate(testloader):
        #print(f'i: {i}')
        # print(data.min())
        # print(data.max())
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        normalized_data = cifar10_resnet_normalize(data)

        output = model(normalized_data)
        init_pred = output.max(1, keepdim=False)[1]

        image_to_perturb = find_target(i)
        init_pred_list = init_pred.tolist()
        target_list = target.tolist()

        for k in range(len(init_pred_list)):
            # print(f'init_pred_list[k]: {init_pred_list[k]}')
            # print(f'target_list[k]: {target_list[k]}')
            if init_pred_list[k] != target_list[k]:
                if k in image_to_perturb:
                    image_to_perturb.remove(k)
                # print('not equal')
                cnt += 1
                continue
            else:
                if k in image_to_perturb:
                    target[k] = classes.index('deer')

        target = target.to(device)
        loss = F.nll_loss(output, target)

        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        data, perturbed_data = fgsm_attack(data, epsilon, data_grad, image_to_perturb)

        normalized_perturbed_data = cifar10_resnet_normalize(perturbed_data)
        output = model(normalized_perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        final_pred = final_pred.squeeze()
        #print(final_pred.shape)
        #print(target.shape)

        if len(target) < batch_size:
            bs = copy.copy(len(target))
        else:
            bs = copy.copy(batch_size)
        for j in range(bs):
            if final_pred[j] == target[j]:
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred[j], final_pred[j], adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred[j], final_pred[j], adv_ex))

    # Calculate final accuracy for this epsilon

    overall = (float(len(testloader)) - 1) * batch_size + bs - cnt
    final_acc = correct / overall
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, overall, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, testloader, eps)
    accuracies.append(acc)
    examples.append(ex)








