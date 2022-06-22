import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as T
import random
import torch.nn.init as init
import os
import numpy as np

MODELS_DIR = '../MNIST_results/old_setup_check_1'
MEASURES_F = '../MNIST_results/all_results_old_check_1.npy'
# structure of the saved measures: first index is seed (see number in the end of the saved network),
# second is layer (0,1,2,3), third is one of the measurements:
# [seed][l][0]=test_loss_overall
# [seed][l][1]=train_loss_overall
# [seed][l][2]=max_flatness
# [seed][l][3]=tracial_flatness

MODELS = [6, 11, 10] # select the networks you want to work with currently
BATCH_SIZE = 512

# loading and transforming the image
train_data = torchvision.datasets.MNIST(root=".", train=True,
                                        transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))]),
                                        download=True)
test_data = torchvision.datasets.MNIST(root=".", train=False,
                                       transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))]),
                                       download=True)

training_generator = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
testing_generator = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
train_dataset_size = 60000
test_dataset_size = 10000

seed = 0


# Network specification
class MnistFCNet(nn.Module):
    def __init__(self):
        torch.manual_seed(100 * seed + 20)
        torch.cuda.manual_seed_all(100 * seed + 20)
        np.random.seed(100 * seed + 20)
        random.seed(100 * seed + 20)

        super(MnistFCNet, self).__init__()
        torch.backends.cudnn.deterministic = True

        self.fc1 = nn.Linear(784, 50)
        init.xavier_normal_(self.fc1.weight.data)
        init.zeros_(self.fc1.bias.data)
        self.fc2 = nn.Linear(50, 50)
        init.xavier_normal_(self.fc2.weight.data)
        init.zeros_(self.fc2.bias.data)
        self.fc3 = nn.Linear(50, 50)
        init.xavier_normal_(self.fc3.weight.data)
        init.zeros_(self.fc3.bias.data)
        self.fc4 = nn.Linear(50, 30)
        init.xavier_normal_(self.fc4.weight.data)
        init.zeros_(self.fc4.bias.data)
        self.fc5 = nn.Linear(30, 10)
        init.xavier_normal_(self.fc5.weight.data)
        init.zeros_(self.fc5.bias.data)

    def forward(self, x):
        act1 = F.relu(self.fc1(x))
        act2 = F.relu(self.fc2(act1))
        act3 = F.relu(self.fc3(act2))
        act4 = F.relu(self.fc4(act3))
        act5 = self.fc5(act4)
        return act5

    def layer1(self, x):
        act1 = F.relu(self.fc1(x))
        return act1

    def layer2(self, x):
        act1 = F.relu(self.fc1(x))
        act2 = F.relu(self.fc2(act1))
        return act2

    def layer3(self, x):
        act1 = F.relu(self.fc1(x))
        act2 = F.relu(self.fc2(act1))
        act3 = F.relu(self.fc3(act2))
        return act3

    def layer4(self, x):
        act1 = F.relu(self.fc1(x))
        act2 = F.relu(self.fc2(act1))
        act3 = F.relu(self.fc3(act2))
        act4 = F.relu(self.fc4(act3))
        return act4


flatness_measures = np.load(MEASURES_F)

loss = nn.CrossEntropyLoss()
for i in MODELS:
    model = torch.load(os.path.join(MODELS_DIR, 'network' + str(i)))
    model = model.cuda()
    train_activations = {"layer1": [], "layer2": [], "layer3": [], "layer4": []}
    train_colors_l = []
    netw_loss = 0.0
    train_data = []
    train_labels = []
    for b, l in training_generator:
        train_data += b.data.numpy().tolist()
        train_labels += l.data.numpy().tolist()
        b = b.to('cuda')
        l = l.to('cuda')
        output = model(b)
        netw_loss += loss(output, l).item()

        train_activations["layer1"] += model.layer1(b).data.cpu().numpy().tolist()
        train_activations["layer2"] += model.layer2(b).data.cpu().numpy().tolist()
        train_activations["layer3"] += model.layer3(b).data.cpu().numpy().tolist()
        train_activations["layer4"] += model.layer4(b).data.cpu().numpy().tolist()
        train_colors_l += l.data.cpu().numpy().tolist()

    print("Network train loss is ", netw_loss / len(training_generator))

    test_activations = {"layer1": [], "layer2": [], "layer3": [], "layer4": []}
    test_colors_l = []
    netw_loss = 0.0
    test_data = []
    test_labels = []
    for b, l in testing_generator:
        test_data += b.data.numpy().tolist()
        test_labels += l.data.numpy().tolist()
        b = b.to('cuda')
        l = l.to('cuda')
        output = model(b)
        netw_loss += loss(output, l).item()

        test_activations["layer1"] += model.layer1(b).data.cpu().numpy().tolist()
        test_activations["layer2"] += model.layer2(b).data.cpu().numpy().tolist()
        test_activations["layer3"] += model.layer3(b).data.cpu().numpy().tolist()
        test_activations["layer4"] += model.layer4(b).data.cpu().numpy().tolist()
        test_colors_l += l.data.cpu().numpy().tolist()

    print("Network test loss is ", netw_loss / len(testing_generator))


