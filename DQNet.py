import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torchsummary import summary
from DQNenv import envCube, Cube


class Qnet(nn.Module):
    def __init__(self, nb_observations, nb_actions):
        super(Qnet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(nb_observations, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, nb_actions)
        )

    def forward(self, status):
        return self.model(status)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Qnet(envCube.OBSEVATION_SPACE_VALUES, envCube.ACTION_SPACE_VALUES).to(device)
# print(model)
# summary(model, input_size=(2,4))
