import torch
from MobileNetV2 import MobileNetV2
import torch.nn.functional as F
import torch.nn as nn


class TransferModel(MobileNetV2):

    def __init__(self, n_classes, model_path, origin_n_class=1000, input_size=224, width_mult=1):
        super().__init__(origin_n_class, input_size, width_mult)

        state_dict = torch.load(model_path, map_location='cpu')
        self.load_state_dict(state_dict)

        # Freeze weights
        for param in self.parameters():
            param.require_grads = False

        # Add output layer
        output_layer = nn.Linear(self.last_channel, n_classes)
        output_layer.weight.size(1)
        output_layer.weight.data.normal_(0, 0.01)
        output_layer.bias.data.zero_()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            output_layer,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x
