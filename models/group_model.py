import torch
import torch.nn as nn
import torchvision.models as models


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        features = self.cnn(x).view(x.size(0), -1)  # (batch, 512)
        return features


class GroupActivityLSTM(nn.Module):
    def __init__(self, cnn_feature_dim=512, lstm_hidden_dim=500, num_classes=5):
        super(GroupActivityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=cnn_feature_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, sequence_len, feature_dim)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out
