import torch
import torch.nn as nn
import torchvision.models as models


class PersonCNNLSTM(nn.Module):
    def __init__(self, cnn_output_dim=512, lstm_hidden_dim=3000, num_classes=5, lstm_layers=1):
        super(PersonCNNLSTM, self).__init__()

        # Pretrained ResNet-18 (new syntax for torchvision >= 0.13)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # Remove the final classification layer
        self.cnn = nn.Sequential(*modules)
        self.cnn_output_dim = cnn_output_dim  # Output is (batch, 512, 1, 1)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Final classification layer (per person action)
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # shape (batch, 3, H, W)
            feature = self.cnn(frame).view(batch_size, -1)  # shape (batch, 512)
            cnn_features.append(feature)

        # (batch, seq_len, feature_dim)
        cnn_features = torch.stack(cnn_features, dim=1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)

        # Use the output from the last time step
        final_output = lstm_out[:, -1, :]

        # Final classification layer
        output = self.fc(final_output)

        return output
