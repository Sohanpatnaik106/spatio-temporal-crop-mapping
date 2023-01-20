import torch.nn as nn
import torch

from torchvision.models import resnet50

class CNNDecoder(nn.Module):
    
    def __init__(self, config):
        super(CNNDecoder, self).__init__()

        self.config = config

        self.deconv1 = nn.ConvTranspose2d(self.config.model.lstm_hidden_size * 2, self.config.model.feature_map_size * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.config.model.feature_map_size * 8)
        self.relu1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(self.config.model.feature_map_size * 8, self.config.model.feature_map_size * 4, 4, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(self.config.model.feature_map_size * 4)
        self.relu2 = nn.ReLU(True)

        self.deconv3 = nn.ConvTranspose2d(self.config.model.feature_map_size * 4, self.config.model.feature_map_size * 3, 4, 2, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.config.model.feature_map_size * 3)
        self.relu3 = nn.ReLU(True)

        self.deconv4 = nn.ConvTranspose2d(self.config.model.feature_map_size * 3, self.config.model.feature_map_size * 2, 4, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(self.config.model.feature_map_size * 2)
        self.relu4 = nn.ReLU(True)

        self.deconv5 = nn.ConvTranspose2d(self.config.model.feature_map_size * 2, self.config.model.num_classes, 4, 2, 3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.relu4(self.bn4(self.deconv4(x)))
        x = self.sigmoid(self.deconv5(x))

        return x


class ResNetLSTM(nn.Module):

    def __init__(self, config):
        super(ResNetLSTM, self).__init__()

        self.config = config

        self.encoder = resnet50(pretrained = self.config.model.pretrained)
        self.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.lstm = nn.LSTM(input_size = self.config.model.lstm_input_size, hidden_size = self.config.model.lstm_hidden_size,
                            num_layers = self.config.model.lstm_num_layers, bias = self.config.model.lstm_bias, 
                            batch_first = self.config.model.batch_first, dropout = self.config.model.lstm_dropout,
                            bidirectional = self.config.model.bidirectional)
        self.decoder = CNNDecoder(config = self.config)

    def forward(self, x):
        
        im_encodings = []
        for t in range(x.shape[1]):
            im = x[:, t, :, :, :]
            im_encodings.append(self.encoder(im))

        im_encodings = torch.stack(im_encodings)
        im_encodings = im_encodings.view(im_encodings.size(1), im_encodings.size(0), im_encodings.size(2))
        lstm_out, lstm_hidden = self.lstm(im_encodings)
        decoded_out = self.decoder(lstm_out[:, -1, :].unsqueeze(-1).unsqueeze(-1))

        # bs x 120 x 24 x 24

        return decoded_out