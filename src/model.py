import torch.nn as nn
import torch

from torchvision.models import resnet50, vgg16

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
        self.encoder.fc = nn.Identity()

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

class VGGLSTM(nn.Module):

    def __init__(self, config):
        super(VGGLSTM, self).__init__()

        self.config = config

        self.encoder = vgg16(pretrained = self.config.model.pretrained)
        self.encoder.features[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

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







class DeformableConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class DeformableConvLSTM(nn.Module):
    def __init__(self, config):
        super(DeformableConvLSTM, self).__init__()


        self.config = config

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        encoder = []
        inplanes = 4
        outplanes = 32
        for i in range(5):
            if self.config.model.deform and self.config.model.min_deform_layer <= i+1:
                encoder.append(DeformableConv2d(inplanes, outplanes, 3, padding=1, bias=False, modulation=self.config.model.modulation))
            else:
                encoder.append(nn.Conv2d(inplanes, outplanes, 3, padding=1, bias=False))
            encoder.append(nn.BatchNorm2d(outplanes))
            encoder.append(self.relu)
            if i == 1:
                encoder.append(self.pool)
            inplanes = outplanes
            outplanes *= 2

        self.encoder = nn.Sequential(*encoder)

        # self.fc = nn.Linear(256, 10)
        self.lstm = nn.LSTM(input_size = self.config.model.lstm_input_size, hidden_size = self.config.model.lstm_hidden_size,
                            num_layers = self.config.model.lstm_num_layers, bias = self.config.model.lstm_bias, 
                            batch_first = self.config.model.batch_first, dropout = self.config.model.lstm_dropout,
                            bidirectional = self.config.model.bidirectional)
        self.decoder = CNNDecoder(config = self.config)

    def forward(self, x):

        im_encodings = []
        for t in range(x.shape[1]):
            im = x[:, t, :, :, :]
            im_encodings.append(self.avg_pool(self.encoder(im)))

        im_encodings = torch.stack(im_encodings)

        im_encodings = im_encodings.view(im_encodings.size(1), im_encodings.size(0), im_encodings.size(2))
        lstm_out, lstm_hidden = self.lstm(im_encodings)
        decoded_out = self.decoder(lstm_out[:, -1, :].unsqueeze(-1).unsqueeze(-1))

        # bs x 120 x 24 x 24

        return decoded_out
        x = self.features(input)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        output = self.fc(x)

        return output