import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.core.extractor import ResidualBlock

autocast = torch.cuda.amp.autocast

class ste_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad):
        return F.hardtanh(grad)

class STE(nn.Module):
    def __init__(self):
        super(STE, self).__init__()
    def forward(self, x):
        return ste_fn.apply(x)

class sci_encoder(nn.Module):
    def __init__(
        self,
        sigma_range=[0, 1e-9],
        n_frame=8,
        in_channels=1,
        n_taps=2,
        resolution=[480, 640]):

        super(sci_encoder, self).__init__()

        assert n_taps in [1, 2], "[ERROR] n_taps should be either 1 or 2."

        self.sigma_range = sigma_range
        self.n_frame = n_frame
        self.in_channels = in_channels
        self.n_taps = n_taps
        self.resolution = resolution

        # -- Shutter code; Learnable parameters
        self.ce_weight = nn.Parameter(torch.Tensor(n_frame, in_channels, *resolution))

        # -- initialize
        nn.init.uniform_(self.ce_weight, a=-1, b=1)

        self.ste = STE()

    def forward(self, frames):

        # -- print ("[INFO] self.ce_weight.device: ", self.ce_weight.device)
        ce_code = self.ste(self.ce_weight)
        # -- print ("[INFO] ce_code.device: ", ce_code.device)

        frames = frames[..., :self.resolution[0], :self.resolution[1]]
        frames = frames.contiguous()
        frames = torch.unsqueeze(frames, 2)

        # -- print ("[INFO] ce_code.shape: ", ce_code.shape)
        # -- print ("[INFO] frames.shape: ", frames.shape)

        # -- repeat by the batch size 
        ce_code = ce_code.repeat(frames.shape[0], 1, 1, 1, 1)
        # -- print ("[INFO] ce_code.shape: ", ce_code.shape)
        # -- print ("[INFO] ce_code.squeeze(2).shape: ", ce_code.squeeze(2).shape)

        ce_blur_img = torch.zeros(frames.shape[0], self.in_channels * self.n_taps, *self.resolution).to(frames.device) # -- (b, c, h, w)

        # -- print ("[INFO] ce_blur_img.shape: ", ce_blur_img.shape)
        ce_blur_img[:, 0, ...] = torch.sum(      ce_code  * frames, axis=1) / self.n_frame
        ce_blur_img[:, 1, ...] = torch.sum((1. - ce_code) * frames, axis=1) / self.n_frame

        # -- add noise
        noise_level = np.random.uniform(*self.sigma_range)
        ce_blur_img_noisy = ce_blur_img + torch.tensor(noise_level).to(frames.device) * torch.randn(ce_blur_img.shape).to(frames.device)

        # -- concat snapshots and mask patterns
        out = torch.zeros(frames.shape[0], self.n_taps + self.n_frame, *self.resolution).to(frames.device)

        # -- print ("[INFO] out.shape: ", out.shape)
        out[:, :self.n_taps, :, :] = ce_blur_img_noisy
        out[:, self.n_taps:, :, :] = ce_code.squeeze(2)

        return out

class sci_decoder(nn.Module):
    def __init__(self,
        n_frame=8,
        n_taps=2,
        output_dim=128,
        norm_fn="batch",
        dropout=.0):

        super(sci_decoder, self).__init__()

        self.norm_fn = norm_fn
        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=4, num_channels=4*n_frame)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(4*n_frame)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(4*n_frame, affine=True)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()

        # -- Input Convoultion
        # -- Assuming n_frame=8; n_ich=10; n_och=32
        self.conv1 = nn.Conv2d(n_taps+n_frame, 4*n_frame, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # -- Residual Blocks
        self.layer1 = self._make_layer( 4*n_frame,  4*n_frame, stride=1)
        self.layer2 = self._make_layer( 4*n_frame, 16*n_frame, stride=2)
        self.layer3 = self._make_layer(16*n_frame, 64*n_frame, stride=1)

        # -- Output Convolution
        self.conv2 = nn.Conv2d(64*n_frame, output_dim*n_frame, kernel_size=1)

        if dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # -- self.modules() is a PyTorch utility function that returns all submodules of this nn.Module recursively.
        # -- This means it will looop through every layer: conv1, layer1, layer2, layer3, conv2 and so on.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # -- Private function to make residual blocks
    def _make_layer(self, n_ich, n_och, stride=1):
        layer1 = ResidualBlock(n_ich, n_och, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(n_och, n_och, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x):
        # -- x = [L, R]
        # -- L, R ~ (b, c, h, w); c=n_taps+n_frame

        # -- if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # -- print ("[INFO] x.shape: ", x.shape)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        # -- expand the temporal dimension
        # -- (b, c, h, w) -> (b*t, c//t, h, w)
        x = x.contiguous()
        x = x.view(x.shape[0]*8, x.shape[1]//8, x.shape[-2], x.shape[-1])

        if self.dropout is not None:
            x = self.dropout(x)

        # -- if input is list, split the first dimension
        if is_list:
            x = torch.split(x, x.shape[0] // 2, dim=0)

        return x

