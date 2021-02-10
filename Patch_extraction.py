import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExtraction(nn.Module):
    def __init__(self, patch_size = (16,16), stride = (16, 16), dilation= (1,1)):
        super(PatchExtraction, self).__init__()

        self.patch_size = patch_size
        self.stride = stride
        self.dilation = dilation
        self.patch_extraction = nn.Unfold(kernel_size=patch_size, dilation=dilation, stride=stride)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.pad(x, (w % self.patch_size[0] // 2, w % self.patch_size[1] // 2,
                      h % self.patch_size[0] // 2, h % self.patch_size[1] // 2))
        patches = self.patch_extraction(x)
        patches = patches.view(b, -1, c, self.patch_size[0], self.patch_size[1])
        return patches


if __name__ == '__main__':
    x = torch.rand(1, 3, 128, 128).float()
    net = PatchExtraction()
    out = net(x)
    print(out.shape)
