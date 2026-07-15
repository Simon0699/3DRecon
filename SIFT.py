import torch 
import cv2 
from PIL import Image
from torchvision import transforms
image_to_tensor = transforms.ToTensor()

gaussian_filter = torch.nn.Conv2d(3, 3, 5, stride = 1, padding = 2, groups = 3, bias = False)
kernel = torch.tensor([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1],
], dtype=torch.float32)
kernel /= kernel.sum()
kernel = kernel.reshape(1, 1, 5, 5)
kernel = kernel.repeat(3, 1, 1, 1)
gaussian_filter.weight.data = kernel
#x = input image s = # of times to apply gaussian filter
def octave(x, s):
    ret = x.unsqueeze(0)
    for i in range(s - 1):
        x = gaussian_filter.forward(x)
        ret = torch.cat((ret, x.unsqueeze(0)))
    return ret


test = image_to_tensor(Image.open(r"F:\Coding\3D Recon\data\dronesplat\Simingsham\2411006_18_001.jpg").convert("RGB"))

def create_octaves(x, s, o):
    octaves = []
    for i in range(o):
        oct = octave(x, s)
        octaves.append(oct)
        shape = [x.shape[len(x.shape) - 2], x.shape[len(x.shape) - 1]]
        shape[0] = int(shape[0]/2)
        shape[1] = int(shape[1]/2) 
        x = oct[len(oct) - 1]
        x = torch.nn.AdaptiveAvgPool2d(shape).forward(x)
    return octaves

p = create_octaves(test, 6, 6)
def GetDoG(octaves):
    DoGs = []
    s = octaves[0].shape[0]
    for i in octaves:
        tcat = []
        for j in range(s - 1):
            diff = i[j + 1] - i[j]
            tcat.append(diff)
        DoGs.append(torch.stack(tcat, dim = 0))
    return DoGs

h = GetDoG(p)
for i in h:
    print(i.shape)
#26 point search
def ExtremaSearch(DoGs):
    extrema = []
    dogs_per_layer = DoGs[0].shape[0]
    for i in range(len(DoGs)):
        for j in range(1, dogs_per_layer - 1):
            DoGs[i][j] DoGs[i][j + 1] DoGs[i][j - 1]
