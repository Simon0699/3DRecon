import torch 
import cv2 
from PIL import Image
from torchvision import transforms
image_to_tensor = transforms.ToTensor()

SIGMA = 1.6

def gaussian_kernel(sigma):
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype = torch.float32)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return torch.outer(g, g)

def blur(x, sigma):
    k = gaussian_kernel(sigma)
    r = (k.shape[0] - 1) // 2
    x = x.unsqueeze(0)
    x = torch.nn.functional.conv2d(x, k.reshape(1, 1, *k.shape), padding = r)
    return x.squeeze(0)

#x = input image s = # of usable scale slices per octave
#blurs are spaced geometrically so each slice is k = 2^(1/s) times the last
def octave(x, s):
    k = 2.0 ** (1.0 / s)
    ret = [x]
    for i in range(s + 2):
        #incremental sigma to get from the current blur level to the next
        sigma_prev = SIGMA * (k ** i)
        sigma_next = SIGMA * (k ** (i + 1))
        sigma_step = (sigma_next ** 2 - sigma_prev ** 2) ** 0.5
        x = blur(x, sigma_step)
        ret.append(x)
    return torch.stack(ret, dim = 0)


test = image_to_tensor(Image.open(r"F:\Coding\3D Recon\data\dronesplat\Simingsham\2411006_18_001.jpg").convert("L"))
#o = # of octaves
def create_octaves(x, s, o):
    octaves = []
    for i in range(o):
        oct = octave(x, s)
        octaves.append(oct)
        #next octave starts from the slice already blurred to 2*sigma,
        #subsampled 2x - that blur level is what makes the resample safe
        x = oct[s][:, ::2, ::2]
    return octaves

p = create_octaves(test, 3, 4)
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
def ExtremaSearch(DoGs, contrast_threshold = 0.03):
    extrema = []
    for dog in DoGs:
        x = dog.squeeze(1).unsqueeze(0).unsqueeze(0)

        maxima = torch.nn.functional.max_pool3d(x, 3, stride = 1, padding = 1)
        minima = -torch.nn.functional.max_pool3d(-x, 3, stride = 1, padding = 1)

        #threshold on magnitude: a dark blob is a large-magnitude minimum
        strong = x.abs() > contrast_threshold
        is_max = (x == maxima) & strong
        is_min = (x == minima) & strong

        keep = (is_max | is_min).squeeze(0).squeeze(0)
        #a 3x3x3 cube needs a full set of neighbours, so drop the borders
        keep[0] = False
        keep[-1] = False
        keep[:, :1, :] = False
        keep[:, -1:, :] = False
        keep[:, :, :1] = False
        keep[:, :, -1:] = False

        s, y, xc = keep.nonzero(as_tuple = True)
        extrema.append(torch.stack([s, y, xc], dim = 1))
    return extrema

e = ExtremaSearch(h)
for i in e:
    print(i.shape)
