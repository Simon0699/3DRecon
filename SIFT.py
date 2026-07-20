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

def GetDoG(octaves):
    DoGs = []
    n_levels = octaves[0].shape[0]
    for i in octaves:
        tcat = []
        for j in range(n_levels - 1):
            diff = i[j + 1] - i[j]
            tcat.append(diff)
        DoGs.append(torch.stack(tcat, dim = 0))
    return DoGs


#26 point search
#s must match the value passed to create_octaves - it sets the geometric
#spacing k = 2^(1/s) that turns a DoG index back into a sigma
def ExtremaSearch(DoGs, s, contrast_threshold = 0.03):
    extrema = []
    for octave_idx, dog in enumerate(DoGs):
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
        print(keep.shape)

        nonz = keep.nonzero(as_tuple = True)
        s_idx, y, xc = nonz
        # loop over non zero indeces to quadratic fit and check maxima
        for i in range(len(s_idx)):
            s = s_idx[i].item()
            y = y[i].item()
            x = xc[i].item()
            hessian = build_hessian(x, s, y, x)
            grad = get_grad(x,s, y, x)
            delta = torch.linaglg.solve(hessian, -grad)
            


        
        #DoG j is blur[j + 1] - blur[j], so it takes the scale of the lower
        #blur: SIGMA * k^j. The 2^octave_idx factor undoes the subsampling so
        #sigma comes out in original-image pixels.
        sigma = SIGMA * (2.0 ** (s_idx.float() / s)) * (2.0 ** octave_idx)

        extrema.append(torch.stack([s_idx.float(), y.float(), xc.float(), sigma], dim = 1))
    return extrema
#s, y, x is the center
#p is the matrix to do finite differences on
def build_hessian(p, s, y, x):
    Dss = -2 * p[s, y, x] + p[s+1,y,x] + p[s-1,y,x]
    Dxx = -2 * p[s, y, x] + p[s,y,x+1] + p[s,y,x-1]
    Dyy = -2 * p[s, y, x] + p[s,y+1,x] + p[s,y-1,x]
    Dsx = (p[s+1,y,x+1] - p[s-1,y,x+1]) - (p[s+1,y,x-1]-  p[s-1,y,x-1])
    Dsy = (p[s+1,y+1,x] - p[s-1,y+1,x]) - (p[s+1,y-1,x]-  p[s-1,y-1,x])
    Dxy = (p[s,y+1,x+1] - p[s,y+1,x-1]) - (p[s,y-1,x+1]-  p[s,y-1,x-1])
    return torch.tensor[[Dss, Dsy, Dsx], [Dsy, Dyy, Dxy], [Dsx, Dxy, Dxx]]
def get_grad(p,s,y,x):
    Dx = p[s,y,x+1] - p[s,y,x-1]
    Dy = p[s,y+1,x] - p[s,y-1,x]
    Ds = p[s+1,y,x] - p[s-1,y,x]
    return torch.tensor([Ds,Dy,Dx])
    
