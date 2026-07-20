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
def ExtremaSearch(DoGs, sig, contrast_threshold = 0.03):
    extrema = []
    for octave_idx, dog in enumerate(DoGs):
        img_t = dog.squeeze(1).unsqueeze(0).unsqueeze(0)

        maxima = torch.nn.functional.max_pool3d(img_t, 3, stride = 1, padding = 1)
        minima = -torch.nn.functional.max_pool3d(-img_t, 3, stride = 1, padding = 1)

        #threshold on magnitude: a dark blob is a large-magnitude minimum
        strong = img_t.abs() > contrast_threshold
        is_max = (img_t == maxima) & strong
        is_min = (img_t == minima) & strong


        keep = (is_max | is_min).squeeze(0).squeeze(0)
        #a 3x3x3 cube needs a full set of neighbours, so drop the borders
        keep[0] = False
        keep[-1] = False
        keep[:, :1, :] = False
        keep[:, -1:, :] = False
        keep[:, :, :1] = False
        keep[:, :, -1:] = False
        
        operable_scale_space = img_t.squeeze(0).squeeze(0)
        nonz = keep.nonzero(as_tuple = True)
        s_i, y_i, x_i = nonz[0].clone(), nonz[1].clone(), nonz[2].clone()
        S, H, W = operable_scale_space.shape
        #mark keypoints to drop by position, resolved once after the loop
        remove = torch.zeros(len(s_i), dtype = torch.bool)
        # loop over non zero indeces to quadratic fit and check maxima
        for i in range(len(s_i)):
            s = s_i[i].item()
            y = y_i[i].item()
            x = x_i[i].item()
            converged = False
            for j in range(5):
                hessian = build_hessian(operable_scale_space, s, y, x)
                grad = get_grad(operable_scale_space,s, y, x)
                delta = torch.linalg.solve(hessian, -grad)
                check = torch.round(delta).to(torch.int32)
                if (check == 0).all():
                    #offset < 0.5 in every axis -> converged, stop
                    converged = True
                    break
                #not converged: move to the nearest integer sample and re-solve
                ds, dy, dx = tuple(check.tolist())
                s += ds
                y += dy
                x += dx
                #re-centering can walk back into the border where the finite
                #differences would read out of bounds - bail if so
                if not (1 <= s < S - 1 and 1 <= y < H - 1 and 1 <= x < W - 1):
                    break
            if converged:
                s_i[i] = s
                y_i[i] = y
                x_i[i] = x
            else:
                #never settled (or walked off the volume) - not a stable keypoint
                remove[i] = True
        #remove bad indeces all at once, by position, to avoid corruption
        s_i = s_i[~remove]
        y_i = y_i[~remove]
        x_i = x_i[~remove]



        
        #DoG j is blur[j + 1] - blur[j], so it takes the scale of the lower
        #blur: SIGMA * k^j. The 2^octave_idx factor undoes the subsampling so
        #sigma comes out in original-image pixels.
        sigma = SIGMA * (2.0 ** (s_i.float() / sig)) * (2.0 ** octave_idx)
        print(len(s_i), len(y_i), len(x_i))
        extrema.append(torch.stack([s_i.float(), y_i.float(), x_i.float(), sigma], dim = 1))
    return extrema
#s, y, x is the center
#p is the matrix to do finite differences on
def build_hessian(p, s, y, x):
    Dss = -2 * p[s, y, x] + p[s+1,y,x] + p[s-1,y,x]
    Dxx = -2 * p[s, y, x] + p[s,y,x+1] + p[s,y,x-1]
    Dyy = -2 * p[s, y, x] + p[s,y+1,x] + p[s,y-1,x]
    Dsx = ((p[s+1,y,x+1] - p[s-1,y,x+1]) - (p[s+1,y,x-1]-  p[s-1,y,x-1])) / 4
    Dsy = ((p[s+1,y+1,x] - p[s-1,y+1,x]) - (p[s+1,y-1,x]-  p[s-1,y-1,x])) / 4
    Dxy = ((p[s,y+1,x+1] - p[s,y+1,x-1]) - (p[s,y-1,x+1]-  p[s,y-1,x-1])) / 4
    return torch.tensor([[Dss, Dsy, Dsx], [Dsy, Dyy, Dxy], [Dsx, Dxy, Dxx]])
def get_grad(p,s,y,x):
    Dx = (p[s,y,x+1] - p[s,y,x-1]) / 2
    Dy = (p[s,y+1,x] - p[s,y-1,x]) / 2
    Ds = (p[s+1,y,x] - p[s-1,y,x]) / 2
    return torch.tensor([Ds,Dy,Dx])
    
