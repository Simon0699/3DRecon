import torch
import math
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
#CAMERA_SIGMA = blur already baked into the raw image by the lens/sensor/sampling
CAMERA_SIGMA = 0.5
def create_octaves(x, s, o):
    #the raw image is not unblurred - it already sits at ~CAMERA_SIGMA. Bring it
    #up to the base SIGMA once (quadrature), so octave 0 level 0 is a true SIGMA.
    #octaves 1+ start from a subsampled slice already at SIGMA, so they must NOT
    #be bootstrapped again - that is why this lives here, not inside octave().
    sigma_bootstrap = (SIGMA ** 2 - CAMERA_SIGMA ** 2) ** 0.5
    x = blur(x, sigma_bootstrap)
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
            #2D hessian for spatial curvature (in x and y only)
            d2_hessian = hessian[1:3,1:3]
            #g(r) = r + 1/r + 2 = trace^2/det where r = l1/l2 (eigenvalue 1 / eigenvalue 2)
            #makes sure that eigenvalues are not imbalanced -> actual blob / curve not straight edge
            g = (d2_hessian.trace()**2)/(d2_hessian.det())
            #g(10) = 10 + 0.1 + 2 = 12.1 so if ratio more than 10 or less than 0.1 (depending on the way you order the eigenvalues)
            if (g > 12.1): remove[i] = True

        #remove bad indeces all at once, by position, to avoid corruption
        s_i = s_i[~remove]
        y_i = y_i[~remove]
        x_i = x_i[~remove]



        
        #DoG j is blur[j + 1] - blur[j], so it takes the scale of the lower
        #blur: SIGMA * k^j. The 2^octave_idx factor undoes the subsampling so
        #sigma comes out in original-image pixels.
        sigma = SIGMA * (2.0 ** (s_i.float() / sig)) * (2.0 ** octave_idx)
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
#takes in which octave along with the feature location x = (s,y,x,sigma)
#oct = whole octave that produced its respective DoG which the feature came from
def bilinear_interpolate(x, y, layer):
    H, W = layer.shape
    #the four surrounding integer pixels, clamped so rotated samples near the
    #image border never index out of bounds
    x0 = int(math.floor(float(x))); y0 = int(math.floor(float(y)))
    x1 = x0 + 1; y1 = y0 + 1
    fx = float(x) - x0
    fy = float(y) - y0
    x0 = min(max(x0, 0), W - 1); x1 = min(max(x1, 0), W - 1)
    y0 = min(max(y0, 0), H - 1); y1 = min(max(y1, 0), H - 1)
    tl = (1 - fx) * (1 - fy)
    tr = fx * (1 - fy)
    bl = (1 - fx) * fy
    br = fx * fy
    return (tl * layer[y0, x0] + tr * layer[y0, x1]
            + bl * layer[y1, x0] + br * layer[y1, x1])
def parabolic_interpolation(indices, bin36):
    doma = []
    for i in indices:
        if (i == 0):
            n1 = 35
            n2 = 1
        elif  (i == 35):
            n1 = 34
            n2 = 0
        else:
            n1 = i - 1
            n2 = i + 1
        v1 = bin36[n1]
        v2 = bin36[i]
        v3 = bin36[n2]
        a = (v1 + v3)/2 - v2
        b = (v3 - v1)/2
        #a == 0 means a flat (non-curved) peak - no sub-bin nudge
        xb = -b/(2*a) if a != 0 else 0.0
        doma.append(10 * (int(i) + float(xb)))
    return doma

def feature_description(oct, feature, descs):
    s_total = oct.shape[0] - 3
    s_f = int(round(feature[0].item())) # which s index did it come from 1 - 6
    y_f = int(feature[1].item())
    x_f = int(feature[2].item())
    sigma = 1.6 * (2 ** (s_f/s_total))
    radius = int(round(1.5 * sigma))
    bin36 = torch.zeros(36) # 36 bins for each 10 bin degree | 10 * 36 = 360
    layer = oct[s_f] # shape either [1,y,x] or [y,x]
    layer = layer.squeeze(0) # squeeze does not affect if dim 0 is already > 1 (what we want)\
    for i in range(-radius, radius + 1, 1):
        for j in range(-radius, radius + 1, 1):
            x = x_f + i
            y = y_f + j
            if not (1 <= y < layer.shape[0]-1 and 1 <= x < layer.shape[1]-1):
                continue
            l22 = (i**2 + j**2)
            Gx = layer[y,x + 1] - layer[y,x - 1]
            Gy = layer[y + 1, x] - layer[y - 1, x]
            theta = torch.atan2(Gy, Gx) * 180/torch.pi 
            theta = theta % 360
            m = torch.sqrt(Gx**2 + Gy**2)
            bin36[int(torch.floor(theta/10).item())] += m * math.exp(-l22/(2*sigma**2))
    top2 = torch.topk(bin36, 2)
    values = top2.values
    indices = top2.indices
    #keep the 2nd orientation only if it is a strong (>=80%) secondary peak
    if (values[1] < 0.8 * values[0]):
        indices = indices[:1]
    dom_angles = parabolic_interpolation(indices, bin36)  # degrees

    #descriptor window is MUCH larger than the orientation radius above:
    #4 cells of ~3*sigma each, plus a sqrt(2) margin for rotation -> ~8.5*sigma.
    #(radius = 1.5*sigma from earlier is only for the orientation histogram.)
    cell_size = 3 * sigma
    desc_radius = int(round(cell_size * 4 * (2 ** 0.5) / 2))
    #weighting Gaussian spans the whole window (sigma_w ~= half its width), so the
    #outer cells still receive meaningful weight instead of decaying to ~0.
    sigma_w = desc_radius / 2.0
    two_sig2_w = 2 * sigma_w ** 2

    #sampling grid: offsets relative to the keypoint, spanning the 4x4 cell window
    xcoords = torch.arange(-desc_radius, desc_radius + 1, 1, dtype = torch.float32)
    ycoords = torch.arange(-desc_radius, desc_radius + 1, 1, dtype = torch.float32)
    frame = torch.cartesian_prod(xcoords, ycoords)  # col0 = x-offset, col1 = y-offset
    cell_w = (2 * desc_radius) / 4.0                 # width of one of the 4 cells per axis

    #descs is the caller's list - append into it directly (do NOT rebind it here,
    #or the appends would land in a local list that is discarded on return)
    #at most 1 or 2 dominant angles - this is not a real triple loop
    for ang_deg in dom_angles:
        ang = ang_deg * math.pi / 180.0             # rotation matrix needs radians
        cos_a = math.cos(ang); sin_a = math.sin(ang)
        rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype = torch.float32)
        offsets = torch.matmul(frame, rot)          # rotated offsets: col0 = u(x), col1 = v(y)
        description = torch.zeros(16, 8)
        for j in range(len(offsets)):
            u = float(offsets[j, 0])                # rotated x-offset from keypoint
            v = float(offsets[j, 1])                # rotated y-offset from keypoint
            #which of the 4x4 cells this sample lands in (from its POSITION, not loop order)
            cx = int((u + desc_radius) // cell_w)
            cy = int((v + desc_radius) // cell_w)
            cx = min(max(cx, 0), 3)
            cy = min(max(cy, 0), 3)
            cell = cy * 4 + cx
            #translate to absolute image coords only for sampling the gradient
            sx = u + x_f
            sy = v + y_f
            Gx = bilinear_interpolate(sx + 1, sy, layer) - bilinear_interpolate(sx - 1, sy, layer)
            Gy = bilinear_interpolate(sx, sy + 1, layer) - bilinear_interpolate(sx, sy - 1, layer)
            #angle relative to the dominant orientation
            angle = (torch.atan2(Gy, Gx) * 180/torch.pi - ang_deg) % 360
            bini = int(torch.floor(angle/45).item())
            m = (Gx**2 + Gy**2)**0.5
            l22 = u**2 + v**2                        # distance to keypoint (rotation-invariant)
            description[cell][bini] += m * math.exp(-l22 / two_sig2_w)
        description = description.flatten()
        #+1e-7 guards a fully flat window (no gradients) from 0/0 -> nan
        description /= (description.norm() + 1e-7)
        description = torch.clamp(description,max = 0.2)
        description /= (description.norm() + 1e-7)
        descs.append(description)
#Full SIFT with image tensor. Returns:
#  descriptors: (N, 128) unit-normalized
#  positions:   (N, 2) as (y, x) in ORIGINAL full-resolution image pixels
#N is the number of descriptors, which may exceed the number of keypoints because
#a keypoint with two dominant orientations yields two descriptors.
NUM_OCTAVES = 4
def SIFT(image):
    full_H, full_W = image.shape[-2], image.shape[-1]
    descs = []
    positions = []
    oct = create_octaves(image, 6, NUM_OCTAVES)
    dog = GetDoG(oct)
    extrema = ExtremaSearch(dog, 6, 0.03)
    for i in range(len(extrema)):
        #octave i is downsampled from the original; use the MEASURED ratio per axis
        #(not 2**i) so odd-sized halvings that floor-round are scaled exactly.
        _, _, oh, ow = oct[i].shape
        sy = full_H / oh
        sx = full_W / ow
        for j in range(len(extrema[i])):
            feat = extrema[i][j]
            before = len(descs)
            feature_description(oct[i], feat, descs)
            #one position per descriptor actually produced (1 or 2), kept aligned
            y_full = feat[1].item() * sy
            x_full = feat[2].item() * sx
            for _ in range(len(descs) - before):
                positions.append([y_full, x_full])

    if not descs:
        return torch.empty(0, 128), torch.empty(0, 2)
    return torch.stack(descs), torch.tensor(positions)




    

    



    
