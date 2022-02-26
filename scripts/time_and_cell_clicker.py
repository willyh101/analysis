from pathlib import Path
from holofun.movies import make_mov_array
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
import pickle

epoch = 4
mouse = 'w39_2'
date = '20211118'

tiff_path = Path('d:/frankenrig/experiments/',mouse, date, str(epoch))

tiff_list = list(tiff_path.glob('*.tif*'))
print(f'found {len(tiff_list)} tiffs.')

ch = 0
zplane = 0
x_cut = slice(110,512-110)
y_cut = slice(None)

mov_array, lengths = make_mov_array(tiff_list, zplane, ch, x_cut, y_cut)
mov = exposure.rescale_intensity(mov_array, out_range='uint16')


# percentile dfof method

f0 = np.percentile(mov, 20, axis=0)
mov_array_dfof = (mov - f0)/f0

mov_cut = np.split(mov_array_dfof, np.cumsum(lengths[:-1]), axis=0)
shortest = min(map(lambda x: x.shape[0], mov_cut))
mmov = np.array([a[:shortest,:,:] for a in mov_cut]).mean(0)

# transpose
mmov = mmov.transpose((0,2,1))


# do clicking
plt.figure()
def click_cells(im):
    plt.clf()
    plt.imshow(im)
    plt.axis('off')
    plt.title('Click your cells. Press enter to quit.')
    pts = plt.ginput(n=-1, timeout=-1)
    plt.close()
    return np.array(pts)

all_pts = []
all_frames = []

for i in range(mmov.shape[0]):
    mframe = mmov[i,:,:]
    pts = click_cells(mframe)
    all_frames.append(i)
    all_pts.append(pts)
    
out = dict(all_pts = all_pts, all_frames = all_frames)

save_path = tiff_path.parent/'locs_data.pickle'

with open(save_path, 'wb') as f:
    pickle.dump(out, f)