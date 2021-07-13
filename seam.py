import cv2
import numpy as np
import math
import matplotlib.pylab as plt
import imageio
import tqdm

MASK_TH = 240
def forward_energy(image):
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    m = np.zeros((height, width))

    for i in range(1, height):        
        for j in range(0, width):
            l_idx = (j + width - 1) % width
            r_idx = (j + 1) % width
            m1 = m[i - 1, l_idx] + abs(image[i,r_idx] - image[i, l_idx]) + abs(image[i - 1, j] - image[i, l_idx])
            m2 = m[i - 1, j] + abs(image[i,r_idx] - image[i, l_idx])
            m3 = m[i - 1,r_idx] + abs(image[i,r_idx] - image[i, l_idx]) + abs(image[i - 1, j] - image[i,r_idx])
            m[i, j] = min(m1, m2, m3)
    return m

def get_min_seam(image, mask):
    height, width = image.shape[:2]
    energy = forward_energy(image)
    x, y = np.where(mask > MASK_TH)
    for temp in zip(x, y):
        energy[temp] = math.inf
    backtrack = np.zeros((height, width), dtype=np.int)

    for i in range(1, height):
        for j in range(0, width):
            if j == 0:
                idx = np.argmin(energy[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = energy[i - 1, idx + j]
            else:
                idx = np.argmin(energy[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = energy[i - 1, idx + j - 1]
            energy[i, j] += min_energy
    
    boolmask = np.ones((height, width), dtype=np.bool)
    min_j = np.argmin(energy[-1])
    for i in range(height - 1, -1, -1):
        boolmask[i, min_j] = False
        min_j = backtrack[i, min_j]
    # plt.matshow(boolmask, cmap=plt.get_cmap('gnuplot2'))
    # plt.show()
    return boolmask

def removal(image, mask, d, x_dir, gif_name):
    gif_frames = []
    gif_frames.append(image)
    if not x_dir:
        image = np.rot90(image, 1)
        mask = np.rot90(mask, 1)
    for i in tqdm.tqdm(range(d), desc='generating {}'.format(gif_name)):
        height, width = image.shape[:2]
        boolmask = get_min_seam(image, mask)
        boolmask3c = np.stack([boolmask] * 3, axis=2)
        image = image[boolmask3c].reshape((height, width - 1, 3))
        mask = mask[boolmask].reshape((height, width - 1))
        if not x_dir:
            gif_frames.append(np.rot90(image, 3))
        else:
            gif_frames.append(image)
    imageio.mimsave(gif_name, gif_frames, 'GIF', duration=0.2)

    if not x_dir:
        return np.rot90(image, 3), np.rot90(mask, 3)
    else:
        return image, mask

if __name__ == '__main__':
    for i in range(12, 1000, 100):
        image = cv2.imread('data/imgs/{}.png'.format(i), 1)
        mask = cv2.imread('data/gt/{}.png'.format(i), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        remove_rate = min(0.4, 1 - 2 * np.sum(mask > MASK_TH) / (mask.shape[0] * mask.shape[1]))
        print('remove_rate for {}.png is'.format(i), remove_rate)
        dx = math.floor(remove_rate * mask.shape[0])
        dy = math.floor(remove_rate * mask.shape[1])
        image, mask = removal(image, mask, dx, True, 'seam_gif/{}_x.gif'.format(i))
        image, mask = removal(image, mask, dy, False, 'seam_gif/{}_y.gif'.format(i))
        cv2.imwrite('seam/{}.png'.format(i), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
