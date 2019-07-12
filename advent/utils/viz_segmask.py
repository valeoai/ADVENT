import numpy as np
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156,
           190, 153, 153, 153, 153, 153, 250,
           170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152,
           70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0,
           142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
