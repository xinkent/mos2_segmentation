import os
import numpy as np

num_list = [str(i).zfill(3) for i in np.arange(1,45)]
for num in num_list:
    os.rename('../data/graphen/label_12/G_labaled_{}.png'.format(num), '../data/graphen/label_12/G_labeled_{}.png'.format(num))
