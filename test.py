import cv2
import matplotlib.pyplot as plt
import numpy as np

import Layers

rng = np.random.RandomState(1234)
image = cv2.imread('./dataset/images/1.jpg', 0)
input = image.reshape(1, 1, image.shape[0], image.shape[1])
input = np.array(input, dtype=np.float64)
conv_3x3 = Layers.conv_3x3(input, rng, input.shape, (4, 1, 3, 3), (6, 4, 3, 3))

out = conv_3x3.outputs
g_p = Layers.global_avearge_pool(out, pool_size=(2, 2))
value = g_p.outputs.eval()
print value.shape
for i in xrange(value.shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.imshow(value[0, i, :, :], 'gray')
plt.show()
