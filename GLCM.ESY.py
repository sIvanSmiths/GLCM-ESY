import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('Image.jpg').convert('L')

img_array = np.array(img, dtype=np.uint8)

glcm = np.zeros((256, 256), dtype=np.uint32)


for i in range(img_array.shape[0]-1):
    for j in range(img_array.shape[1]-1):
        i0 = int(img_array[i, j])
        i1 = int(img_array[i+1, j+1])
        glcm[i0, i1] += 1


glcm = glcm / np.sum(glcm)


contrast = np.sum((np.arange(glcm.shape[0]) - np.arange(glcm.shape[1])) ** 2 * glcm)
dissimilarity = np.sum(np.abs(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1])) * glcm)
homogeneity = np.sum(glcm / (1 + np.abs(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1]))))
energy = np.sum(glcm ** 2)
correlation = np.sum((np.arange(glcm.shape[0])[:, np.newaxis] - np.mean(np.arange(glcm.shape[0]))) *
                     (np.arange(glcm.shape[1])[np.newaxis, :] - np.mean(np.arange(glcm.shape[1]))) *
                     glcm / np.std(np.arange(glcm.shape[0])) / np.std(np.arange(glcm.shape[1])))


labels = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']

values = [contrast, dissimilarity, homogeneity, energy, correlation]

plt.bar(labels, values)
plt.show()