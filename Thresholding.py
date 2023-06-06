# **** imports ****
import matplotlib.pyplot as plt

from skimage import data, io, color
from skimage.filters import threshold_otsu, threshold_local

# **** additional import used in second pass ****
from skimage.filters import try_all_threshold


# **** read cathedral image (convert to grayscale) ****
cathedral = color.rgb2gray(io.imread('./images/pexels-painting.jpg'))

# **** display cathedral ****
plt.figure(figsize=(8, 8))
plt.imshow(cathedral, cmap='gray')
plt.title('Cathedral')
plt.show()


# **** Convert values from 0 to 1 to 0 to 255 ****
cathedral = cathedral * 255


# **** Apply Otsu thresholding (first pass)
#      Calculates an optimal threshold so that the intra-class
#      variance is minimal and inter-class variance is maximal
thresh = threshold_otsu(cathedral)

# **** Apply local thresholding (second pass)
#      Local filtering is useful when there is a wide variation
#      in the illumination across the image
#      Local because neighboring pixels are used to calculate the threshold
#thresh = threshold_local(cathedral, block_size=25)

# **** display threshold ****
print(f'thresh: {thresh}')


# **** create binary image ****
cathedral_binary = cathedral > thresh

# **** display the values in cathedral_binary ****
print('cathedral_binary:\n',  cathedral_binary)


# **** display images ****
fig, axes = plt.subplots(   nrows=1,
                            ncols=2, 
                            figsize=(14, 8),
                            sharex=True,
                            sharey=True)

# **** flatten axes ****
ax = axes.ravel()

# **** display original image ****
ax[0].imshow(cathedral, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# **** display binary image ****
ax[1].imshow(cathedral_binary, cmap='gray')
ax[1].set_title('Thresholded Image')
ax[1].axis('off')

fig.tight_layout()
plt.show()


# **** On second pass exit here! ****
#exit(0)


# **** ****
fig, ax = plt.subplots(figsize=(6, 6))

ax.hist(cathedral.ravel())

# **** display histogram ****
ax.set_title('Histogram of Grayscale Image')
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Number of Pixels')

# **** display histogram ****
ax.axvline(thresh, color='y')
plt.show()


# **** generates a number of threshold images ****
fig, ax = try_all_threshold(cathedral, 
                            figsize=(16, 12), 
                            verbose=False)

# **** display ****
plt.show()
