import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, exposure, img_as_float, img_as_ubyte, metrics
from skimage.transform import resize
from scipy.ndimage import convolve, gaussian_gradient_magnitude

class MediaPlayer:
    @staticmethod
    def display_images(images, titles=None, x_labels=None, suptitle=None, cmaps=None):
        num_images = len(images)
        if titles is None:
            titles = [''] * num_images
        if x_labels is None:
            x_labels = [''] * num_images
        if cmaps is None:
            cmaps = ['gray'] * num_images

        assert len(titles) == num_images, "The number of images must match the number of titles."
        assert len(x_labels) == num_images, "The number of images must match the number of X-axis labels."
        assert len(cmaps) == num_images, "The number of images must match the number of colormaps."
    #Tam görüntüyü görmek için yorum satırlarını kaldır.
        #plt.figure(figsize=(num_images * 4, 4))
        #plt.suptitle(suptitle, fontsize=20, y=1.05)
        #for i, (image, title, x_label, cmap) in enumerate(zip(images, titles, x_labels, cmaps)):
            #ax = plt.subplot(1, num_images, i + 1)
            #plt.imshow(image, cmap=cmap)
            #plt.title(title)
            #ax.set_xlabel(x_label)
            #ax.axis('off')
        #plt.tight_layout()
        #plt.show()

    @staticmethod
    def display_histograms(images, titles=None, bins=256, color='black', suptitle=None):
        num_images = len(images)
        if titles is None:
            titles = [''] * num_images

        assert len(titles) == num_images, "The number of images must match the number of titles."

        plt.figure(figsize=(num_images * 4, 4))
        plt.suptitle(suptitle, fontsize=16)
        for i, (image, title) in enumerate(zip(images, titles)):
            ax = plt.subplot(1, num_images, i + 1)
            if image.ndim == 2:
                plt.hist(image.ravel(), bins=bins, color=color, alpha=0.75)
            else:
                for j, color in enumerate(['red', 'green', 'blue']):
                    plt.hist(image[:, :, j].ravel(), bins=bins, color=color, alpha=0.75, label=f'{color} channel')
                plt.legend()
            plt.title(title)
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Number of Pixels')
        plt.tight_layout()
        plt.show()

def resize_image(image, new_height, new_width):
    resized_image = resize(image, (new_height, new_width), anti_aliasing=True)
    return resized_image

def normalize_image(image):
    image = img_as_float(image)
    min_value = np.min(image)
    dynamic_range = np.max(image) - min_value
    normalized_image = (image - min_value) / dynamic_range
    return normalized_image

def read_image(image_path):
    image = io.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = color.rgb2gray(image)
    image = resize_image(image, 256, 256)
    return img_as_ubyte(image)

# Burada, görüntü dosyalarını yüklemek için dosya yolunu belirtin.
image_files = ['MRI_of_Human_Brain.jpg']  # Örnek: ['image1.jpg', 'image2.jpg']
images = [read_image(file_name) for file_name in image_files]

titles = [f'Image {i+1}' for i in range(len(images))]
MediaPlayer.display_images(images, titles)

image = images[0]

def edge_stop_function(gradient, beta):
    return 1.0 / (1.0 + (gradient / beta)**2)

def anisotropic_diffusion_filter(image, alpha, beta, iterations):
    image = img_as_float(image.copy())
    for _ in range(iterations):
        gradient_magnitude = gaussian_gradient_magnitude(image, sigma=1)
        c = edge_stop_function(gradient_magnitude, beta)
        diff_north = convolve(image, np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]]))
        diff_south = convolve(image, np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]))
        diff_east = convolve(image, np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]))
        diff_west = convolve(image, np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]]))
        image += alpha * (c * diff_north + c * diff_south + c * diff_east + c * diff_west)
    return img_as_ubyte(image)

alpha = 0.1
beta = 0.1
iterations = 3

image_filtered = anisotropic_diffusion_filter(image, alpha, beta, iterations)
score_value = metrics.structural_similarity(image, image_filtered)
MediaPlayer.display_images([image, image_filtered], ["Original Image", "Filtered Image"], ["", f'SSIM: {score_value:.5f}'])

def skull_stripping(image, se_closing, se_erosion, skull_remove_area=2000, show_pipeline=False):
    threshold_value = filters.threshold_otsu(image)
    binary_image = image > threshold_value
    filled_image = morphology.closing(binary_image, se_closing)
    filled_image = morphology.remove_small_holes(filled_image, area_threshold=skull_remove_area)
    eroded_image = morphology.erosion(filled_image, se_erosion)
    skull_stripped_image = np.where(eroded_image, image, 0)
    if show_pipeline:
        MediaPlayer.display_images(images=[image, binary_image, filled_image, eroded_image, skull_stripped_image],
                                   titles=['Original Image', 'Thresholded Image', 'Hole Filled Image', 'Eroded Image', "Skull Stripped Image"],
                                   x_labels=["", f'Threshold at {threshold_value:.2f}', '', '', ''])
    return img_as_ubyte(skull_stripped_image)

skull_stripped_image = skull_stripping(image, se_closing=morphology.disk(2), skull_remove_area=2000, se_erosion=morphology.disk(16), show_pipeline=True)

score_value = metrics.structural_similarity(image, skull_stripped_image)
MediaPlayer.display_images([image, skull_stripped_image], ["Original Image", "Skull Stripped Image"], ["", ''])

def contrast_enhancement(image):
    contrasted_image = exposure.equalize_hist(image)
    contrasted_image = normalize_image(contrasted_image)
    return img_as_ubyte(contrasted_image)

def check_pixel_intensity(image, lower_bound=50, upper_bound=100, threshold=2300):
    # Piksel yoğunluklarını ve sayısını hesapla
    pixel_counts, pixel_intensities = np.histogram(image.ravel(), bins=256, range=(0, 256))
    
    # İlgili yoğunluk aralığındaki piksellerin sayısını hesapla
    intensity_range_count = np.sum(pixel_counts[lower_bound:upper_bound+1])
    
    # Uyarı yazdır
    if intensity_range_count > threshold:
        print(f"Uyarı: Piksel yoğunluğu {lower_bound} ile {upper_bound} arasında olan piksellerin sayısı {intensity_range_count} ({threshold}'den fazla)!")
    else:
        print(f"Piksel yoğunluğu {lower_bound} ile {upper_bound} arasında olan piksellerin sayısı {intensity_range_count}.")
    
    return pixel_counts, pixel_intensities



contrasted_image = contrast_enhancement(skull_stripped_image)
pixel_counts, pixel_intensities = check_pixel_intensity(contrasted_image, lower_bound=50, upper_bound=100, threshold=2500)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].imshow(skull_stripped_image, cmap='gray')
axs[0, 0].set_title('Skull Stripped Image')
axs[0, 0].axis('off')
axs[0, 1].imshow(contrasted_image, cmap='gray')
axs[0, 1].set_title('Contrasted Image')
axs[0, 1].axis('off')
axs[1, 0].hist(skull_stripped_image.ravel(), bins=256, range=(0, 256), color='black')
axs[1, 0].set_title('Skull Stripped Image Histogram')
axs[1, 1].hist(contrasted_image.ravel(), bins=256, range=(0, 256), color='black')
axs[1, 1].set_title('Contrasted Image Histogram')
plt.tight_layout()
plt.show()
