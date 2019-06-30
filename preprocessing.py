# contains all preprocessing functions for photo images and animation images

import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm


def preprocess_photo_images(source_path, save_path, target_size=256):
    """
    Converts all images in source_path to image with size target_size x target_size,
    then save to save_path.
    If original image has height or width smaller than target_size, then the image is not converted and ignored.

    :param source_path: directory containing photo images
    :param save_path: directory to save resized and cropped images
    """
    
    image_paths = glob.glob(os.path.join(source_path, '*'))

    resize_and_crop = transforms.Compose([
                                          transforms.Resize(target_size),
                                          transforms.CenterCrop(target_size)
                                         ])

    for image_path in tqdm(image_paths):
        if os.path.isdir(image_path):
            # directory. ignore
            continue

        image = Image.open(image_path)
        if image.height < target_size or image.width < target_size:
            # image too small. ignore
            continue
    
        new_image = resize_and_crop(image)
        image_name = (image_path.split('/')[-1]).split('\\')[-1]  # ex. images/1.jpg -> 1.jpg
        new_image_path = os.path.join(save_path, image_name)

        new_image.save(new_image_path)
        

def preprocess_animation_images(source_path, save_path, save_smoothed_path, target_size=256):
    """
    Converts all images in source_path to image with size target_size x target_size,
    then save to save_path.
    If original image has height or width smaller than target_size, then the image is not converted and ignored.
    Also, perform edge smoothing for resized and cropped images.

    :param source_path: directory containing photo images
    :param save_path: directory to save resized and cropped images
    :param save_smoothed_path: directory to save edge-smoothed images
    """

    image_paths = glob.glob(os.path.join(source_path, '*'))

    resize_and_crop = transforms.Compose([
                                          transforms.Resize(target_size),
                                          transforms.CenterCrop(target_size)
                                         ])

    for image_path in tqdm(image_paths):
        if os.path.isdir(image_path):
            # directory. ignore
            continue

        image = Image.open(image_path)
        if image.height < target_size or image.width < target_size:
            # image too small. ignore
            continue
    
        new_image = resize_and_crop(image)
        image_name = (image_path.split('/')[-1]).split('\\')[-1]  # ex. images/1.jpg -> 1.jpg
        new_image_path = os.path.join(save_path, image_name)
        new_image.save(new_image_path)

        # let's do edge smoothing
        edge_smoothed_image_path = os.path.join(save_smoothed_path, image_name)
        perform_edge_smoothing(new_image_path, edge_smoothed_image_path)
        

def perform_edge_smoothing(img_path, save_path, kernel_size=5, canny_threshold1=100, canny_threshold2=200):
    """
    Perform edge smoothing to given image, then save.

    :param img_path: path to an image
    :param save_path: path to save an image
    :kernel_size: kernel size to be used for edge dilation and Gaussian kernel. Must be an odd number.
    :canny_threshold1: first threshold for cv2.Canny
    :canny_threshold2: second threshold for cv2.Canny
    """
    
    assert kernel_size % 2 == 1  # kernel size must be odd

    bgr_img = cv2.imread(img_path)
    gray_img = cv2.imread(img_path, 0)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    padding = (kernel_size - 1) // 2
    pad_img = np.pad(bgr_img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')

    edges = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)  # detect edges
    dilation = cv2.dilate(edges, kernel)  # dilate edges

    smoothed_img = np.copy(bgr_img)
    idx = np.where(dilation != 0)  # index of dilation where value is not 0 == index near edges

    for i in range(len(idx[0])):
        # index of dilation where value is not 0 means index is near edges
        # change these points value using convolution between gauss kernel and are near the points
        # causing these points (and thus image) to become less sharp and blurry
        smoothed_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(
                    pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        smoothed_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(
                    pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        smoothed_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(
                    pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        # here, pad_img[i:i+kernel, j:j+kernel] == bgr_img[i-pad:i-pad+kernel, j-pad:j-pad+kernel]

    cv2.imwrite(save_path, smoothed_img)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('image_type_to_process',
                        choices=['photo', 'animation'],
                        help='Which preprocessing do you want?')

    parser.add_argument('--photo_image_source_path', 
                        help='Path to directory of photo images to be preprocessed. '
                             'Required when image_type_to_process is photo')

    parser.add_argument('--photo_image_save_path',
                        help='Path to directory where preprocessed photo images will be saved. '
                             'Required when image_type_to_process is photo')

    parser.add_argument('--animation_image_source_path',
                        help='Path to directory of animation images to be preprocessed. '
                             'Required when image_type_to_process is animation')

    parser.add_argument('--animation_image_save_path',
                        help='Path to directory where preprocessed animation images will be saved. '
                             'Required when image_type_to_process is animation')

    parser.add_argument('--animation_edge_smoothed_save_path',
                        help='Path to directory where preprocessed and edge smoothed animation images will be saved. '
                             'Required when image_type_to_process is animation')

    parser.add_argument('--target_size',
                        type=int,
                        default=256,
                        help='target size of preprocessed images')


    args = parser.parse_args()

    if args.image_type_to_process == 'photo':
        if args.photo_image_source_path is None or args.photo_image_save_path is None:
            parser.error('--photo_image_source_path and --photo_image_save_path required.')
        elif not os.path.isdir(args.photo_image_source_path) or not os.path.isdir(args.photo_image_save_path):
            parser.error('--photo_image_source_path and --photo_image_save_path must be existing directories.')
        else:
            preprocess_photo_images(args.photo_image_source_path, args.photo_image_save_path, target_size=args.target_size)

    elif args.image_type_to_process == 'animation':
        if args.animation_image_source_path is None or args.animation_image_save_path is None or args.animation_edge_smoothed_save_path is None:
            parser.error('--animation_image_source_path, --animation_image_save_path and --animation_edge_smoothed_save_path is required.')
        elif not os.path.isdir(args.animation_image_source_path) or not os.path.isdir(args.animation_image_save_path) or not os.path.isdir(args.animation_edge_smoothed_save_path):
            parser.error('--animation_image_source_path, --animation_image_save_path and --animation_edge_smoothed_save_path must be existing directories.')
        else:
            preprocess_animation_images(args.animation_image_source_path, args.animation_image_save_path, args.animation_edge_smoothed_save_path, target_size=args.target_size)

if __name__ == '__main__':

    """
    HOW TO USE

    Sorry, it became more complicated than what I intended...

    To process photo images, run following command:

    python preprocessing.py photo --photo_image_source_path [PHOTO_IMAGE_SOURCE_PATH] --photo_image_save_path [PHOTO_IMAGE_SAVE_PATH]
    
    To process animation images, run following command:

    python preprocessing.py animation --animation_image_source_path [ANIMATION_IMAGE_SOURCE_PATH] --animation_image_save_path [ANIMATION_IMAGE_SAVE_PATH] --animation_edge_smoothed_save_path [ANIMATION_EDGE_SMOOTHED_SAVE_PATH]

    Replace [...] with actual path. These path/directories must exist.
    """

    main()
        


