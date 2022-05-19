import cv2
import numpy as np

# https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape


def adjust_gamma(image, gamma=1.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


BLOCK_SIZE = 40
DELTA = 25


# Do the necessary noise cleaning and other stuffs.
# I just do a simple blurring here but you can optionally
# add more stuffs.
def preprocess(image):
    image = cv2.medianBlur(image, 3)
    return 255 - image


# Again, this step is fully optional and you can even keep
# the body empty. I just did some opening. The algorithm is
# pretty robust, so this stuff won't affect much.
def postprocess(image):
    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


# Just a helper function that generates box coordinates
def get_block_index(image_shape, yx, block_size):
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return np.meshgrid(y, x)


# Here is where the trick begins. We perform binarization from the
# median value locally (the img_in is actually a slice of the image).
# Here, following assumptions are held:
#   1.  The majority of pixels in the slice is background
#   2.  The median value of the intensity histogram probably
#       belongs to the background. We allow a soft margin DELTA
#       to account for any irregularities.
#   3.  We need to keep everything other than the background.
#
# We also do simple morphological operations here. It was just
# something that I empirically found to be "useful", but I assume
# this is pretty robust across different datasets.
def adaptive_median_threshold(img_in):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < DELTA] = 255
    kernel = np.ones((3,3),np.uint8)
    img_out = 255 - cv2.dilate(255 - img_out,kernel,iterations = 2)
    return img_out


# This function just divides the image into local regions (blocks),
# and perform the `adaptive_mean_threshold(...)` function to each
# of the regions.
def block_image_process(image, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[tuple(block_idx)] = adaptive_median_threshold(image[tuple(block_idx)])
    return out_image


# This function invokes the whole pipeline of Step 2.
def get_mask(img):
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    image_out = postprocess(image_out)
    return image_out


# This is the function used for composing
def sigmoid(x, orig, rad):
    k = np.exp((x - orig) * 5 / rad)
    return k / (k + 1.)


# Here, we combine the local blocks. A bit lengthy, so please
# follow the local comments.
def combine_block(img_in, mask):
    # First, we pre-fill the masked region of img_out to white
    # (i.e. background). The mask is retrieved from previous section.
    img_out = np.zeros_like(img_in)
    img_out[mask == 255] = 255
    fimg_in = img_in.astype(np.float32)

    # Then, we store the foreground (letters written with ink)
    # in the `idx` array. If there are none (i.e. just background),
    # we move on to the next block.
    idx = np.where(mask == 0)
    if idx[0].shape[0] == 0:
        img_out[idx] = img_in[idx]
        return img_out

    # We find the intensity range of our pixels in this local part
    # and clip the image block to that range, locally.
    lo = fimg_in[idx].min()
    hi = fimg_in[idx].max()
    v = fimg_in[idx] - lo
    r = hi - lo

    # Now we use good old OTSU binarization to get a rough estimation
    # of foreground and background regions.
    img_in_idx = img_in[idx]
    ret3,th3 = cv2.threshold(img_in[idx],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if np.alltrue(th3[:, 0] != 255):
        img_out[idx] = img_in[idx]
        return img_out

    # Then we normalize the stuffs and apply sigmoid to gradually
    # combine the stuffs.
    bound_value = np.min(img_in_idx[th3[:, 0] == 255])
    bound_value = (bound_value - lo) / (r + 1e-5)
    f = (v / (r + 1e-5))
    f = sigmoid(f, bound_value + 0.05, 0.2)

    # Finally, we re-normalize the result to the range [0..255]
    img_out[idx] = (255. * f).astype(np.uint8)
    return img_out


# We do the combination routine on local blocks, so that the scaling
# parameters of Sigmoid function can be adjusted to local setting
def combine_block_image_process(image, mask, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[tuple(block_idx)] = combine_block(
                image[tuple(block_idx)], mask[tuple(block_idx)])
    return out_image


# The main function of this section. Executes the whole pipeline.
def binarize(img):
    img = adjust_gamma(img)
    mask = get_mask(img)
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_out = combine_block_image_process(image_in, mask, 20)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)
    return image_out
