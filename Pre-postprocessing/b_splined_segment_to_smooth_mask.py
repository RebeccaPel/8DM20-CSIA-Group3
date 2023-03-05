import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_mhd(filepath):
    itk_image = sitk.ReadImage(filepath)
    image_array = sitk.GetArrayViewFromImage(itk_image)
    plt.figure()
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(image_array.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(image_array[i])
        plt.axis('off')
    plt.show()

def find_mask_window(image_arr):
    relevant_rows = []
    for r in range(image_arr.shape[2]):
        if np.sum(image_arr[:, :, r]) != 0:
            relevant_rows.append(r)
    relevant_columns = []
    for c in range(image_arr.shape[1]):
        if np.sum(image_arr[:, c, :]) != 0:
            relevant_columns.append(c)

    return (relevant_rows[0], relevant_rows[-1]), (relevant_columns[0], relevant_columns[-1])



def fill_mask(image_array, z, row_range, column_range):
    image_arr = image_array.copy()[z,:,:]
    changes = 1
    while changes > 0:
        changes = 0
        for r in range(row_range[0], row_range[1]):
            for c in range(column_range[0], column_range[1]):
                if image_arr[c, r] < 1:
                    above = image_arr[c, r - 1]
                    below = image_arr[c, r + 1]
                    left = image_arr[c - 1, r]
                    right = image_arr[c + 1, r]
                    if (above + below + left + right) >= 3:
                        image_arr[c, r] = 1
                        changes += 1
    return image_arr

def shape_flooding(image_arr):
    im_floodfill = image_arr.copy()
     
    # Notice the size needs to be 2 pixels than the image.
    h, w = image_arr.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = image_arr | im_floodfill_inv
    
    return im_out


result_path = "./results/result.mhd"
# show_mhd(result_path)

itk_image = sitk.ReadImage(result_path)
image_array = sitk.GetArrayViewFromImage(itk_image)
rr, cr = find_mask_window(image_array)

outp = fill_mask(image_array, 60, cr, rr)
outp = np.uint8(outp)
im_out = shape_flooding(outp)

plt.figure()
plt.gray()
plt.imshow(image_array[60, cr[0]:cr[1], rr[0]:rr[1]])
plt.show()

plt.figure()
plt.gray()
plt.imshow(outp[cr[0]:cr[1], rr[0]:rr[1]])
plt.show()

plt.figure()
plt.gray()
plt.imshow(im_out[cr[0]:cr[1], rr[0]:rr[1]])
plt.show()


# using a findContours() function, not working yet
# outp = outp.astype(np.uint8)
# contours, _ = cv2.findContours(outp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     cv2.drawContours(outp, [contour], 0, (0, 0, 255), 5)
#     cv2.imshow('shapes', outp)
# cv2.drawContours(outp, contours, -1, (0,255,0), 3)
# cv2.imshow('shapes', outp)

