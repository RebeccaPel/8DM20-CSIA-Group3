import imageio.v2 as imageio
import matplotlib.pyplot as plt
import elastix
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import subprocess
from elastix_registration import elastix_registration


# make sure you select right your .exe paths
ELASTIX_PATH = os.path.join(
    r'C:/Utrecht_Stuff/TC/2ndPart/TCdata/Data_Generation-master/Elastix/elastix-5.0.1-win64/elastix.exe')
TRANSFORMIX_PATH = os.path.join(
    r'C:/Utrecht_Stuff/TC/2ndPart/TCdata/Data_Generation-master/Elastix/elastix-5.0.1-win64/transformix.exe')

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')



if __name__ == "__main__":
    # Make a results directory if non exists
    if os.path.exists('results') is False:
        os.mkdir('results')

    # choosing patient 102 as fixed image
    image_1 = sitk.ReadImage("./TrainingData/p102/mr_bffe.mhd")
    image_1_seg = sitk.ReadImage("./TrainingData/p102/prostaat.mhd")
    fixed_image = sitk.GetArrayViewFromImage(image_1)
    fixed_image_seg = sitk.GetArrayViewFromImage(image_1_seg)

    # for now, considering patient 120 as moving image
    image_2 = sitk.ReadImage("./TrainingData/p120/mr_bffe.mhd")
    moving_image = sitk.GetArrayViewFromImage(image_2)

    # Perform_elastix_registration(fixed, fixed_segmentation, moving, ELASTIX_PATH, TRANSFORMIX_PATH)
    # here we need to take the paths:
    fixed_image_path = "./TrainingData/p102/mr_bffe.mhd"
    fixed_image_seg_path = "./TrainingData/p102/prostaat.mhd"
    moving_image_path = "./TrainingData/p120/mr_bffe.mhd"
	
	# anytime you want to see visualize different, you can comment this line to avoid waiting too much time
    jacobian_determinant_path = elastix_registration(fixed_image_path, fixed_image_seg_path, moving_image_path, ELASTIX_PATH, TRANSFORMIX_PATH)


    # look at the result image (create the path)
    # result_path = os.path.join('results', 'result.0.tiff')
    # result_path_transf = os.path.join('results', 'result.tiff')
    # transformed_bsplined_image = imageio.imread(result_path)
    # transformed_bsplined_segmentation = imageio.imread(result_path_transf)
    result_path = os.path.join('results', 'result.0.mhd')
    result_path_transf = os.path.join('results', 'result.mhd')

    transformed_bsplined_image_on = sitk.ReadImage(result_path)
    transformed_bsplined_image = sitk.GetArrayViewFromImage(transformed_bsplined_image_on)
    transformed_bsplined_segmentation_on = sitk.ReadImage(result_path_transf)
    transformed_bsplined_segmentation = sitk.GetArrayViewFromImage(transformed_bsplined_segmentation_on)
    slice_nr = 30
    # display images and graphs
    fig, axarr = plt.subplots(1, 5)
    plt.suptitle('Slice number: ' + str(slice_nr), fontsize=16)
    axarr[0].imshow(fixed_image[slice_nr, :, :], cmap='gray')
    axarr[0].imshow(fixed_image_seg[slice_nr, :, :], alpha=0.5*(fixed_image_seg[slice_nr, :, :]>0), cmap='Purples')
    axarr[0].set_title('Fixed image')
    axarr[1].imshow(moving_image[slice_nr, :, :], cmap='gray')
    axarr[1].set_title('Moving image')
    axarr[1].imshow(transformed_bsplined_segmentation[slice_nr, :, :], alpha=0.8*(transformed_bsplined_segmentation[slice_nr, :, :]>0), cmap='Purples')
    axarr[2].imshow(transformed_bsplined_segmentation[slice_nr, :, :], cmap='gray')
    axarr[2].set_title('Resulted B-splined segmentation')
    axarr[3].imshow(transformed_bsplined_image[slice_nr, :, :], cmap='gray')
    axarr[3].set_title('Resulted B-splined image')
    # axarr[4].imshow(imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))[30,:,:])
    # axarr[4].set_title('Jacobian\ndeterminant')
    # plt.axis('off')

    plt.tight_layout()

    fig2, axarr = plt.subplots(1, 1)
    # Open the logfile into the dictionary log
    for i in range(4):
        log_path = os.path.join('results', 'IterationInfo.0.R{}.txt'.format(i))
        log = elastix.logfile(log_path)
        # Plot the 'metric' against the iteration number 'itnr'
        plt.plot(log['itnr'], log['metric'])
    plt.legend(['Resolution {}'.format(i) for i in range(5)])


    plt.show()

