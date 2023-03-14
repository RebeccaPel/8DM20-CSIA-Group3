from __future__ import print_function
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import elastix
import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd
import subprocess
from elastix_registration import elastix_registration
import seg_metrics.seg_metrics as sg

# Main path to folder where elastix-5.0.x-win64 and TrainingData is found:
# Flavius:
# personal_path = os.path.join(
#     r'C:/Utrecht_Stuff/TC/2ndPart/TCdata/Data_Generation-master)
# Rebecca:
personal_path = os.path.join(
    r'C:/Users/20192157/OneDrive - TU Eindhoven/Documents/Master/8DM20 Capita selecta in medical image analysis')

ELASTIX_PATH = personal_path + '/elastix-5.0.0-win64/elastix.exe'
TRANSFORMIX_PATH = personal_path + '/elastix-5.0.0-win64/transformix.exe'

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def command_iteration(filter) :
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),
                                    filter.GetMetric()))
def evaluate_images(gdth_file,pred_file, folder_fixed, folder_moving):
    read_gdth_image = sitk.ReadImage(gdth_file)
    read_pred_image = sitk.ReadImage(pred_file)
    read_gdth_image_arr = sitk.GetArrayViewFromImage(read_gdth_image)
    read_pred_image_arr = sitk.GetArrayViewFromImage(read_pred_image)
    labels_gdth = np.nonzero(read_gdth_image_arr)
    labels_gdth = np.unique(labels_gdth[0])
    labels_pred = np.nonzero(read_pred_image_arr)
    labels_pred = np.unique(labels_pred[0])
    labels_not_matching = labels_pred[~np.isin(labels_pred,labels_gdth)].tolist()
    labels_not_matching.extend(labels_gdth[~np.isin(labels_gdth, labels_pred)].tolist())
    csv_file = "./results/" + folder_moving + "/metrics.csv"
    try:
        os.remove(csv_file)
    except OSError:
        pass
    store_dices = []
    store_precision = []
    for i in range(read_gdth_image_arr.shape[0]):
        metrics = sg.write_metrics(labels=[1],  # exclude background
                                   gdth_img=read_gdth_image_arr[i,:,:],
                                   pred_img=read_pred_image_arr[i,:,:],
                                   csv_file=csv_file,
                                   metrics=['dice', 'precision'])
        store_dices.append(metrics[0]['dice'])
        store_precision.append(metrics[0]['precision'])
    take_maximum_index = store_dices.index(max(store_dices))
    print(store_dices.index(max(store_dices)))
    print(store_precision[take_maximum_index])
    write_records = "./results/"+folder_fixed+"_checkmissing_affine.txt"
    file1 = open(write_records, "a")  # append mode
    file1.write(folder_moving + " Labels not matching: " + str(len(labels_not_matching)) + " Dice Score Max: " + str(max(store_dices)) + " Precision: " + str(store_precision[take_maximum_index]) + "\n")
    file1.close()
if __name__ == "__main__":
    # Make a results directory if non exists
    if os.path.exists('results') is False:
        os.mkdir('results')

    folder = "p102"
    # choosing patient 102 as fixed image
    image_1 = sitk.ReadImage(personal_path + "/TrainingData/p102/mr_bffe.mhd")
    image_1_seg = sitk.ReadImage(personal_path + "/TrainingData/p102/prostaat.mhd")
    fixed_image = sitk.GetArrayViewFromImage(image_1)
    fixed_image_seg = sitk.GetArrayViewFromImage(image_1_seg)

    fixed_image_path = "./TrainingData/p102/mr_bffe.mhd"
    fixed_image_seg_path = "./TrainingData/p102/prostaat.mhd"
    patients_list = [f.name for f in os.scandir("./TrainingData") if f.is_dir()]
    patients_list.remove(folder)
    for patient in patients_list:
        moving_image_path = "./TrainingData/" + patient + "/mr_bffe.mhd"
        moving_image_seg_path = "./TrainingData/" + patient + "/prostaat.mhd"
        result_path_transf = "./results/"+patient + "/result.mhd"
        jacobian_determinant_path = elastix_registration(moving_image_path,moving_image_seg_path,fixed_image_path, ELASTIX_PATH, TRANSFORMIX_PATH, patient)
        evaluate_images(fixed_image_seg_path, result_path_transf, folder, patient)
    # for now, considering patient 120 as moving image
    folder_2 = "p107"
    image_2 = sitk.ReadImage("./TrainingData/" + folder_2 + "/mr_bffe.mhd")
    image_2_seg = sitk.ReadImage("./TrainingData/" + folder_2 + "/prostaat.mhd")
    moving_image = sitk.GetArrayViewFromImage(image_2)
    moving_image_seg = sitk.GetArrayViewFromImage(image_2_seg)

    # Perform_elastix_registration(fixed, fixed_segmentation, moving, ELASTIX_PATH, TRANSFORMIX_PATH)
    # here we need to take the paths:

	# Computing the transformation, only needs to be done once.

    # look at the result image (create the path)
    # result_path = os.path.join('results', 'result.0.tiff')
    # result_path_transf = os.path.join('results', 'result.tiff')
    # transformed_bsplined_image = imageio.imread(result_path)
    # transformed_bsplined_segmentation = imageio.imread(result_path_transf)
    result_path ="./results/"+folder_2+ "/result.0.mhd"
    result_path_transf = "./results/"+folder_2 + "/result.mhd"

    transformed_bsplined_image_on = sitk.ReadImage(result_path)
    transformed_bsplined_image = sitk.GetArrayViewFromImage(transformed_bsplined_image_on)
    transformed_bsplined_segmentation_on = sitk.ReadImage(result_path_transf)
    transformed_bsplined_segmentation = sitk.GetArrayViewFromImage(transformed_bsplined_segmentation_on)
    jacobian_determinant_path = os.path.join("./results/"+folder_2, 'spatialJacobian.mhd')
    slice_nr = 43
    jacobian_determinant = imageio.imread(jacobian_determinant_path)
    jacobian_determinant_final = jacobian_determinant # > 0
    jacobian_sitk = sitk.ReadImage(jacobian_determinant_path)
    jacobian_sitk_array = sitk.GetArrayViewFromImage(jacobian_sitk)
    print('Minimum value: ', jacobian_sitk_array.min())
    print('Maximum value: ', jacobian_sitk_array.max())
    # display images and graphs
    fig, axarr = plt.subplots(1, 5)
    plt.suptitle('Slice number: ' + str(slice_nr), fontsize=16)
    axarr[0].imshow(fixed_image[slice_nr, :, :], cmap='gray')
    axarr[0].imshow(transformed_bsplined_segmentation[slice_nr, :, :], alpha=0.5*(transformed_bsplined_segmentation[slice_nr, :, :]>0.9), cmap='Purples')
    axarr[0].imshow(fixed_image_seg[slice_nr, :, :], alpha=0.5*(fixed_image_seg[slice_nr, :, :]>0.9), cmap='winter')
    axarr[0].set_title('Fixed image')
    axarr[1].imshow(moving_image[slice_nr, :, :], cmap='gray')
    axarr[1].set_title('Moving image')
    axarr[1].imshow(transformed_bsplined_segmentation[slice_nr, :, :], alpha=0.5*(transformed_bsplined_segmentation[slice_nr, :, :]>0.9), cmap='Purples')
    axarr[2].imshow(transformed_bsplined_segmentation[slice_nr, :, :], cmap='gray')
    axarr[2].set_title('Resulted B-splined segmentation')
    axarr[3].imshow(transformed_bsplined_image[slice_nr, :, :], cmap='gray')
    axarr[3].set_title('Resulted B-splined image')
    axarr[4].imshow(jacobian_determinant_final[30,:,:])
    axarr[4].set_title('Jacobian\ndeterminant')
    # plt.axis('off')

    plt.tight_layout()

    fig2, axarr = plt.subplots(1, 1)
    # Open the logfile into the dictionary log
    for i in range(4):
        # log_path = os.path.join('results', f'IterationInfo.0.R{i}.txt')
        log_path = os.path.join("./results/"+folder_2, 'IterationInfo.0.R{}.txt'.format(i))
        log = elastix.logfile(log_path)
        # Plot the 'metric' against the iteration number 'itnr'
        plt.plot(log['itnr'], log['metric'])
    plt.legend(['Resolution {}'.format(i) for i in range(5)])


    plt.show()

