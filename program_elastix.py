from __future__ import print_function
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import elastix
import os
import sys
import numpy as np
import SimpleITK as sitk
import subprocess
from elastix_registration import elastix_registration
import seg_metrics.seg_metrics as sg
from datetime import datetime

# Main path to folder where elastix-5.0.x-win64 is found:
# Flavius:
personal_path = os.path.join(
      r'C:/Utrecht_Stuff/TC/2ndPart/TCdata/Data_Generation-master/Elastix')

ELASTIX_PATH = personal_path + '/elastix-5.0.1-win64/elastix.exe'
TRANSFORMIX_PATH = personal_path + '/elastix-5.0.1-win64/transformix.exe'

# Rebecca:
# personal_path = os.path.join(
#     r'C:/Users/20192157/OneDrive - TU Eindhoven/Documents/Master/8DM20 Capita selecta in medical image analysis')

# ELASTIX_PATH = personal_path + '/elastix-5.0.0-win64/elastix.exe'
# TRANSFORMIX_PATH = personal_path + '/elastix-5.0.0-win64/transformix.exe'

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')


def fast_scandir(dirname):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders


def command_iteration(filter):
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),
                                     filter.GetMetric()))


def get_array_from_filepath(filepath):
    image = sitk.ReadImage(filepath)
    image_arr = sitk.GetArrayFromImage(image)
    return image_arr


def evaluate_images(gdth_file, pred_file, folder_fixed, folder_moving, write_records):
    read_gdth_image_arr = get_array_from_filepath(gdth_file)
    read_pred_image_arr = get_array_from_filepath(pred_file)

    labels_gdth = np.unique(np.nonzero(read_gdth_image_arr)[0])
    labels_pred = np.unique(np.nonzero(read_pred_image_arr)[0])

    labels_not_matching = labels_pred[~np.isin(labels_pred,labels_gdth)].tolist()
    labels_not_matching.extend(labels_gdth[~np.isin(labels_gdth, labels_pred)].tolist())
    csv_file = f"./results/{folder_moving}/metrics.csv"
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
    # print(store_dices.index(max(store_dices)))
    # print(store_precision[take_maximum_index])

    file1 = open(write_records, "a")  # append mode
    file1.write(folder_moving +
                " Labels not matching: " + str(len(labels_not_matching)) +
                " Dice Score Max: " + str(max(store_dices)) +
                " Precision: " + str(store_precision[take_maximum_index]) + "\n")
    file1.close()


def visualise_results(moving_patient,
                      fixed_patient,
                      slice_nr,
                      path_to_training_data="./TrainingData/",
                      path_to_results="./results/"):

    moving_image = get_array_from_filepath(path_to_training_data + moving_patient + "/mr_bffe.mhd")
    moving_image_seg = get_array_from_filepath(path_to_training_data + moving_patient + "/prostaat.mhd")
    fixed_image = get_array_from_filepath(path_to_training_data + fixed_patient + "/mr_bffe.mhd")

    result_path = path_to_results + moving_patient + "/result.0.mhd"
    transformed_bsplined_image = get_array_from_filepath(result_path)
    result_path_transf = path_to_results + moving_patient + "/result.mhd"
    transformed_bsplined_segmentation = get_array_from_filepath(result_path_transf)

    jacobian_determinant_path = os.path.join(path_to_results + moving_patient, 'spatialJacobian.mhd')
    jacobian_determinant = imageio.imread(jacobian_determinant_path)

    fig, axarr = plt.subplots(1, 5)
    plt.suptitle('Slice number: ' + str(slice_nr), fontsize=16)
    axarr[0].imshow(fixed_image[slice_nr, :, :], cmap='gray')
    axarr[0].imshow(transformed_bsplined_segmentation[slice_nr, :, :],
                    alpha=0.5 * (transformed_bsplined_segmentation[slice_nr, :, :] > 0.9), cmap='Purples')
    axarr[0].imshow(fixed_image_seg[slice_nr, :, :], alpha=0.5 * (fixed_image_seg[slice_nr, :, :] > 0.9), cmap='winter')
    axarr[0].set_title('Fixed image')
    axarr[1].imshow(moving_image[slice_nr, :, :], cmap='gray')
    axarr[1].set_title('Moving image')
    axarr[1].imshow(transformed_bsplined_segmentation[slice_nr, :, :],
                    alpha=0.5 * (transformed_bsplined_segmentation[slice_nr, :, :] > 0.9), cmap='Purples')
    axarr[2].imshow(transformed_bsplined_segmentation[slice_nr, :, :], cmap='gray')
    axarr[2].set_title('Resulted B-splined segmentation')
    axarr[3].imshow(transformed_bsplined_image[slice_nr, :, :], cmap='gray')
    axarr[3].set_title('Resulted B-splined image')
    axarr[4].imshow(jacobian_determinant[30, :, :])
    axarr[4].set_title('Jacobian\ndeterminant')

    plt.tight_layout()

    # # fig2, axarr = plt.subplots(1, 1)
    # # Open the logfile into the dictionary log
    # for i in range(4):
    #     # log_path = os.path.join('results', f'IterationInfo.0.R{i}.txt')
    #     log_path = os.path.join("./results/" + moving_patient, 'IterationInfo.0.R{}.txt'.format(i))
    #     log = elastix.logfile(log_path)
    #     # Plot the 'metric' against the iteration number 'itnr'
    #     plt.plot(log['itnr'], log['metric'])
    # plt.legend(['Resolution {}'.format(i) for i in range(5)])

    plt.show()


if __name__ == "__main__":
    start_time = datetime.now()
    # Make a results directory if none exists
    if os.path.exists('results') is False:
        os.mkdir('results')



    patient_1 = "p102"

    #change here the name for the checkmissing file each time you change the parameter file name in elastix_registration
    write_records = f"./results/{patient_1}_checkmissing_translation.txt"

    # if write_records exists, delete it.
    try:
        os.remove(write_records)
    except OSError:
        pass

    fixed_image_path = f"./TrainingData/{patient_1}/mr_bffe.mhd"
    fixed_image = get_array_from_filepath(fixed_image_path)
    fixed_image_seg_path = f"./TrainingData/{patient_1}/prostaat.mhd"
    fixed_image_seg = get_array_from_filepath(fixed_image_seg_path)

    patients_list = [f.name for f in os.scandir("./TrainingData") if f.is_dir()]
    patients_list.remove(patient_1)

    print(patients_list)

    for patient in patients_list:
        moving_image_path = f"./TrainingData/{patient}/mr_bffe.mhd"
        moving_image_seg_path = f"./TrainingData/{patient}/prostaat.mhd"
        result_path_transf = f"./results/{patient}/result.mhd"
        # activate/deactivate registration.
        jacobian_determinant_path = elastix_registration(moving_image_path,
                                                         moving_image_seg_path,
                                                         fixed_image_path,
                                                         ELASTIX_PATH,
                                                         TRANSFORMIX_PATH,
                                                         patient)
        evaluate_images(fixed_image_seg_path, result_path_transf, patient_1, patient, write_records)
        print("transformation of ", patient, patient_1, "is done")

    end_time = datetime.now()
    print('Execution Time: {}'.format(end_time - start_time))

    visualise_results("p107",
                      patient_1,
                      43)
