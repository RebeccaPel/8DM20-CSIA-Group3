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
# from lmi import lmi
from mutual_info import nmi, ssim_inf
from statistics import mean
import csv
import shutil

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


def evaluate_labels(gdth_file, pred_file, folder_fixed, folder_moving, similarity="None", path=None):
    read_gdth_image_arr = get_array_from_filepath(gdth_file)
    read_pred_image_arr = get_array_from_filepath(pred_file)

    # for i in range(read_gdth_image_arr.shape[0]):
    metrics = sg.write_metrics(labels=[1],  # exclude background
                               gdth_img=read_gdth_image_arr,
                               pred_img=read_pred_image_arr,
                               metrics=['dice', 'precision', 'jaccard'])

    to_write = {'patient': folder_moving, 'dice': str(round(metrics[0]['dice'][0],5)), 'precision':str(round(metrics[0]['precision'][0],5)),
                'jaccard': str(round(metrics[0]['jaccard'][0],5))}

    path_to_save = path+'/'+folder_fixed+'_metrics_'+similarity+'.csv'

    if os.path.exists(path_to_save) is False:
        with open(path_to_save, 'a',
                  newline='') as f:
            fieldnames = ['patient', 'dice', 'precision', 'jaccard']
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            f.close()

    with open(path_to_save, 'a', newline='') as f:
        fieldnames = ['patient', 'dice', 'precision', 'jaccard']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writerow(to_write)
        f.close()

def evaluate_labels_on_images(fixed_image, moving_image, folder_fixed, folder_moving, similarity="None", path=None):
    # read_gdth_image_arr = get_array_from_filepath(gdth_file)
    # read_pred_image_arr = get_array_from_filepath(pred_file)

    # for i in range(read_gdth_image_arr.shape[0]):
    metrics = sg.write_metrics(labels=[1],  # exclude background
                               gdth_img=fixed_image,
                               pred_img=moving_image,
                               metrics=['dice', 'precision', 'jaccard'])

    to_write = {'patient_fixed': folder_fixed, 'dice': str(round(metrics[0]['dice'][0],5)), 'precision':str(round(metrics[0]['precision'][0],5)),
                'jaccard': str(round(metrics[0]['jaccard'][0],5))}

    path_to_save = path

    if os.path.exists(path_to_save) is False:
        with open(path_to_save, 'a',
                  newline='') as f:
            fieldnames = ['patient_fixed', 'dice', 'precision', 'jaccard']
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            f.close()

    with open(path_to_save, 'a', newline='') as f:
        fieldnames = ['patient_fixed', 'dice', 'precision', 'jaccard']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writerow(to_write)
        f.close()


def visualise_results(moving_patient,
                      fixed_patient,
                      slice_nr,
                      fixed_image_seg,
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

    fig2, axarr = plt.subplots(1, 1)
    # Open the logfile into the dictionary log
    for i in range(3):
        # log_path = os.path.join('results', f'IterationInfo.0.R{i}.txt')
        log_path = os.path.join("./results/" + moving_patient, 'IterationInfo.1.R{}.txt'.format(i))
        log = elastix.logfile(log_path)
        # Plot the 'metric' against the iteration number 'itnr'
        plt.plot(log['itnr'], log['metric'])
    plt.legend(['Resolution {}'.format(i) for i in range(5)])

    plt.show()

def run_pipe(patients_list_fixed, similarity=None, registration=False, path_for_score=None):

    for patient_fixed in patients_list_fixed:

        patient_1 = patient_fixed

        if registration is True:
            if os.path.exists('results/' + patient_1):
                shutil.rmtree('results/' + patient_1)

        # patient_1 = "p107"

        fixed_image_path = f"./TrainingData/{patient_1}/mr_bffe.mhd"
        fixed_image = get_array_from_filepath(fixed_image_path)
        fixed_image_seg_path = f"./TrainingData/{patient_1}/prostaat.mhd"

        patients_list = [f.name for f in os.scandir("./TrainingData") if f.is_dir()]
        patients_list.remove(patient_1)

        print(patients_list)

        if similarity == "ssim" or similarity == "nmi":

            nmi_dict = {}
            ssim_dict = {}

            for patient in patients_list:

                moving_image_path = f"./TrainingData/{patient}/mr_bffe.mhd"
                moving_image = get_array_from_filepath(moving_image_path)
                nmi_res = nmi(fixed_image, moving_image)
                nmi_dict[patient] = nmi_res

                ssim = ssim_inf(fixed_image, moving_image)
                ssim_dict[patient] = ssim

            max_5_nmi = sorted(nmi_dict, key=nmi_dict.get, reverse=True)[:5]
            print("Max_5 nmi: ", max_5_nmi)

            max_5_ssim = sorted(ssim_dict, key=ssim_dict.get, reverse=True)[:5]
            print("Max_5 ssim: ", max_5_ssim)


        for patient in patients_list:

            if similarity == "ssim":
                if patient in max_5_ssim:
                    moving_image_path = f"./TrainingData/{patient}/mr_bffe.mhd"
                    moving_image_seg_path = f"./TrainingData/{patient}/prostaat.mhd"
                    result_path_transf = f"./results/{patient_1}/{patient}/result.mhd"

                    # moving_image = get_array_from_filepath(moving_image_path)
                    # activate/deactivate registration.
                    if registration is True:
                        jacobian_determinant_path = elastix_registration(moving_image_path,
                                                                         moving_image_seg_path,
                                                                         fixed_image_path,
                                                                         ELASTIX_PATH,
                                                                         TRANSFORMIX_PATH,
                                                                         patient,
                                                                         patient_1)
                    evaluate_labels(fixed_image_seg_path, result_path_transf, patient_1, patient, similarity, path_for_score)
                    print("transformation of ", patient, patient_1, "is done")

            elif similarity == "nmi":
                if patient in max_5_nmi:
                    moving_image_path = f"./TrainingData/{patient}/mr_bffe.mhd"
                    moving_image_seg_path = f"./TrainingData/{patient}/prostaat.mhd"
                    result_path_transf = f"./results/{patient_1}/{patient}/result.mhd"

                    # moving_image = get_array_from_filepath(moving_image_path)
                    # activate/deactivate registration.
                    if registration is True:
                        jacobian_determinant_path = elastix_registration(moving_image_path,
                                                                         moving_image_seg_path,
                                                                         fixed_image_path,
                                                                         ELASTIX_PATH,
                                                                         TRANSFORMIX_PATH,
                                                                         patient,
                                                                         patient_1)
                    evaluate_labels(fixed_image_seg_path, result_path_transf, patient_1, patient, similarity, path_for_score)
                    print("transformation of ", patient, patient_1, "is done")

            else:
                moving_image_path = f"./TrainingData/{patient}/mr_bffe.mhd"
                moving_image_seg_path = f"./TrainingData/{patient}/prostaat.mhd"
                result_path_transf = f"./results/{patient_1}/{patient}/result.mhd"

                # moving_image = get_array_from_filepath(moving_image_path)
                # activate/deactivate registration.
                if registration is True:
                    jacobian_determinant_path = elastix_registration(moving_image_path,
                                                                     moving_image_seg_path,
                                                                     fixed_image_path,
                                                                     ELASTIX_PATH,
                                                                     TRANSFORMIX_PATH,
                                                                     patient,
                                                                     patient_1)
                evaluate_labels(fixed_image_seg_path, result_path_transf, patient_1, patient, path_for_score)
                print("transformation of ", patient, patient_1, "is done")

        nmi_dict_2 = {}
        ssim_dict_2 = {}

        path_to_save_ssim = path_for_score + '/' + patient_1 + '_after_reg_ssim.csv'
        path_to_save_nmi = path_for_score + '/' + patient_1 + '_after_reg_nmi.csv'

        #evaluate_similarity_after_reg
        for patient in patients_list:
            if similarity == "ssim":
                if patient in max_5_ssim:
                    moving_image_result_path = "./results/" + patient_1 + "/" + patient + "/result.2.mhd"
                    moving_image = get_array_from_filepath(moving_image_result_path)

                    ssim = ssim_inf(fixed_image, moving_image)
                    ssim_dict_2[patient] = ssim

            elif similarity == "nmi":
                if patient in max_5_nmi:
                    moving_image_result_path = "./results/" + patient_1 + "/" + patient + "/result.2.mhd"
                    moving_image = get_array_from_filepath(moving_image_result_path)

                    nmi_res = nmi(fixed_image, moving_image)
                    nmi_dict_2[patient] = nmi_res

            else:
                moving_image_result_path = "./results/" + patient_1 + "/" + patient + "/result.2.mhd"
                moving_image = get_array_from_filepath(moving_image_result_path)
                nmi_res = nmi(fixed_image, moving_image)
                nmi_dict_2[patient] = nmi_res

                ssim = ssim_inf(fixed_image, moving_image)
                ssim_dict_2[patient] = ssim

        if similarity == "ssim":
            if os.path.exists(path_to_save_ssim) is False:
                with open(path_to_save_ssim, 'a',
                          newline='') as f:
                    fieldnames = ['patient', 'ssim']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writeheader()
                    f.close()

            for w in sorted(ssim_dict_2, key=ssim_dict_2.get, reverse=True):
                to_write = {'patient': w, 'ssim': str(round(ssim_dict_2[w], 5))}

                with open(path_to_save_ssim, 'a', newline='') as f:
                    fieldnames = ['patient', 'ssim']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writerow(to_write)
                    f.close()
        elif similarity =="nmi":
            if os.path.exists(path_to_save_nmi) is False:
                with open(path_to_save_nmi, 'a',
                          newline='') as f:
                    fieldnames = ['patient', 'nmi']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writeheader()
                    f.close()

            for w in sorted(nmi_dict_2, key=nmi_dict_2.get, reverse=True):
                to_write = {'patient': w, 'nmi': str(float(nmi_dict_2[w].round(3)))}

                with open(path_to_save_nmi, 'a', newline='') as f:
                    fieldnames = ['patient', 'nmi']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writerow(to_write)
                    f.close()
        else:
            # both nmi and ssim are possible
            if os.path.exists(path_to_save_ssim) is False:
                with open(path_to_save_ssim, 'a',
                          newline='') as f:
                    fieldnames = ['patient', 'ssim']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writeheader()
                    f.close()

            for w in sorted(ssim_dict_2, key=ssim_dict_2.get, reverse=True):
                to_write = {'patient': w, 'ssim': str(round(ssim_dict_2[w], 5))}

                with open(path_to_save_ssim, 'a', newline='') as f:
                    fieldnames = ['patient', 'ssim']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writerow(to_write)
                    f.close()

            if os.path.exists(path_to_save_nmi) is False:
                with open(path_to_save_nmi, 'a',
                          newline='') as f:
                    fieldnames = ['patient', 'nmi']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writeheader()
                    f.close()

            for w in sorted(nmi_dict_2, key=nmi_dict_2.get, reverse=True):
                to_write = {'patient': w, 'nmi': str(round(nmi_dict_2[w], 5))}

                with open(path_to_save_nmi, 'a', newline='') as f:
                    fieldnames = ['patient', 'nmi']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writerow(to_write)
                    f.close()




if __name__ == "__main__":
    start_time = datetime.now()
    # Make a results directory if none exists
    if os.path.exists('results') is False:
        os.mkdir('results')

    check_folder = f"./results/Score_Rigid_Affine_Bspline"

    # if Score_Rigid_Affine_Bspline exists, delete it.
    try:
        shutil.rmtree(check_folder)
    except OSError:
        pass

    # create new empty folder Score_Rigid_Affine_Bspline
    if os.path.exists(check_folder) is False:
        os.mkdir(check_folder)

    patients_list_fixed = [f.name for f in os.scandir("./TrainingData") if f.is_dir()]

    run_pipe(patients_list_fixed[:5], similarity="ssim")

    end_time = datetime.now()
    print('Execution Time: {}'.format(end_time - start_time))

    # fixed_image = "p102"
    # fixed_image_seg_path = f"./TrainingData/{fixed_image}/prostaat.mhd"
    # fixed_image_seg = get_array_from_filepath(fixed_image_seg_path)
    #
    # visualise_results("p116",
    #                   fixed_image,
    #                   43, fixed_image_seg)
