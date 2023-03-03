import matplotlib.pyplot as plt
import elastix
import os
import SimpleITK as sitk
from elastix_registration import elastix_registration
import numpy as np

# make sure you select right your .exe paths
ELASTIX_PATH = os.path.join(
    r'C:\Users\anahs\Documents\elastix-5.1.0-win64\elastix.exe')
TRANSFORMIX_PATH = os.path.join(
    r'C:\Users\anahs\Documents\elastix-5.1.0-win64\transformix.exe')

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

if __name__ == '__main__':
    # Make a results directory if non exists
    if os.path.exists('results') is False:
        os.mkdir('results')
    patients = os.listdir(os.path.join("./TrainingData/"))

    dice_scores = {}

    Parameter_files = ["Par0001bspline04.txt", "Par0001bspline64.txt", "Parameters_BSpline.txt"]
    train = ['p102', 'p107', 'p108', 'p109', 'p115']  # , 'p108', 'p109', 'p115'

    for train_patient in train:
        if os.path.exists(f"results/{train_patient}") is False:
            os.mkdir(f"results/{train_patient}")

        for patient in patients:
            if patient[0] == 'p' and patient != train_patient:

                parameters = os.path.join('ImagesforPractical/ImagesforPractical/MR/', Parameter_files[2])

                path_fixed = os.path.abspath(f"TrainingData/{train_patient}")
                path_moving = os.path.abspath(f"TrainingData/{patient}")

                fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(f"{path_fixed}/mr_bffe.mhd"))
                fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(f"{path_fixed}/prostaat.mhd"))
                moving_image = sitk.GetArrayFromImage(sitk.ReadImage(f"{path_moving}/mr_bffe.mhd"))
                moving_label = sitk.GetArrayFromImage(sitk.ReadImage(f"{path_moving}/prostaat.mhd"))

                # anytime you want to see visualize different, you can comment this line to avoid waiting too much time
                jacobian_determinant_path = elastix_registration(train_patient, patient, parameters,
                                                                 ELASTIX_PATH, TRANSFORMIX_PATH)

                transformed_bsplined_image = sitk.GetArrayFromImage(
                    sitk.ReadImage(os.path.join("results", train_patient, patient, "result.0.mhd")))
                transformed_bsplined_segmentation = sitk.GetArrayFromImage(
                    sitk.ReadImage(os.path.join("results", train_patient, patient, "result.mhd")))

                """
                slice_nr = 30
                # display images and graphs
                fig, axarr = plt.subplots(1, 5)
                plt.suptitle('Slice number: ' + str(slice_nr), fontsize=16)
                axarr[0].imshow(fixed_image[slice_nr, :, :], cmap='gray')
                axarr[0].imshow(fixed_label[slice_nr, :, :], alpha=0.5 * (fixed_label[slice_nr, :, :] > 0),
                                cmap='Purples')
                axarr[0].set_title('Fixed image')
                axarr[1].imshow(moving_image[slice_nr, :, :], cmap='gray')
                axarr[1].set_title('Moving image')
                axarr[1].imshow(transformed_bsplined_segmentation[slice_nr, :, :],
                                alpha=0.8 * (transformed_bsplined_segmentation[slice_nr, :, :] > 0), cmap='Purples')
                axarr[2].imshow(transformed_bsplined_segmentation[slice_nr, :, :], cmap='gray')
                axarr[2].set_title('Resulted B-splined segmentation')
                axarr[3].imshow(transformed_bsplined_image[slice_nr, :, :], cmap='gray')
                axarr[3].set_title('Resulted B-splined image')

                plt.tight_layout()

                fig2, axarr = plt.subplots(1, 1)
                # Open the logfile into the dictionary log
                for i in range(4):
                    log_path = os.path.join("results", train_patient, patient, 'IterationInfo.0.R{}.txt'.format(i))
                    log = elastix.logfile(log_path)
                    # Plot the 'metric' against the iteration number 'itnr'
                    plt.plot(log['itnr'], log['metric'])
                plt.legend(['Resolution {}'.format(i) for i in range(5)])

                plt.show()
                """
                # Compute dice score between training patient and patient
                dice_score = 2 * np.sum(moving_label * transformed_bsplined_segmentation) / (
                        np.sum(moving_label) + np.sum(transformed_bsplined_segmentation))

                dice_scores[train_patient, patient] = dice_score
