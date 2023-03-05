import matplotlib.pyplot as plt
import elastix
import os
import SimpleITK as sitk
from elastix_registration_rebecca import elastix_registration
import numpy as np

# make sure you select right your .exe paths
ELASTIX_PATH = os.path.join(
    r'C:/Users/20192157/OneDrive - TU Eindhoven/Documents/Master/8DM20 Capita selecta in medical image analysis/elastix-5.0.0-win64/elastix.exe')
TRANSFORMIX_PATH = os.path.join(
    r'C:/Users/20192157/OneDrive - TU Eindhoven/Documents/Master/8DM20 Capita selecta in medical image analysis/elastix-5.0.0-win64/transformix.exe')

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

if __name__ == '__main__':
    # Make a results directory if non exists
    if os.path.exists('results') is False:
        os.mkdir('results')
    patient_path = r'C:/Users/20192157/OneDrive - TU Eindhoven/Documents/Master/8DM20 Capita selecta in medical image analysis/TrainingData'
    patients = os.listdir(os.path.join(patient_path))

    dice_scores = {}

    Parameter_files = ["Par0001bspline04.txt", "Par0001bspline64.txt", "Parameters_BSpline.txt"]
    train = ['p102', 'p107', 'p108', 'p109', 'p115']

    for train_patient in train:
        if os.path.exists(f"results/{train_patient}") is False:
            os.mkdir(f"results/{train_patient}")

        for patient in patients:
            if patient[0] == 'p' and patient != train_patient:
                parameters = os.path.join('ImagesforPractical/ImagesforPractical/MR/', Parameter_files[2])

                # path_fixed = os.path.abspath(f"TrainingData/{train_patient}")
                # path_moving = os.path.abspath(f"TrainingData/{patient}")
                a, b = f"/{train_patient}", f"/{patient}"
                path_fixed = patient_path + a
                path_moving = patient_path + b

                fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(f"{path_fixed}\mr_bffe.mhd"))
                fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(f"{path_fixed}\prostaat.mhd"))
                moving_image = sitk.GetArrayFromImage(sitk.ReadImage(f"{path_moving}\mr_bffe.mhd"))
                moving_label = sitk.GetArrayFromImage(sitk.ReadImage(f"{path_moving}\prostaat.mhd"))

                # anytime you want to see visualize different, you can comment this line to avoid waiting too much time
                jacobian_determinant_path = elastix_registration(train_patient, patient, parameters,
                                                                 ELASTIX_PATH, TRANSFORMIX_PATH)

                transformed_bsplined_image = sitk.GetArrayFromImage(
                    sitk.ReadImage(os.path.join("results", train_patient, patient, "result.0.mhd")))
                transformed_bsplined_segmentation = sitk.GetArrayFromImage(
                    sitk.ReadImage(os.path.join("results", train_patient, patient, "result.mhd")))

                # Compute dice score between training patient and patient
                dice_score = 2 * np.sum(moving_label * transformed_bsplined_segmentation) / (
                        np.sum(moving_label) + np.sum(transformed_bsplined_segmentation))

                dice_scores[train_patient, patient] = dice_score
        print(dice_scores)
