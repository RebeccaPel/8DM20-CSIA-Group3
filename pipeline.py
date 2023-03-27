from datetime import datetime
import os
import shutil
import numpy as np
import SimpleITK as sitk

from program_elastix import run_pipe, get_array_from_filepath, visualise_results, fast_scandir
from majority_vote import average_surfaces, label_with_average_surface
from weighted_decision_fusing import weighted_decision_fusing, show_array


def main(majority_voting=True, weighted_decision=True):
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

    #settings
    similarity = "ssim" #'ssim' (structural similarity), 'nmi' (normalized mutual information), or None


    # load patients. now we load patients from TrainingData. Here we can configure to load the unseen patients.
    patients_list_fixed = [f.name for f in os.scandir("./TrainingData") if f.is_dir()]

    # run the pipe with 5 patients for speed (should be around 9 minutes).
    # Can be executed with 'ssim' (structural similarity), 'nmi' (normalized mutual information), or None
    # run_pipe(patients_list_fixed[:5], similarity=similarity)

    if majority_voting is True:

        subfolders =[f.path for f in os.scandir('.\\results') if f.is_dir()]
        saved_paths = []

        for x in patients_list_fixed:
            matching_1 = [s for s in subfolders if x in s]
            if len(matching_1) !=0:
                subfolders_2 = [f.path for f in os.scandir(matching_1[0]) if f.is_dir()]
                saved_paths.append(subfolders_2)


        for patient in saved_paths:
            # Run majority voting
            pred_segmentations = np.zeros((len(subfolders_2), 86, 333, 271))
            for i, s in enumerate(patient):
                pred_segmentations[i, :, :, :] = get_array_from_filepath(s + "\\result.mhd")
                pred_segmentations[:, :, :, :] = (pred_segmentations[:, :, :, :] > 0.001).astype(int)

            # Visulatisation of patient masks:
            # for a in range(pred_segmentations.shape[0]):
            #     show_array(pred_segmentations[a,:,:,:])

            # Simple majority vote:
            # mv_mask = majority_vote(pred_segmentations, 3)
            # show_array(mv_mask)

            # Surface calculations:
            # one_patient_surfaces = calculate_surface_per_slice(pred_segmentations[0,:,:,:])
            all_surfaces, avg_surfaces = average_surfaces(pred_segmentations, plot=False)

            # Majority vote with the average surface goal:
            mv_surface_threshold = label_with_average_surface(pred_segmentations)
            # show_array(mv_surface_threshold)

    if weighted_decision is True:

        subfolders = [f.path for f in os.scandir('.\\results') if f.is_dir()]
        saved_paths = []
        saved_fixed_patients = []

        for x in patients_list_fixed:
            matching_1 = [s for s in subfolders if x in s]
            if len(matching_1) != 0:
                saved_fixed_patients.append(matching_1[0].strip('.\\results\\'))
                subfolders_2 = [f.path for f in os.scandir(matching_1[0]) if f.is_dir()]
                saved_paths.append(subfolders_2)

        for patient in saved_paths:
            registrations, pred_segmentations = (np.zeros((len(subfolders_2), 86, 333, 271)),
                                                     np.zeros((len(subfolders_2), 86, 333, 271)))
            for i, s in enumerate(patient):
                pred_segmentations[i, :, :, :] = get_array_from_filepath(s + "\\result.mhd")
                registrations[i, :, :, :] = get_array_from_filepath(s + "\\result.2.mhd") # here has to be 2 if you want to take the output from Bspline

        for x in saved_fixed_patients:
            # Unseen image is the patient to which the images above are registrated
            unseen_img = sitk.ReadImage(f"./TrainingData/{x}/mr_bffe.mhd")
            unseen_pat = sitk.GetArrayViewFromImage(unseen_img)

            # Run code
            outc = weighted_decision_fusing(np.asarray(unseen_pat), np.asarray(registrations), np.asarray(pred_segmentations))

            show_array(outc)


# only visualization
    #
    # fixed_image = "p102"
    # fixed_image_seg_path = f"./TrainingData/{fixed_image}/prostaat.mhd"
    # fixed_image_seg = get_array_from_filepath(fixed_image_seg_path)
    #
    # visualise_results("p116",
    #                   fixed_image,
    #                   43, fixed_image_seg)

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print('Execution Time: {}'.format(end_time - start_time))

