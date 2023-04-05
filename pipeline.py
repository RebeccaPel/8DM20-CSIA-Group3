from datetime import datetime
import os
import shutil
import numpy as np
import SimpleITK as sitk
from datetime import datetime
import csv

from program_elastix import run_pipe, get_array_from_filepath, visualise_results, fast_scandir, evaluate_labels_on_images
from majority_vote import average_surfaces, label_with_average_surface
from weighted_decision_fusing import weighted_decision_fusing, show_array


def main(majority_voting=True, weighted_decision=True, top_var=1):
    # Make a results directory if none exists
    if os.path.exists('results') is False:
        os.mkdir('results')

    var_important = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    #settings
    similarity = "None" #'ssim' (structural similarity), 'nmi' (normalized mutual information), or "None"
    #if similarity is None, top will be ignored and every fixed image will be registered to all other moving images.
    top = 1

    check_folder = f"./results/Score_" + var_important + "_top" + str(top) + "_" + similarity

    # create new empty folder Score_Rigid_Affine_Bspline
    if os.path.exists(check_folder) is False:
        os.mkdir(check_folder)

    # create new empty folder Score_Rigid_Affine_Bspline
    if os.path.exists(check_folder + "/weighted_segmentations/") is False:
        os.mkdir(check_folder + "/weighted_segmentations/")

    # create new empty folder Score_Rigid_Affine_Bspline
    if os.path.exists(check_folder + "/majority_voting/") is False:
        os.mkdir(check_folder + "/majority_voting/")


    # load patients. now we load patients from TrainingData. Here we can configure to load the unseen patients.
    patients_list_fixed = [f.name for f in os.scandir("./ValidatieData") if f.is_dir()]
    patients_list_fixed = patients_list_fixed
    # run the pipe with 5 patients (patients_list_fixed[:5]) for speed (should be around 7 minutes).
    # Can be executed with 'ssim' (structural similarity), 'nmi' (normalized mutual information), or None
    # set registration to False if you want to only compute metrics of already registered.
    # if you run with similarity None, all fixed images (14) will be registered to the actual one, (see next line)
    # then you can also change patients_list_fixed[:5] to patients_list_fixed, this will register all the other 14 moving images to the actual fixed image
    # run_pipe(patients_list_fixed, similarity=similarity, registration=True, path_for_score=check_folder, top=top)

    patients_moving_images = [f.name for f in os.scandir("./TrainingData") if f.is_dir()]

    if majority_voting is True:

        subfolders =[f.path for f in os.scandir('.\\results') if f.is_dir()]
        saved_paths = []
        saved_fixed_patients = []

        for x in patients_list_fixed:
            matching_1 = [s for s in subfolders if x in s]
            saved_fixed_patients.append(matching_1[0].strip('.\\results\\'))
            if len(matching_1) != 0:
                subfolders_2 = [f.path for f in os.scandir(matching_1[0]) if f.is_dir()]
                saved_paths.append(subfolders_2)

        # for i, x in enumerate(patients_moving_images):
        # matching_1 = [s for s in saved_paths if x in s]
        # path_to_save = check_folder + '/' + 'metrics_majority_voting.csv'

        # if os.path.exists(path_to_save) is True:
        #     os.remove(path_to_save)



        for x,patient in enumerate(saved_paths):
            pred_segmentations = np.zeros((len(patients_moving_images), 86, 333, 271))
            for i, s in enumerate(patient):
                # Run majority voting
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

            # Unseen image seg is the patient to which the images above are registrated
            # unseen_pat_seg = get_array_from_filepath(f"./TrainingData/{saved_fixed_patients[x]}/prostaat.mhd")

            # Majority vote with the average surface goal:
            mv_surface_threshold = label_with_average_surface(pred_segmentations)
            # show_array(mv_surface_threshold)

            # evaluate_labels_on_images(unseen_pat_seg, mv_surface_threshold, saved_fixed_patients[x], "majority_voting",
            #                           similarity="None", path=path_to_save)

            # Convert the array to a SimpleITK image
            write_image = sitk.GetImageFromArray(mv_surface_threshold)

            # Save the image to an MHD file without setting origin, spacing, and direction
            sitk.WriteImage(write_image, check_folder +"/majority_voting/"+saved_fixed_patients[x]+"_output.mhd")


    if weighted_decision is True:

        subfolders =[f.path for f in os.scandir('.\\results') if f.is_dir()]
        saved_paths = []
        saved_fixed_patients = []

        for i,x in enumerate(patients_list_fixed):
            matching_1 = [s for s in subfolders if x in s]
            saved_fixed_patients.append(matching_1[0].strip('.\\results\\'))
            if len(matching_1) != 0:
                subfolders_2 = [f.path for f in os.scandir(matching_1[0]) if f.is_dir()]
                saved_paths.append(subfolders_2)


        # path_to_save = check_folder + '/' + 'metrics_weighted_decision_fusing.csv'
        #
        # if os.path.exists(path_to_save) is True:
        #     os.remove(path_to_save)

        for x,patient in enumerate(saved_paths):
            registrations, pred_segmentations = (np.zeros((len(patients_moving_images), 86, 333, 271)),
                                                     np.zeros((len(patients_moving_images), 86, 333, 271)))
            for i, s in enumerate(patient):
                pred_segmentations[i, :, :, :] = get_array_from_filepath(s + "\\result.mhd")
                registrations[i, :, :, :] = get_array_from_filepath(s + "\\result.2.mhd") # here has to be 2 if you want to take the output from Bspline


            # Unseen image is the patient to which the images above are registrated
            unseen_pat = get_array_from_filepath(f"./ValidatieData/{saved_fixed_patients[x]}/mr_bffe.mhd")
            # unseen_pat_seg = get_array_from_filepath(f"./TrainingData/{saved_fixed_patients[x]}/prostaat.mhd")

            # Run code
            outc = weighted_decision_fusing(np.asarray(unseen_pat), np.asarray(registrations), np.asarray(pred_segmentations))

            # evaluate_labels_on_images(unseen_pat_seg, outc, saved_fixed_patients[x], "weighted_decision_fusing", similarity="None", path=path_to_save)

            # Convert the array to a SimpleITK image
            write_image = sitk.GetImageFromArray(outc)

            # Save the image to an MHD file without setting origin, spacing, and direction
            sitk.WriteImage(write_image, check_folder +"/weighted_segmentations/"+saved_fixed_patients[x]+"_output.mhd")

    return check_folder


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
    # activate/deactivate tops below
    activate_tops = False

    if activate_tops is True:
        # set tops from 1 to 7.
        for i in range(1,8):
            start_time = datetime.now()
            check_folder = main(majority_voting=True, weighted_decision=True, top_var=i)
            end_time = datetime.now()
            execution_time = end_time - start_time
            print('Execution Time: {}'.format(execution_time))

            path_to_save_time = check_folder + '/' + 'execution_time.csv'

            to_write = ['execution_time', str(execution_time)]

            with open(path_to_save_time, 'a', newline='') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerows([to_write])
                f.close()
    else:
        start_time = datetime.now()
        check_folder = main(majority_voting=True, weighted_decision=True, top_var=1)
        end_time = datetime.now()
        execution_time = end_time - start_time
        print('Execution Time: {}'.format(execution_time))

        path_to_save_time = check_folder + '/' + 'execution_time.csv'

        to_write = ['execution_time', str(execution_time)]

        with open(path_to_save_time, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows([to_write])
            f.close()

