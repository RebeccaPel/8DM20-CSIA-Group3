
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def calculate_dice(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    else:
        intersection = np.logical_and(im1, im2)
        if im1.sum() == 0 and im2.sum() == 0:
            # No prostate in both images
            dice = 1.0
        elif im1.sum() == 0 or im2.sum() == 0:
            # Prostate in only ONE image
            dice = np.nan
        else:
            # Prostate in both -> Dice score
            dice =  2 * intersection.sum() / (im1.sum() + im2.sum())
    return dice

def scores_3d(filepath1, filepath2):
    itk_image1 = sitk.ReadImage(filepath1)
    im1_stack = sitk.GetArrayViewFromImage(itk_image1)
    itk_image2 = sitk.ReadImage(filepath2)
    im2_stack = sitk.GetArrayViewFromImage(itk_image2)
    if im1_stack.shape != im2_stack.shape:
        raise ValueError("Shape mismatch: img1_stack and img2_stack must have the same number of images.")
    else:
        dice_list = []
        for z in range(im1_stack.shape[0]):
            dice_list.append(calculate_dice(im1_stack[z], im2_stack[z]))
    return dice_list

# # Path to patient 1 
# p_1 = r"C:\Users\20192157\OneDrive - TU Eindhoven\Documents\Master\8DM20 Capita selecta in medical image analysis\TrainingData\p102\prostaat.mhd"
# # Path to patient 2
# p_2 = r"C:\Users\20192157\OneDrive - TU Eindhoven\Documents\Master\8DM20 Capita selecta in medical image analysis\TrainingData\p107\prostaat.mhd"
# ds = scores_3d(p_1, p_2)
# # plt.plot(ds)

def compare_all_patients(pat_list, show = True):
    scores_lists = []
    for pat1 in pat_list:
        row = []
        path1 = f"C:/Users/20192157/OneDrive - TU Eindhoven/Documents/Master/8DM20 Capita selecta in medical image analysis/TrainingData/{pat1}/prostaat.mhd"
        for pat2 in pat_list:
            path2 = f"C:/Users/20192157/OneDrive - TU Eindhoven/Documents/Master/8DM20 Capita selecta in medical image analysis/TrainingData/{pat2}/prostaat.mhd"
            ds = scores_3d(path1, path2)
            row.append(ds)
        scores_lists.append(row)
    
    if show:
        fig, ax = plt.subplots(len(pat_list), 
                               len(pat_list), 
                               sharex = True, sharey = True)
        for i, r in enumerate(scores_lists):
            for j, c in enumerate(r):
                ax[i][j].plot(c)
        plt.show()
    return scores_lists


pat_lst = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 
           'p120', 'p125', 'p127', 'p128', 'p129', 'p133', 'p135']
all_scores = compare_all_patients(pat_lst)









