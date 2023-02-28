import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# pat_lst = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119',
#            'p120', 'p125', 'p127', 'p128', 'p129', 'p133', 'p135']

# Take 5 images to create the average masks (why 5? -> Is there an optimal?)
avg_pats = ['p102', 'p107', 'p108', 'p109', 'p115']
# Other patients are 'unseen' -> used to calculate performance of average mask
unseen_pats = ['p116', 'p117', 'p119', 'p120', 'p125', 'p127', 'p128', 'p129', 'p133', 'p135']

def create_average_mask(patients, show = True, normalize = False, cut_off_value = 0.5):
    avg_mask = np.zeros((86,333,271))
    for pat in patients:
        # pat_path = f"./TrainingData/{pat}/prostaat.mhd"
        pat_path = f"C:/Users/20192157/OneDrive - TU Eindhoven/Documents/Master/8DM20 Capita selecta in medical image analysis/TrainingData/{pat}/prostaat.mhd"
        itk_image = sitk.ReadImage(pat_path)
        img_array = sitk.GetArrayViewFromImage(itk_image)
        for i in range(img_array.shape[0]):
            avg_mask[i] += img_array[i]

    if normalize:
        avg_mask = avg_mask / len(patients)
        avg_mask[avg_mask < cut_off_value] = 0
        avg_mask[avg_mask >= cut_off_value] = 1

    if show:
        plt.figure()
        plt.gray()
        plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
        for i in range(avg_mask.shape[0]):
            plt.subplot(10, 10, i + 1)
            plt.imshow(avg_mask[i])
            plt.axis('off')
        plt.show()

    return avg_mask

# avg_mask_pat = create_average_mask(avg_pats)
normalized_avg_mask = create_average_mask(avg_pats, normalize = True, show = True)

# Question: The slice 'surface' is now always decreased, how do we solve this?
# Find the cut-off value where the new surface is similar to the original?

# Have two dice score methods -> 1 for the average masks, 1 for two 'hard' masks

# def calculate_dice_score(mask_1, mask_2):
    



























