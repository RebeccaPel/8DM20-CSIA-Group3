from program_elastix import fast_scandir, get_array_from_filepath
import numpy as np
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
import matplotlib.pyplot as plt
from majority_vote import show_array

def weighted_decision_fusing(fixed_im, 
                             reg_ims, 
                             pred_segs):
    '''
    This function combines the labels of multiple registrations to an unseen
    patient (fixed image), according to the absolute differences in the 
    registrations. Based on the technique described by Isgum et al. (2009)
    in "Multi-Atlas-Based Segmentation With Local Decision Fusion—Application 
    to Cardiac and Aortic Segmentation in CT Scans".    

    Parameters
    ----------
    fixed_im : Array of float (z,x,y)
        3D Array of (MR) images of fixed patient.
    reg_ims : Array of float (nr_patients,z,x,y)
        4D Array of all registrations to the fixed images of multiple patients.
    pred_segs : Array of float (nr_patients,z,x,y)
        4D Array of the transformed labels (segmentations) belonging to reg_ims.

    Returns
    -------
    Array of int (0,1) (z,x,y)
        The result of the weighted label fusion in a 3D Array.
    '''
    # Constants
    SCALE1 = 0.5
    SCALE2 = 0.5
    EPSILON = 0.001
    
    assert fixed_im.shape == reg_ims[0,:,:,:].shape, "Fixed image dimensions do not match registered images."
    assert fixed_im.shape == pred_segs[0,:,:,:].shape, "Fixed image dimensions do not match predicted segmentation images."
    
    # Calculate difference images
    diff_ims = np.zeros(reg_ims.shape, dtype = reg_ims.dtype)
    for i in range(reg_ims.shape[0]):
        diff_ims[i] = np.abs(reg_ims[i] - fixed_im)
    
    # Blur all volumes with gaussian kernel, then calculate weights
    for i in range(reg_ims.shape[0]):
        diff_ims[i] = gaussian_filter(diff_ims[i], sigma=SCALE1)
    weights = 1 / (diff_ims + EPSILON)
    
    # Calculate weighted average of the transformed binary images
    norm_factor = 1 / (np.sum(weights, axis=0))    
    fused_pred_segs = norm_factor * np.sum(weights*pred_segs, axis=0)
    
    # Lastly, blur predictions with a second Gaussian filter 
    fused_pred_segs = gaussian_filter(fused_pred_segs, sigma=SCALE2)
    
    return (fused_pred_segs > 0.5).astype(int) # Treshold for final segmentation

if __name__ == "__main__":
    # Prepare data:
    subfolders = fast_scandir('.\\results')
    registrations, pred_segmentations = (np.zeros((len(subfolders),86,333,271)), 
                                         np.zeros((len(subfolders),86,333,271)))
    for i, s in enumerate(subfolders):
        pred_segmentations[i,:,:,:] = get_array_from_filepath(s + "\\result.mhd")
        registrations[i,:,:,:] = get_array_from_filepath(s + "\\result.0.mhd")
    
    # Unseen image is the patient to which the images above are registrated    
    unseen_img = sitk.ReadImage("./TrainingData/p102/mr_bffe.mhd")
    unseen_pat = sitk.GetArrayViewFromImage(unseen_img)   
    
    # Run code
    outc = weighted_decision_fusing(np.asarray(unseen_pat), 
                                    np.asarray(registrations),
                                    np.asarray(pred_segmentations))
    
    show_array(outc)
