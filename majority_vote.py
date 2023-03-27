from program_elastix import fast_scandir, get_array_from_filepath
import matplotlib.pyplot as plt
import numpy as np

def show_array(arr):
    '''
    This function plots all slices of an array of a .mhd file of a patiently in a
    convenient overview.

    Parameters
    ----------
    arr : Array (z,x,y)
        Array of the image.

    Returns
    -------
    None.

    '''
    plt.figure()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(arr.shape[0]):
        plt.subplot(10, 10, i+1)
        # Cutoff chosen manually:
        mask = (arr[i] > 0.001).astype(int)
        plt.imshow(mask)
        plt.axis('off')
    plt.show()
    
def majority_vote(pred_segs, 
                  automatic_threshold=-1):
    '''
    This function combines the labels of multiple registrations to an unseen
    patient (fixed image), according to the majority vote.
    When the automatic_threshold is left as -1, the true majority is used (e.g.
    6 when there are 11 patients). When given a parameter, this integer is used
    as the threshold.
    
    Important: pred_segs needs to be in binary values.

    Parameters
    ----------
    pred_segs : Array (patients,z,x,y)
        The arrays of images of labels. 
    automatic_threshold : int, optional
        When given a parameter, the majority vote is thresholded at this 
        value, otherwise the default value indicates the true majority. 
        The default is -1.

    Returns
    -------
    all_masks_combined : Array (z,y,x)
        The result of the combined segmentations in a 3D array.
    '''
    full_dimensions = pred_segs.shape
    all_masks_combined = np.zeros(full_dimensions[1:])
    
    for slice_nr in range(full_dimensions.shape[1]):
        all_slices = []
        for pat_nr in range(full_dimensions.shape[0]):
            mask = pred_segs[pat_nr,slice_nr,:,:]
            all_slices.append(mask)
        sum_mask = sum(all_slices)
    
        if automatic_threshold == -1:
            # Majority vote -> over half of votes needed
            threshold = all_masks_combined.shape[0] // 2 + 1
        else:
            threshold = automatic_threshold
        mv_mask = (sum_mask > threshold).astype(int)
        all_masks_combined[slice_nr,:,:] = mv_mask
    
    return all_masks_combined

def calculate_surface_per_slice(patient_array):
    '''
    This function takes a 3D array of patient segmentations (of binary values) 
    as input and returns a list of the surface (nr of pixels in the label)
    per slice.

    Parameters
    ----------
    patient_array : Array (z,x,y)
        3D Array of the segmentation images.

    Returns
    -------
    surface_nr : list of int
        The surface count per slice.
    '''
    surface_nr = []
    for slice_nr in range(patient_array.shape[0]):
        s = np.sum(np.asarray(patient_array[slice_nr,:,:]))
        surface_nr.append(s)
    return surface_nr

def average_surfaces(pred_segs, 
                     plot=True):
    '''
    This function calculates (and optionally plots) the average surface
    of a segmentation per slice, using the function 
    calculate_surface_per_slice().

    Parameters
    ----------
    patient_folders : Array (patients,z,x,y)
        3D Array of the segmentation images.
    plot : Bool, optional
        Whether the result should be plotted. The default is True.

    Returns
    -------
    all_surfaces : Array (patients,z)
        Surface of each patient per slice.
    avg_surfaces : Array (z,)
        Average surfaces per slice.
    '''
    all_surfaces = np.zeros((pred_segs.shape[:2]))
    for pat in range(pred_segs.shape[0]):
        surf = calculate_surface_per_slice(pred_segs[pat,:,:,:])
        surf_array = np.asarray(surf)
        all_surfaces[pat,:] = surf_array
    
    avg_surfaces = np.mean(all_surfaces, axis=0)
    
    if plot:
        plt.figure()
        for i in range(len(all_surfaces)):
            plt.plot(all_surfaces[i])
        plt.plot(avg_surfaces, linewidth=5, c='k')
        plt.title("Number of pixels per patient per slice")
        plt.show()
    
    return all_surfaces, avg_surfaces

def label_with_average_surface(pred_segs):
    '''
    This function returns the combination of several segmentation predictions,
    according to a majority vote, in which the threshold for the majority is
    decided by which threshold will be closest to the average surface of 
    all patient segmentations.

    Parameters
    ----------
    pred_segs : Array (patients,z,x,y)
        The arrays of images of labels. 

    Returns
    -------
    all_masks_combined : Array of int (0,1) (z,x,y)
        The result of the combined labels in a 3D Array.
    '''
    _, avg_surfaces = average_surfaces(pred_segs, False)
    
    all_masks_combined = np.zeros(pred_segs.shape[1:])
    for slice_nr in range(pred_segs.shape[1]):
        all_slices = []
        for pat in range(pred_segs.shape[0]):
            all_slices.append(pred_segs[pat,slice_nr,:,:])
        all_masks_combined[slice_nr,:,:] = sum(all_slices)
    
    mi = np.min(all_masks_combined).astype(int)
    ma = np.max(all_masks_combined).astype(int)
    for slice_nr in range(pred_segs.shape[1]):
        thresholded_surface_counts = []
        for thresh in range(mi, ma+1):
            current_slice = all_masks_combined[slice_nr,:,:]
            thresholded_slice = current_slice > thresh
            surface_count = np.sum(np.asarray(thresholded_slice))
            thresholded_surface_counts.append((abs(surface_count - avg_surfaces[slice_nr]), thresh))
        
        # Find surface count which is closest to the average value:
        best_threshold_value = min(thresholded_surface_counts, key = lambda t: t[0])[1]
        
        # Calculate this mask
        current_slice = all_masks_combined[slice_nr,:,:]
        final_thresholded_slice = (current_slice > best_threshold_value).astype(int)
        all_masks_combined[slice_nr,:,:] = final_thresholded_slice
            
    return all_masks_combined
    return thresholded_surface_counts


if __name__ == "__main__":
    # Prepare data:
    subfolders = fast_scandir('.\\results')
    pred_segmentations = np.zeros((len(subfolders),86,333,271))
    for i, s in enumerate(subfolders):
        pred_segmentations[i,:,:,:] = get_array_from_filepath(s + "\\result.mhd")
        pred_segmentations[:,:,:,:] = (pred_segmentations[:,:,:,:] > 0.001).astype(int)
        
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
    
