
import SimpleITK as sitk
import matplotlib.pyplot as plt

pat_lst = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 
           'p120', 'p125', 'p127', 'p128', 'p129', 'p133', 'p135']

# itk_image = sitk.ReadImage("./TrainingData/p102/prostaat.mhd")
itk_image = sitk.ReadImage("./TrainingData/p102/mr_bffe.mhd")
image_array = sitk.GetArrayViewFromImage(itk_image)

# print the image's dimensions
# print(image_array.shape)

def show_mhd(filepath):
    itk_image = sitk.ReadImage(filepath)
    image_array = sitk.GetArrayViewFromImage(itk_image)
    plt.figure()
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(image_array.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(image_array[i])
        plt.axis('off')
    plt.show()

show_mhd("./TrainingData/p102/mr_bffe.mhd")
# show_mhd("./TrainingData/p102/prostaat.mhd")
    
def get_files_information(patient_list, prostaat = True):
    
    images_information = {}
    
    for pat in patient_list:
        
        if prostaat:
            pat_path = f"./TrainingData/{pat}/prostaat.mhd"
        else:
            pat_path = f"./TrainingData/{pat}/mr_bffe.mhd"
        
        itk_image = sitk.ReadImage(pat_path)
        img_array = sitk.GetArrayViewFromImage(itk_image)
        images_information[pat] = image_array.shape
        
    return images_information
        
# prostaat_img_info = get_files_information(pat_lst)
# mr_img_info = get_files_information(pat_lst, False)