from skimage.metrics import normalized_mutual_information, structural_similarity

def nmi(image1, image2):
    bins = 256
    nmi = normalized_mutual_information(image1, image2, bins=bins)

    return nmi

def ssim_inf(image1,image2):

    dist = abs(image1.max() - image1.min())
    ssim = structural_similarity(image1, image2, win_size=9, data_range= dist)

    return ssim