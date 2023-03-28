import os
import elastix
import shutil

def elastix_registration(moving_path, moving_segmented_path, fixed_path,  ELASTIX_PATH,
                         TRANSFORMIX_PATH, folder, patient_1):
    '''
    Performs b-spline registration according to the parameters in parameters_blspine.txt
    Saves resulting image, log-files and transformation field in the folder "./results"
    Names are [result.0.tiff, elastix.log, TransformParameters.0.txt]

    Parameters
    ----------
    fixed_path : fixed image file path (string)
    fixed_segmented_path: fixed image file segmented path (string)
    moving_path : moving image file path (string)
    ELASTIX_PATH : Elastix executable file path (string)
    TRANSFORMIX_PATH : Transformix executable file path (string)

    Returns
    -------
    logs
    '''
    if os.path.exists('results/' + patient_1 + "/" + folder):
        shutil.rmtree('results/' + patient_1 + "/" + folder)

    if not os.path.exists('results/' + patient_1 + "/" + folder):
        os.makedirs('results/' + patient_1 + "/" + folder)
    # Elastix b-spline multi resolution registration
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    el.register(
        fixed_image=fixed_path,
        moving_image=moving_path,
        # parameters=[os.path.join('RegistrationParametersFiles/', 'Parameters_BSpline.txt')],
		parameters=[os.path.join('RegistrationParametersFiles/', 'Parameters_Rigid_def.txt'),os.path.join('RegistrationParametersFiles/', 'Parameters_Affine_def.txt'), os.path.join('RegistrationParametersFiles/', 'Parameters_BSpline_def.txt')],
        output_dir="./results/" + patient_1 + "/" + folder)

    path_transform = "./results/" + patient_1 + "/" + folder + "/TransformParameters.2.txt"

    # Apply transform to segmented image for final result
    transform = elastix.TransformixInterface(parameters=path_transform,
                                             transformix_path=TRANSFORMIX_PATH)
    transform.transform_image(moving_segmented_path, output_dir="./results/" + patient_1 + "/" + folder)

    # # Get the Jacobian matrix
    # jacobian_matrix_path = transform.jacobian_matrix(output_dir="./results/" + patient_1 + "/" + folder)
    #
    # # Get the Jacobian determinant
    # jacobian_determinant_path = transform.jacobian_determinant(output_dir="./results/" + patient_1 + "/" + folder)

    jacobian_determinant_path = "ok"
    #
    # # Get the full deformation field
    # deformation_field_path = transform.deformation_field(output_dir="./results/" + patient_1 + "/" + folder)

    return jacobian_determinant_path