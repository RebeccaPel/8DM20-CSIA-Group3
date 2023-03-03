import os
import elastix


def elastix_registration(train_patient, patient, parameter_path, ELASTIX_PATH,
                         TRANSFORMIX_PATH):
    """
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

    """
    # Perform_elastix_registration(fixed, fixed_segmentation, moving, ELASTIX_PATH, TRANSFORMIX_PATH)
    # here we need to take the paths:
    path_fixed = os.path.abspath(f"TrainingData/{train_patient}/mr_bffe.mhd")
    path_fixed_seg = os.path.abspath(f"TrainingData/{train_patient}/prostaat.mhd")
    path_moving = os.path.abspath(f"TrainingData/{patient}/mr_bffe.mhd")

    if os.path.exists(os.path.abspath(f"results/{train_patient}/{patient}")) is False:
        os.mkdir(os.path.abspath(f"results/{train_patient}/{patient}"))

    # elastix b-spline multi resolution registration
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    el.register(
        fixed_image=path_fixed,
        moving_image=path_moving,
        # parameters=[os.path.join('RegistrationParametersFiles/', 'Par0000affine.txt')],
        parameters=[parameter_path],
        output_dir=os.path.abspath(f"results/{train_patient}/{patient}")
    )

    # apply transform to segmented image for final result
    transform = elastix.TransformixInterface(parameters=os.path.abspath(f"results/{train_patient}/{patient}/TransformParameters.0.txt"),
                                             transformix_path=TRANSFORMIX_PATH)
    transform.transform_image(path_fixed_seg, output_dir=os.path.abspath(f"results/{train_patient}/{patient}"))

    # Get the Jacobian matrix
    jacobian_matrix_path = transform.jacobian_matrix(output_dir=os.path.abspath(f"results/{train_patient}/{patient}"))

    # Get the Jacobian determinant
    jacobian_determinant_path = transform.jacobian_determinant(output_dir=os.path.abspath(f"results/{train_patient}/{patient}"))

    # Get the full deformation field
    deformation_field_path = transform.deformation_field(output_dir=os.path.abspath(f"results/{train_patient}/{patient}"))

    return jacobian_determinant_path

