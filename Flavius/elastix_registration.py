import os
import elastix

def elastix_registration(fixed_path, fixed_segmented_path, moving_path,  ELASTIX_PATH,
                         TRANSFORMIX_PATH):
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

    # elastix b-spline multi resolution registration
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    el.register(
        fixed_image=fixed_path,
        moving_image=moving_path,
        parameters=[os.path.join('ImagesforPractical/ImagesforPractical/MR/', 'Parameters_BSpline.txt')],
        output_dir='results')

    # apply transform to segmented image for final result
    transform = elastix.TransformixInterface(parameters="./results/TransformParameters.0.txt",
                                       transformix_path=TRANSFORMIX_PATH)
    transform.transform_image(fixed_segmented_path, output_dir="./results")

    # Get the Jacobian matrix
    jacobian_matrix_path = transform.jacobian_matrix(output_dir="./results")

    # Get the Jacobian determinant
    jacobian_determinant_path = transform.jacobian_determinant(output_dir="./results")

    # Get the full deformation field
    deformation_field_path = transform.deformation_field(output_dir="./results")


    return jacobian_determinant_path


########################