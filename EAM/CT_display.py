import os
import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path


def load_dicom_series(dicom_folder):
    """
    Load a series of DICOM images from a folder, even if files have no extensions.

    Parameters:
        dicom_folder (str): Path to the folder containing DICOM files.

    Returns:
        numpy.ndarray: 3D array of stacked image slices.
        pydicom.Dataset: DICOM metadata from the first slice.
    """
    # Get all files in the folder (even without extensions)
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder)]

    # Try reading all files as DICOM
    slices = []
    for file in dicom_files:
        try:
            dcm = pydicom.dcmread(file)
            slices.append(dcm)
        except:
            continue  # Ignore files that are not DICOM

    if not slices:
        raise ValueError("No valid DICOM images found!")

    # Sort slices by Z-axis position (ensures correct 3D stacking)
    slices.sort(key=lambda x: getattr(x, "ImagePositionPatient", [0, 0, float("inf")])[2])

    # Stack image slices into a 3D volume
    volume = np.stack([s.pixel_array for s in slices])

    return volume, slices[0]


def dicom_to_nifti(dicom_folder, output_path):
    """
    Convert a folder of DICOM images to a single NIfTI (.nii.gz) file using SimpleITK.

    SimpleITK handles DICOM orientation, spacing, and metadata automatically,
    preserving the correct anatomical orientation.

    Parameters:
    -----------
    dicom_folder : str or Path
        Path to the folder containing DICOM files
    output_path : str or Path
        Path for the output .nii.gz file

    Returns:
    --------
    str : Path to the created NIfTI file
    """
    dicom_folder = str(Path(dicom_folder))

    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)

    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_folder}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Ensure output path has .nii.gz extension
    output_path = str(output_path)
    if not output_path.endswith('.nii.gz'):
        output_path += '.nii.gz'

    # Write as NIfTI
    sitk.WriteImage(image, output_path)

    print(f"Successfully converted {len(dicom_names)} DICOM slices to {output_path}")
    print(f"Image size: {image.GetSize()}")
    print(f"Image spacing: {image.GetSpacing()}")
    print(f"Image origin: {image.GetOrigin()}")

    return output_path


def dicom_to_nifti_multi_series(root_folder, output_folder):
    """
    Recursively search for DICOM files in all subfolders and convert to NIfTI.

    For each folder containing DICOM files, the output filename is constructed as:
    {folder_name_above_DICOM}_{series_id}.nii.gz

    where folder_name_above_DICOM is the name of the folder directly above
    any folder named "DICOM" in the path (or the immediate parent if no "DICOM" folder exists).

    Parameters:
    -----------
    root_folder : str or Path
        Root directory to start searching for DICOM files
    output_folder : str or Path
        Directory where NIfTI files will be saved

    Returns:
    --------
    list : Paths to all created NIfTI files
    """
    root_folder = Path(root_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_files = []
    reader = sitk.ImageSeriesReader()

    # Walk through all subdirectories
    for current_dir in root_folder.rglob('*'):
        if not current_dir.is_dir():
            continue

        # Check if this directory contains DICOM files
        series_ids = reader.GetGDCMSeriesIDs(str(current_dir))

        if not series_ids:
            continue

        # Find the folder name to use for output
        # Look for "DICOM" in the path and get the folder directly above it
        path_parts = current_dir.parts
        folder_name = current_dir.name  # default to immediate parent

        for i, part in enumerate(path_parts):
            if part.upper() == "DICOM" and i > 0:
                folder_name = path_parts[i - 1]
                break

        # Convert each series in this directory
        for series_id in series_ids:
            try:
                # Clean series_id for use in filename (remove problematic characters)
                clean_series_id = series_id.replace('/', '_').replace('\\', '_')
                output_filename = f"{folder_name}_{clean_series_id}.nii.gz"
                output_path = output_folder / output_filename

                # Use the existing conversion function
                dicom_to_nifti(str(current_dir), str(output_path))
                output_files.append(str(output_path))

            except Exception as e:
                print(f"Error converting series {series_id} in {current_dir}: {e}")
                continue

    print(f"\nTotal: Converted {len(output_files)} series to NIfTI format")
    return output_files


def list_dicom_series(dicom_dir):
    """
    List all DICOM series in a directory with metadata.

    Parameters:
    -----------
    dicom_dir : str or Path
        Path to directory containing DICOM files

    Returns:
    --------
    list : List of dictionaries containing series information
    """

    dicom_dir = Path(dicom_dir)

    if not dicom_dir.exists():
        raise FileNotFoundError(f"Directory {dicom_dir} does not exist")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    series_info = []

    for i, series_id in enumerate(series_ids):
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)

        # Read just the first file to get metadata
        if dicom_names:
            reader_single = sitk.ImageFileReader()
            reader_single.SetFileName(dicom_names[0])
            reader_single.LoadPrivateTagsOn()
            reader_single.ReadImageInformation()

            # Try to get series description
            try:
                series_description = reader_single.GetMetaData("0008|103e")
            except:
                series_description = "Unknown"

            # Try to get modality
            try:
                modality = reader_single.GetMetaData("0008|0060")
            except:
                modality = "Unknown"

            series_info.append({
                'index': i,
                'series_id': series_id,
                'num_files': len(dicom_names),
                'series_description': series_description,
                'modality': modality
            })

    print(series_info)
    return series_info


# Path to the parent directory that contains multiple DICOM series folders
test_dir = "C:/Users/steph/Downloads/Series0010"
sort_dir = "C:/Users/steph/Documents/UNC Cardiac Imaging/CT Data/Ablationsx2/35595/DICOM/0000E9E7/AAFDD708/AA8FC687"
dicom_dir = "C:/Users/steph/Documents/UNC Cardiac Imaging/CT Data/raw/1L/DICOM/00000227/AACC8AC4/AA3ED952/"
nifti_path = "C:/Users/steph/Documents/UNC Cardiac Imaging/CT Data/raw_nii/"

dicom_to_nifti_multi_series(dicom_dir, nifti_path)
