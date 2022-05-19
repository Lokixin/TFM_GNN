import pathlib
import numpy as np
import scipy.io as sio


def __format_path(path_to_file: str) -> pathlib.Path:
    """Converts the path to a file to the corresponding
    OS format.

    Args:
        path_to_file (str): Path to desired file

    Returns:
        format_path (pathlib.Path): Full path to the file with
        the propper format depending on the OS being used.
    """
    format_path = pathlib.Path(path_to_file).resolve()
    return format_path
    
    
def read_record(path_to_file: str) -> np.ndarray:
    """Reads the content from a .mat file and returns
    it as Python array

    Args:
        path_to_file (str): Full Path to the .mat file

    Returns:
        np.ndarray: Array containing the data stored at
        the mat file
    """
    record_path = __format_path(path_to_file)
    record_contents = sio.loadmat(record_path)["EEG"]
    return record_contents