import numpy as np

def to_tensor_format(x: np.array, **kwargs) -> np.array:
    """transforms np array into tensor shate (c, h, w)

    Args:
        x (np.array): input np array

    Returns:
        np.array: np array in tensor format (c, h, w)
    """
    return x.transpose(2, 0, 1).astype('float32')
