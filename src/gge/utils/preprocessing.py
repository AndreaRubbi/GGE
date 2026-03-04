from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd
from scipy import sparse

def normalize_data(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Normalize the gene expression data to have zero mean and unit variance.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        The gene expression data to normalize.

    Returns
    -------
    np.ndarray or pd.DataFrame
        The normalized gene expression data.
    """
    if isinstance(data, pd.DataFrame):
        return (data - data.mean()) / data.std()
    elif isinstance(data, np.ndarray):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    else:
        raise TypeError("Input data must be a numpy array or a pandas DataFrame.")

def log_transform(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Apply log transformation to the gene expression data.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        The gene expression data to transform.

    Returns
    -------
    np.ndarray or pd.DataFrame
        The log-transformed gene expression data.
    """
    if isinstance(data, pd.DataFrame):
        return np.log1p(data)
    elif isinstance(data, np.ndarray):
        return np.log1p(data)
    else:
        raise TypeError("Input data must be a numpy array or a pandas DataFrame.")

def scale_data(data: Union[np.ndarray, pd.DataFrame], min_val: float = 0, max_val: float = 1) -> Union[np.ndarray, pd.DataFrame]:
    """
    Scale the gene expression data to a specified range.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        The gene expression data to scale.
    min_val : float
        The minimum value of the scaled data.
    max_val : float
        The maximum value of the scaled data.

    Returns
    -------
    np.ndarray or pd.DataFrame
        The scaled gene expression data.
    """
    if isinstance(data, pd.DataFrame):
        return (data - data.min()) / (data.max() - data.min()) * (max_val - min_val) + min_val
    elif isinstance(data, np.ndarray):
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)) * (max_val - min_val) + min_val
    else:
        raise TypeError("Input data must be a numpy array or a pandas DataFrame.")

def to_dense(X):
    """
    Safely convert a matrix-like to a dense numpy array without copying if already dense.
    Handles scipy.sparse matrices.
    """
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)