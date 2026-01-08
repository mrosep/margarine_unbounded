"""Processing and transformation utilities for margarine_unbounded.

This module provides transformation functions for converting between different
parameter spaces used in the MAF training and sampling process.

Note: These functions are typically not called directly by users, but are used
internally by the MAF class.
"""

from tensorflow_probability import distributions as tfd
import tensorflow as tf
import random


@tf.function(jit_compile=True)
def _forward_transform(x, min=0, max=1):
    r"""Transform samples to standard normal via uniform CDF.

    Converts samples to uniform [0,1] using the CDF of a uniform distribution,
    then transforms to standard normal N(0,1) using the inverse CDF (quantile function).

    This is used internally during MAF construction but not typically during
    training/sampling in the unbounded version (which uses direct standardization).

    Args:
        x: Samples to be transformed
        min: Lower bound of uniform distribution (default: 0)
        max: Upper bound of uniform distribution (default: 1)

    Returns:
        Transformed samples in standard normal space
    """
    
    x = tfd.Uniform(min, max).cdf(x)
    x = tfd.Normal(0, 1).quantile(x)
    return x


@tf.function(jit_compile=True)
def _inverse_transform(x, min, max):
    r"""Inverse of _forward_transform.

    Transforms samples from standard normal space back to the original space
    via uniform distribution. Inverts the process in _forward_transform.

    Args:
        x: Samples in standard normal space
        min: Lower bound of target uniform distribution
        max: Upper bound of target uniform distribution

    Returns:
        Samples transformed back to original space
    """
    x = tfd.Normal(0, 1).cdf(x)
    x = tfd.Uniform(min, max).quantile(x)
    return x


def pure_tf_train_test_split(a, b, test_size=0.2):
    """Split data into training and testing sets using TensorFlow operations.

    TensorFlow-native implementation of train/test splitting, equivalent to
    sklearn.model_selection.train_test_split but works directly with TensorFlow tensors.

    Args:
        a: Tensor of samples to be split
        b: Tensor of weights corresponding to the samples
        test_size: Fraction of data to use for testing (default: 0.2)

    Returns:
        tuple: (a_train, a_test, b_train, b_test)
            - a_train: Training samples
            - a_test: Test samples
            - b_train: Training weights
            - b_test: Test weights
    """

    idx = random.sample(range(len(a)), int(len(a)*test_size))

    a_train = tf.gather(a,
                        tf.convert_to_tensor(
                            list(set(range(len(a))) - set(idx))))
    b_train = tf.gather(b,
                        tf.convert_to_tensor(
                            list(set(range(len(b))) - set(idx))))
    a_test = tf.gather(a, tf.convert_to_tensor(idx))
    b_test = tf.gather(b, tf.convert_to_tensor(idx))

    return a_train, a_test, b_train, b_test
