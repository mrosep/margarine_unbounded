"""Masked Autoregressive Flow (MAF) implementation for unbounded density estimation.

This module provides the MAF class for learning probability distributions using
normalizing flows. This is part of margarine_unbounded, a fork of the original
margarine package by Harry T. J. Bevins.

Key differences from original margarine:
    - Removes implicit parameter bounds during training
    - Uses automatic standardization (mean/std) instead of min/max normalization
    - Provides .quantile() method for clean uniform->physical transforms
    - Designed for use with nested sampling and posterior repartitioning

Original margarine: https://github.com/htjb/margarine
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow_probability import (bijectors as tfb, distributions as tfd)
from .processing import pure_tf_train_test_split
import numpy as np
import tqdm
import warnings
import pickle
import anesthetic


class MAF:
    r"""Masked Autoregressive Flow for unbounded density estimation.

    This class trains and uses a normalizing flow built from chained autoregressive
    neural networks to learn probability distributions. This unbounded version uses
    standardization (mean/std) instead of min/max normalization, avoiding implicit
    parameter bounds.

    **Key Features of Unbounded Version:**
        - Uses mean/std standardization (not min/max normalization)
        - Provides .quantile() method for nested sampling applications
        - No implicit bounds on parameter space
        - Handles standardization/unstandardization automatically

    **Parameters:**

        theta: **numpy array or anesthetic.samples**
            | The samples from the probability distribution that we require the
                MAF to learn. This can either be a numpy array or an anesthetic
                NestedSamples or MCMCSamples object.

    **kwargs:**

        weights: **numpy array / default=np.ones(len(theta))**
            | The weights associated with the samples above. If an anesthetic
                NestedSamples or MCMCSamples object is passed the code
                draws the weights from this.

        number_networks: **int / default = 6**
            | The bijector is built by chaining a series of
                autoregressive neural
                networks together and this parameter is used to determine
                how many networks there are in the chain.

        learning_rate: **float / default = 1e-3**
            | The learning rate determines the 'step size' of the optimization
                algorithm used to train the MAF. Its value can effect the
                quality of emulation.

        hidden_layers: **list / default = [50, 50]**
            | The number of layers and number of nodes in each hidden layer for
                each neural network. The default is two hidden layers with
                50 nodes each and each network in the chain has the same hidden
                layer structure.

        activation_func: **string / default = 'tanh'**
            | The choice of activation function. It must be an activation
                function keyword recognisable by TensorFlow. The default is
                'tanh', the hyperbolic tangent activation function.

        parameters: **list of strings**
            | A list of the relevant parameters to train on. Only needed
                if theta is an anesthetic samples object. If not provided,
                all parameters will be used.

    **Important Methods:**

        train(): Train the MAF on the provided samples
        sample(n): Generate n samples from the learned distribution
        quantile(u): Transform uniform [0,1] samples to physical parameters (for nested sampling)
        log_prob(params): Compute log-probability of parameters
        save(filename): Save trained model to file
        load(filename): Load trained model from file (class method)

    **Attributes:**

        mean: **numpy array**
            | Mean of the training data, used for standardization

        std: **numpy array**
            | Standard deviation of the training data, used for standardization

        loss_history: **list**
            | This list contains the value of the loss function at each epoch
                during training.

    """

    def __init__(self, theta, **kwargs):
        self.number_networks = kwargs.pop('number_networks', 6)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])
        self.parameters = kwargs.pop('parameters', None)
        self.activation_func = kwargs.pop('activation_func', 'tanh')

        # Avoids unintended side effects outside the class
        if not isinstance(theta, tf.Tensor):
            theta = theta.copy()
        else:
            theta = tf.identity(theta)

        if isinstance(theta, 
                      (anesthetic.samples.NestedSamples, 
                       anesthetic.samples.MCMCSamples)):
            weights = theta.get_weights()
            if self.parameters:
                theta = theta[self.parameters].values
            else:
                if isinstance(theta, anesthetic.samples.NestedSamples):
                    self.parameters = theta.columns[:-3].values
                    theta = theta[theta.columns[:-3]].values
                else:
                    self.parameters = theta.columns[:-1].values
                    theta = theta[theta.columns[:-1]].values
        else:
            weights = kwargs.pop('weights', np.ones(len(theta)))

        self.theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        if not isinstance(weights, tf.Tensor):
            weights = tf.convert_to_tensor(weights.copy(), dtype=tf.float32)
        else:
            weights = tf.identity(weights)
        if weights.dtype != tf.float32:
            weights = tf.cast(weights, tf.float32)
        self.sample_weights = weights

        mask = np.isfinite(theta).all(axis=-1)
        self.theta = tf.boolean_mask(self.theta, mask, axis=0)
        self.sample_weights = tf.boolean_mask(
                                              self.sample_weights,
                                              mask, axis=0)

        self.n = tf.math.reduce_sum(self.sample_weights)**2 / \
            tf.math.reduce_sum(self.sample_weights**2)

        # Calculate and store standardization parameters (mean and standard deviation)
        self.mean = tf.math.reduce_mean(self.theta, axis=0)
        self.std = tf.math.reduce_std(self.theta, axis=0)

        # Edge case handling: if a parameter is constant, its std dev is 0.
        # This would cause NaN during division. We replace std=0 with 1.0,
        # which means the parameter is centered but not scaled.
        self.std = tf.where(tf.equal(self.std, 0), 1.0, self.std)

        # Standardize the training data. The MAF will be trained on this
        # whitened data.
        self.theta = (self.theta - self.mean) / self.std

        self.D = self.theta.shape[-1]

        if type(self.number_networks) is not int:
            raise TypeError("'number_networks' must be an integer.")
        if not isinstance(self.learning_rate,
                          (int, float,
                           keras.optimizers.schedules.LearningRateSchedule)):
            raise TypeError("'learning_rate', " +
                            "must be an integer, float or keras scheduler.")
        if type(self.hidden_layers) is not list:
            raise TypeError("'hidden_layers' must be a list of integers.")
        else:
            for i in range(len(self.hidden_layers)):
                if type(self.hidden_layers[i]) is not int:
                    raise TypeError(
                        "One or more values in 'hidden_layers'" +
                        "is not an integer.")

        self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate)

        self.gen_mades()

    def gen_mades(self):

        """Generating the masked autoregressive flow."""

        self.mades = [tfb.AutoregressiveNetwork(params=2,
                      hidden_units=self.hidden_layers, activation=self.activation_func,
                      input_order='random')
                      for _ in range(self.number_networks)]

        self.bij = tfb.Chain([
            tfb.MaskedAutoregressiveFlow(made) for made in self.mades])

        self.base = tfd.Blockwise(
            [tfd.Normal(loc=0, scale=1)
             for _ in range(self.D)])

        self.maf = tfd.TransformedDistribution(self.base, bijector=self.bij)

        return self.bij, self.maf

    def train(self, epochs=100, early_stop=False, loss_type='sum'):

        r"""

        This function is called to train the MAF once it has been
        initialised. It calls the `_training()` function.

        .. code:: python

            from margarine_unbounded.maf import MAF

            bij = MAF(theta, weights=weights)
            bij.train()

        **Kwargs:**

            epochs: **int / default = 100**
                | The number of iterations to train the neural networks for.

            early_stop: **boolean / default = False**
                | Determines whether or not to implement an early stopping
                    algorithm or
                    train for the set number of epochs. If set to True then the
                    algorithm will stop training when test loss has not
                    improved for 2% of the requested epochs. At this point
                    margarine will roll back to the best model and return this
                    to the user.

            loss_type: **string / default = 'sum'**
                | Determines whether to use the sum or mean of the weighted
                    log probabilities to calculate the loss function.


        """

        if type(epochs) is not int:
            raise TypeError("'epochs' is not an integer.")
        if type(early_stop) is not bool:
            raise TypeError("'early_stop' must be a boolean.")

        self.epochs = epochs
        self.early_stop = early_stop
        self.loss_type = loss_type

        self.maf = self._training(self.theta,
                                  self.sample_weights, self.maf)

    def _training(self, theta, sample_weights, maf):

        """Training the masked autoregressive flow."""

        phi = theta
        weights_phi = sample_weights/tf.reduce_sum(sample_weights)

        phi_train, phi_test, weights_phi_train, weights_phi_test = \
            pure_tf_train_test_split(phi, weights_phi, test_size=0.2)

        self.loss_history = []
        self.test_loss_history = []
        c = 0
        for i in tqdm.tqdm(range(self.epochs)):
            loss = self._train_step(phi_train,
                                    weights_phi_train,
                                    self.loss_type, maf)
            self.loss_history.append(loss)

            self.test_loss_history.append(self._test_step(phi_test,
                                          weights_phi_test,
                                          self.loss_type, maf))

            if self.early_stop:
                c += 1
                if i == 0:
                    minimum_loss = self.test_loss_history[-1]
                    minimum_epoch = i
                    minimum_model = None
                else:
                    if self.test_loss_history[-1] < minimum_loss:
                        minimum_loss = self.test_loss_history[-1]
                        minimum_epoch = i
                        minimum_model = maf.copy()
                        c = 0
                if minimum_model:
                    if c == round((self.epochs/100)*2):
                        print('Early stopped. Epochs used = ' + str(i) +
                              '. Minimum at epoch = ' + str(minimum_epoch))
                        return minimum_model
        return maf

    @tf.function(jit_compile=True)
    def _test_step(self, x, w, loss_type, maf):

        r"""
        This function is used to calculate the test loss value at each epoch
        for early stopping.
        """

        if loss_type == 'sum':
            loss = -tf.reduce_sum(w*maf.log_prob(x))
        elif loss_type == 'mean':
            loss = -tf.reduce_mean(w*maf.log_prob(x))
        return loss

    @tf.function(jit_compile=True)
    def _train_step(self, x, w, loss_type, maf):

        r"""
        This function is used to calculate the loss value at each epoch and
        adjust the weights and biases of the neural networks via the
        optimizer algorithm.
        """

        with tf.GradientTape() as tape:
            if loss_type == 'sum':
                loss = -tf.reduce_sum(w*maf.log_prob(x))
            elif loss_type == 'mean':
                loss = -tf.reduce_mean(w*maf.log_prob(x))
        gradients = tape.gradient(loss, maf.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients,
                maf.trainable_variables))
        return loss

    @tf.function(jit_compile=True)
    def sample(self, length=1000):
        r"""
        This function is used to generate samples from the trained MAF.
        """
        if type(length) is not int:
            raise TypeError("'length' must be an integer.")

        # 1. Generate samples in the standardized latent space
        standardized_samples = self.maf.sample(length)

        # 2. Unstandardize the samples back to the original data scale
        unstandardized_samples = standardized_samples * self.std + self.mean

        return unstandardized_samples

    @tf.function(jit_compile=True)
    def quantile(self, u):
        r"""
        Transforms samples from the unit uniform hypercube to the learned
        data space. This is the inverse of the cumulative distribution function.

        This is the function that should be used for applications like
        nested sampling.

        Args:
            u (tf.Tensor or array-like): Samples from the uniform unit
                hypercube, with shape (n_samples, n_dims).

        Returns:
            tf.Tensor: The transformed samples in the original data space.
        """
        if not isinstance(u, tf.Tensor):
            u = tf.convert_to_tensor(u, dtype=tf.float32)
        if u.dtype != tf.float32:
            u = tf.cast(u, tf.float32)

        # Step 1: Transform uniform samples to the base distribution (Standard Normal)
        # using the inverse CDF (quantile function).
        z = tfd.Normal(loc=0.0, scale=1.0).quantile(u)

        # Step 2: Push the base samples through the learned bijector to get
        # samples in the standardized data space.
        standardized_samples = self.maf.bijector.forward(z)

        # Step 3: Unstandardize the samples back to the original data scale.
        unstandardized_samples = standardized_samples * self.std + self.mean

        return unstandardized_samples


    @tf.function(jit_compile=True, reduce_retracing=True)
    def log_prob(self, params):
        """
        Function to calculate the log-probability for a given MAF and
        set of parameters in the original, unstandardized space.
        """
        if params.dtype != tf.float32:
            params = tf.cast(params, tf.float32)

        # 1. Standardize the input parameters
        standardized_params = (params - self.mean) / self.std

        # 2. Calculate the log_prob in the standardized space
        log_prob_standardized = self.maf.log_prob(standardized_params)

        # 3. Add the log-determinant of the Jacobian for the standardization
        # transformation to get the correct log_prob in the original space.
        # log p(x) = log p(z) - sum(log(sigma_i))
        log_det_jacobian = -tf.reduce_sum(tf.math.log(self.std))

        return log_prob_standardized + log_det_jacobian

    def log_like(self, params, logevidence, prior_de=None):

        r"""
        This function should return the log-likelihood for a given set of
        parameters.

        It requires the logevidence from the original nested sampling run
        in order to do this and in the case that the prior is non-uniform
        a trained prior density estimator should be provided.

        **Parameters:**

            params: **numpy array**
                | The set of samples for which to calculate the log
                    probability.

            logevidence: **float**
                | Should be the log-evidence from the full nested sampling
                    run with nuisance parameters.

            prior_de: **margarine.maf.MAF / default=None**
                | If the prior is non-uniform then a trained prior density
                    estimator should be provided. Otherwise the prior
                    is assumed to be uniform and the prior probability
                    is calculated analytically from the minimum and maximum
                    values of the parameters.

        """

        if prior_de is None:
            warnings.warn('Assuming prior is uniform!')
            prior_logprob = tf.math.log(
                                        tf.math.reduce_prod(
                                            [1/(self.theta_max[i] -
                                                self.theta_min[i])
                                                for i in range(
                                                    len(self.theta_min))]))
        else:
            prior_logprob = self.prior_de.log_prob(params)

        posterior_logprob = self.log_prob(params)

        loglike = posterior_logprob + logevidence - prior_logprob

        return loglike

    def save(self, filename):
        r"""

        This function can be used to save an instance of a trained MAF as
        a pickled class so that it can be loaded and used in differnt scripts.

        **Parameters:**

            filename: **string**
                | Path in which to save the pickled MAF.

        """

        nn_weights = [made.get_weights() for made in self.mades]
        with open(filename, 'wb') as f:
            pickle.dump([self.theta,
                        nn_weights,
                        self.sample_weights,
                        self.number_networks,
                        self.hidden_layers,
                        self.learning_rate,
                        self.mean,
                        self.std], f)

    @classmethod
    def load(cls, filename):
        r"""

        This function can be used to load a saved MAF. For example

        .. code:: python

            from margarine_unbounded.maf import MAF

            file = 'path/to/pickled/MAF.pkl'
            bij = MAF.load(file)

        **Parameters:**

            filename: **string**
                | Path to the saved MAF.

        """

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            theta, nn_weights, \
                sample_weights, \
                number_networks, \
                hidden_layers, \
                learning_rate, mean, std = data

        # Note: theta is the standardized data here. We pass it to the constructor
        # so the MAF is built with the correct (standardized) training data.
        bijector = cls(
            theta, weights=sample_weights, number_networks=number_networks,
            learning_rate=learning_rate, hidden_layers=hidden_layers)

        # The `theta` loaded is already standardized, so __init__ calculates
        # mean~0 and std~1. We MUST overwrite these with the true mean and std
        # from the original data space to ensure sample() and log_prob() work correctly.
        bijector.mean = mean
        bijector.std = std

        _ = bijector.sample(1)
        for made, nn_weights in zip(bijector.mades, nn_weights):
            made.set_weights(nn_weights)

        return bijector
