# margarine_unbounded

Unbounded fork of [margarine](https://github.com/htjb/margarine) by Harry T. J. Bevins for flow-based inference without implicit parameter bounds.

## What is margarine_unbounded?

This is a fork of the original `margarine` package that removes implicit bounds on parameters during normalizing flow training. The key modification is using **mean/std standardization** instead of **min/max normalization**, making the flows "unbounded" and better suited for certain applications like posterior repartitioning in nested sampling.

## Key Differences from Original margarine

| Feature | Original margarine | margarine_unbounded |
|---------|-------------------|---------------------|
| **Normalization** | Min/max bounds | Mean/std standardization |
| **Parameter bounds** | Implicit bounds at training data limits | Unbounded (no hard limits) |
| **`.quantile()` method** | Not available | ✅ Available for nested sampling |
| **Use case** | General density estimation | Posterior repartitioning, nested sampling |

## Why Unbounded?

The original margarine uses min/max normalization, which implicitly bounds parameters to the range of the training data. For applications like posterior repartitioning, where you sample from a trained flow and need to evaluate probabilities slightly outside the training region, these implicit bounds can cause issues.

margarine_unbounded uses standardization (`(x - mean) / std`), which:
- Has no hard boundaries
- Handles edge cases more gracefully
- Provides cleaner transformations for nested sampling applications
- Works better when parameters may approach prior boundaries

## Installation

```bash
git clone https://github.com/mrosep/margarine_unbounded.git
cd margarine_unbounded
pip install .
```

## Usage

The API is identical to the original margarine:

```python
from margarine_unbounded.maf import MAF
import numpy as np

# Your posterior samples (from a preliminary run)
samples = np.random.randn(10000, 5)  # 10000 samples, 5 parameters

# Train the flow
maf = MAF(samples, number_networks=10, hidden_layers=[128, 128])
maf.train(epochs=100)

# Save the trained model
maf.save('trained_flow.pkl')

# Load later
maf_loaded = MAF.load('trained_flow.pkl')

# Generate new samples
new_samples = maf_loaded.sample(1000)

# For nested sampling: transform uniform [0,1] to physical parameters
import tensorflow as tf
u = tf.random.uniform((1000, 5))  # Uniform samples
physical_params = maf_loaded.quantile(u)  # Transform to physical space

# Compute log-probability
log_prob = maf_loaded.log_prob(physical_params)
```

## New Method: `.quantile()`

The `.quantile()` method is specifically designed for nested sampling applications:

```python
# Nested sampling prior transform
def prior_transform(u):
    """Transform uniform [0,1]^D to physical parameters"""
    return maf.quantile(u).numpy()
```

This method:
1. Transforms uniform [0,1] → standard normal N(0,1) via inverse CDF
2. Pushes through the trained bijector
3. Unstandardizes back to original parameter space

## Use with bilby-pr

This package is designed to work with [bilby-pr](https://github.com/mrosep/bilby-pr) for posterior repartitioning in gravitational wave parameter estimation:

```python
import bilby

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler='dynesty_pr',
    weights_file='trained_flow.pkl',          # Trained margarine_unbounded model
    flow_params=['mass_ratio', 'chirp_mass'],  # Parameters modeled by flow
    nlive=500
)
```

## Citation

If you use this package, please cite:

- **Original margarine**: Bevins et al. 2021, MNRAS, 508, 2923 ([arXiv:2102.12248](https://arxiv.org/abs/2102.12248))
- **This fork**: When publishing results using margarine_unbounded

Original margarine repository: https://github.com/htjb/margarine

## License

MIT License (same as original margarine) - see LICENSE file for details.

## Credits

- **Original margarine author**: Harry T. J. Bevins
- **Unbounded modifications**: Metha Prathaban
