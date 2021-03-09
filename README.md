```python
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# use ggplot styles for graphs
plt.style.use('ggplot')
```


```python
import tensorflow as tf
import tensorflow_probability as tfp

# set tf logger to log level ERROR to avoid warnings
tf.get_logger().setLevel('ERROR')

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
```


```python
# import probabilistic models
from bayes_vi.model import Model

# import utils
from bayes_vi.utils import to_ordered_dict
from bayes_vi.utils.datasets import make_dataset_from_df
from bayes_vi.utils.symplectic_integrators import LeapfrogIntegrator
```


```python
# mcmc imports
from bayes_vi.inference.mcmc import MCMC
from bayes_vi.inference.mcmc.transition_kernels import HamiltonianMonteCarlo, NoUTurnSampler, RandomWalkMetropolis
from bayes_vi.inference.mcmc.stepsize_adaptation_kernels import SimpleStepSizeAdaptation, DualAveragingStepSizeAdaptation
```


```python
# vi imports 
from bayes_vi.inference.vi import VI
from bayes_vi.inference.vi.surrogate_posteriors import ADVI, NormalizingFlow
from bayes_vi.inference.vi.flow_bijectors import HamiltonianFlow, AffineFlow, make_energy_fn, make_scale_fn, make_shift_fn
```

## 1. Example Workflow

A Bayesian modeling worklow using this package could look like this.

### 1.1. Create a Dataset
Create a `tf.data.Dataset` from your data. For this example consider some case some univariate Gaussian fake data.


```python
num_datapoints = 20
loc = 7
scale = 3
true_dist = tfd.Normal(loc=loc, scale=scale)
y = true_dist.sample(num_datapoints)
```

The package provides a utility function `make_dataset_from_df` for easy construction of a dataset from a `pd.DataFrame`.


```python
data = pd.DataFrame(dict(y=y))
dataset = make_dataset_from_df(data, target_names=['y'], format_features_as='dict')
```


```python
ax = data.plot(kind='kde', title='Univariate Gaussian Test Data')
```


    
![png](img/output_11_0.png)
    


### 1.2. Define a Bayesian Model

To define a Bayesian `Model`, simply provide an `collections.OrderedDict` of prior distributions as `tfp.distribution.Distirbution`s and a likelihood function, which takes the parameters as inputs by name.


```python
priors = collections.OrderedDict(
    loc = tfd.Normal(loc=0., scale=10.),
    scale = tfd.HalfNormal(scale=10.)
)

def likelihood(loc, scale): 
    return tfd.Normal(loc=loc, scale=scale)
```


```python
model = Model(priors=priors, likelihood=likelihood)
```

It is possible to provide a list of constraining bijectors on construction of the model, which allows for computations on an unconstrained space.

If no such list is defined, default bijectors for each prior are chosen.

### 1.3. Run Inference Algorithm of Choice

With this simple setup, you can now use either MCMC or variational inference to do Bayesian inference.

#### 1.3.1. Run MCMC (e.g. with No-U-Turn Sampler with dual averaging stepsize adaptation)

Note that the transition kernel may have various additional parameters. Here we use just the defaults and only provide the stepsize adaptation kernel.


```python
NUM_CHAINS = 2
NUM_SAMPLES = 4000
NUM_BURNIN_STEPS = 1000

stepsize_adaptation_kernel = DualAveragingStepSizeAdaptation(num_adaptation_steps=int(NUM_BURNIN_STEPS*0.8))

kernel = NoUTurnSampler(stepsize_adaptation_kernel=stepsize_adaptation_kernel)

mcmc = MCMC(model=model, dataset=dataset, transition_kernel=kernel)

mcmc_result = mcmc.fit(
    num_chains=NUM_CHAINS, 
    num_samples=NUM_SAMPLES, 
    num_burnin_steps=NUM_BURNIN_STEPS,
    progress_bar=True,
)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4978' class='' max='5000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  99.56% [4978/5000 00:14<00:00]
</div>



The `mcmc_result` is a wrapper containing the sample results `mcmc_result.samples` and additional traced metrics specific to the kernel `mcmc_result.trace`.

To visualize the results, one can use existing libraries like arviz (https://arviz-devs.github.io/arviz/index.html).

Later versions of this package should allow for a seamless integration with arviz, but for now we have to reformat the sample results to conform to arviz conventions.


```python
# adapt sample results to arviz required format --- later versions should allow for seamless integration with arviz
posterior_samples = to_ordered_dict(model.param_names, tf.nest.map_structure(lambda v: v.numpy().T, mcmc_result.samples))
```


```python
az.summary(posterior_samples)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loc</th>
      <td>6.135</td>
      <td>0.646</td>
      <td>4.890</td>
      <td>7.310</td>
      <td>0.009</td>
      <td>0.007</td>
      <td>4897.0</td>
      <td>4897.0</td>
      <td>4967.0</td>
      <td>4232.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>2.833</td>
      <td>0.514</td>
      <td>1.989</td>
      <td>3.828</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>4597.0</td>
      <td>4217.0</td>
      <td>5307.0</td>
      <td>3819.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = az.plot_trace(posterior_samples)
```


    
![png](img/output_25_0.png)
    


#### 1.3.2. Run Variational Inference (e.g. meanfield ADVI)


Instead of MCMC methods, we might want to use variational inference.

Choose a surrogate posterior, e.g. meanfield `ADVI`, a discrepancy function from `tfp.vi`, e.g. `tfp.vi.kl_reverse` , and select some optimization parameters (number of optimzation steps, sample size to approximate variational loss, optimizer, learning rate).


```python
NUM_STEPS = 1000
SAMPLE_SIZE = 10
LEARNING_RATE = 1e-2

surrogate_posterior = ADVI(model=model, mean_field=True)

vi = VI(model=model, dataset=dataset, surrogate_posterior=surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse)

approx_posterior, meanfield_advi_losses = vi.fit(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), 
    num_steps=NUM_STEPS, 
    sample_size=SAMPLE_SIZE, 
    progress_bar=True
)

```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:02<00:00 avg loss: 52.846]
</div>




```python
ax = plt.plot(meanfield_advi_losses)
```


    
![png](img/output_29_0.png)
    



```python
df = pd.DataFrame(approx_posterior.sample(5000))

df.describe(percentiles=[0.03, 0.5, 0.97]).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>3%</th>
      <th>50%</th>
      <th>97%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loc</th>
      <td>5000.0</td>
      <td>6.104273</td>
      <td>0.621283</td>
      <td>3.760990</td>
      <td>4.924492</td>
      <td>6.115833</td>
      <td>7.258379</td>
      <td>8.165336</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>5000.0</td>
      <td>2.807804</td>
      <td>0.435924</td>
      <td>1.296883</td>
      <td>1.978388</td>
      <td>2.813874</td>
      <td>3.635254</td>
      <td>4.231164</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = df.plot(kind='kde')
```


    
![png](img/output_31_0.png)
    


## 2. Normalizing Flows for Variational Inference

Continuing the simple example above, this is how you can flexibly define surrogate posteriors based on Normalizing flows for variational Bayesian inference.

By default, the base distribution is a standard diagonal Gaussian.

This example uses the predefined trainable `AffineFlow` bijector, but any parameterized `tfp.bijectors.Bijector` can be used. Just define your own!
You can construct more expressive flows by making use of the `tfp.bijectors.Chain` bijector to define compositions of flows.


```python
ndims = model.flat_unconstrained_param_event_ndims # this is the number of dimensions of the unconstrained parameter space
```

### 2.1. Affine Flows


```python
flow_bijector = AffineFlow(ndims)

surrogate_posterior = NormalizingFlow(model=model, flow_bijector=flow_bijector)

vi = VI(model=model, dataset=dataset, surrogate_posterior=surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse)

approx_posterior, affine_flow_losses = vi.fit(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), 
    num_steps=NUM_STEPS, 
    sample_size=SAMPLE_SIZE, 
    progress_bar=True
)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:03<00:00 avg loss: 52.864]
</div>



To define continuous normalizing flows, you may use is the `tfp.bijectors.FFJORD` bijector.

### 2.2. Continuous Flows


```python
def get_continuous_flow_bijector(unconstrained_event_dims):
    state_fn = tfk.Sequential()
    state_fn.add(tfk.layers.Dense(64, activation=tfk.activations.tanh))
    state_fn.add(tfk.layers.Dense(unconstrained_event_dims))
    state_fn.build((None, unconstrained_event_dims+1))
    state_time_derivative_fn = lambda t, state: state_fn(tf.concat([tf.fill((state.shape[0],1), t), state], axis=-1))
    return tfb.FFJORD(state_time_derivative_fn, 
                      ode_solve_fn=tfp.math.ode.DormandPrince(first_step_size=0.1).solve, 
                      trace_augmentation_fn=tfb.ffjord.trace_jacobian_hutchinson)

flow_bijector = get_continuous_flow_bijector(ndims)

surrogate_posterior = NormalizingFlow(model=model, flow_bijector=flow_bijector)

vi = VI(model=model, dataset=dataset, surrogate_posterior=surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse)

approx_posterior, cnf_losses = vi.fit(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), 
    num_steps=NUM_STEPS, 
    sample_size=SAMPLE_SIZE, 
    progress_bar=True
)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:58<00:00 avg loss: 52.855]
</div>



### 2.3. Comparison

Have a look at the optimzation behaviors in comparison.


```python
ax = pd.DataFrame({'meanfield_advi': meanfield_advi_losses, 'affine_flow': affine_flow_losses, 'cnf': cnf_losses}).plot(ylim=(-10,500))
```


    
![png](img/output_42_0.png)
    


## 3. Augmented Normalizing Flows for Variational Inference

The package allows to easily define augmented normalizing flows. This obviously has no advantages, at least for such a simple example model.

Simple choose a number of extra dimensions `extra_ndims` and define the flow bijector on the augmented space, i.e. on dimensions `ndims + extra_ndims`.

The optimization problem is solved on the augmented space by lifting the target distribution with the conditional `posterior_lift_distribution`, which by default is simply `lambda _: tfd.MultivariateNormalDiag(loc=tf.zeros(extra_ndims), scale_diag=tf.ones(extra_ndims))`.


```python
ndims = model.flat_unconstrained_param_event_ndims # this is the number of dimensions of the unconstrained parameter space

extra_ndims = 2 # dimensions of the augmentation space 
```


```python
flow_bijector = AffineFlow(ndims + extra_ndims)

surrogate_posterior = NormalizingFlow(model=model, flow_bijector=flow_bijector, extra_ndims=extra_ndims)

vi = VI(model=model, dataset=dataset, surrogate_posterior=surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse)

approx_posterior, affine_flow_losses = vi.fit(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), 
    num_steps=NUM_STEPS, 
    sample_size=SAMPLE_SIZE, 
    progress_bar=True
)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:03<00:00 avg loss: 52.876]
</div>




```python
def get_continuous_flow_bijector(unconstrained_event_dims):
    state_fn = tfk.Sequential()
    state_fn.add(tfk.layers.Dense(64, activation=tfk.activations.tanh))
    state_fn.add(tfk.layers.Dense(unconstrained_event_dims))
    state_fn.build((None, unconstrained_event_dims+1))
    state_time_derivative_fn = lambda t, state: state_fn(tf.concat([tf.fill((state.shape[0],1), t), state], axis=-1))
    return tfb.FFJORD(state_time_derivative_fn, 
                      ode_solve_fn=tfp.math.ode.DormandPrince(first_step_size=0.1).solve, 
                      trace_augmentation_fn=tfb.ffjord.trace_jacobian_hutchinson)

flow_bijector = get_continuous_flow_bijector(ndims + extra_ndims)

surrogate_posterior = NormalizingFlow(model=model, flow_bijector=flow_bijector, extra_ndims=extra_ndims)

vi = VI(model=model, dataset=dataset, surrogate_posterior=surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse)

approx_posterior, cnf_losses = vi.fit(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), 
    num_steps=NUM_STEPS, 
    sample_size=SAMPLE_SIZE, 
    progress_bar=True
)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 01:03<00:00 avg loss: 53.075]
</div>




```python
ax = pd.DataFrame({'meanfield_advi': meanfield_advi_losses, 'affine_flow': affine_flow_losses, 'cnf': cnf_losses}).plot(ylim=(-10,500))
```


    
![png](img/output_48_0.png)
    


## 4. Hamiltonian Normalizing Flows

As another example consider Hamiltonian Normalizing Flows (HNF) as motivated by Toth, P., Rezende, D. J., Jaegle, A., Racanière, S., Botev, A., & Higgins, I. (2019). Hamiltonian Generative Networks. ArXiv. http://arxiv.org/abs/1909.13789 .

HNFs are based on defining augmented flows as the Hamiltonian dynamics of a family of Hamiltonian systems. 

To compute such dynamics, we use the symplectic leapfrog integrator. 

The package provides an implementation of a general Hamiltonian Flow bijector.

### 4.1. Verify Hamiltonian Flow Bijector
To test the implementation the Hamiltonian Flow Bijector, consider the Hamiltonian of a 1d harmonic oscillator with mass $m=1$ and constant $k=1$:

$H(q,p) = T(p) + U(q) = 0.5 p^2 + 0.5 q^2$.

We know that this should yield an oscillating dynamic, which manifests as a circle on phase space.


```python
T = lambda p: 0.5*p**2
U = lambda q: 0.5*q**2

# event dims can be ignore in this example, because they are only used to construct trainable default energy function if none are provided explicitly
hamiltonian_flow = HamiltonianFlow(
    event_dims=1, 
    potential_energy_fn=U, 
    kinetic_energy_fn=T, 
    symplectic_integrator=LeapfrogIntegrator(), 
    step_sizes=0.5, 
    num_integration_steps=1
) 

initial_state = tf.constant([1.0, 0.0])
path = [initial_state.numpy()]

for _ in range(20):
    path.append(hamiltonian_flow.forward(path[-1]).numpy())
path = np.stack(path)
```


```python
fig, [ax1, ax2] = plt.subplots(1,2, figsize=(6,3))

ax1.plot(path[:, 0])
ax1.set_title('configuration plot')

ax2.plot(path[:,0], path[:,1])
ax2.set_title('phase space plot')
plt.tight_layout()
```


    
![png](img/output_54_0.png)
    


### 4.2. Hamiltonian Normalizing Flows

In the case of HNFs the augmentation space has the same dimension as the parameter space.


```python
ndims = model.flat_unconstrained_param_event_ndims # this is the number of dimensions of the unconstrained parameter space

extra_ndims = ndims # dimensions of the augmentation space 
```

For simplicity, we use the default MLP based potential and kinetic energy functions.


```python
flow_bijector = HamiltonianFlow(ndims, step_sizes=tf.Variable(0.1), num_integration_steps=2)

surrogate_posterior = NormalizingFlow(model=model, flow_bijector=flow_bijector, extra_ndims=extra_ndims)

vi = VI(model=model, dataset=dataset, surrogate_posterior=surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse)

approx_posterior, hnf_losses = vi.fit(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), 
    num_steps=NUM_STEPS, 
    sample_size=SAMPLE_SIZE, 
    progress_bar=True
)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:06<00:00 avg loss: 55.220]
</div>




```python
ax = pd.DataFrame({'meanfield_advi': meanfield_advi_losses, 'affine_flow': affine_flow_losses, 'cnf': cnf_losses, 'hnf': hnf_losses}).plot(ylim=(-10,500))
```


    
![png](img/output_60_0.png)
    


An interesting observation is that HNFs become more stable and perform better, if the base distribution is trainable.

This compensates for the problem, that volume preserving flows like HNFs can only permute the densities of the base distribution. 


```python
flow_bijector = HamiltonianFlow(ndims, step_sizes=tf.Variable(0.1), num_integration_steps=2)

base_distribution = tfd.MultivariateNormalDiag(loc=tf.Variable([0.0]*2*ndims), scale_diag=tf.Variable([0.1]*2*ndims))

surrogate_posterior = NormalizingFlow(model=model, flow_bijector=flow_bijector, base_distribution=base_distribution, extra_ndims=extra_ndims)

vi = VI(model=model, dataset=dataset, surrogate_posterior=surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse)

approx_posterior, hnf_losses = vi.fit(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), 
    num_steps=NUM_STEPS, 
    sample_size=SAMPLE_SIZE, 
    progress_bar=True
)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:06<00:00 avg loss: 53.273]
</div>




```python
ax = pd.DataFrame({'meanfield_advi': meanfield_advi_losses, 'affine_flow': affine_flow_losses, 'cnf': cnf_losses, 'hnf': hnf_losses}).plot(ylim=(-10,500))
```


    
![png](img/output_63_0.png)
    


## 5. Regression Model Example

As a final example, this is how to define regression models.

### 5.1. Create a Dataset
Again, create a `tf.data.Dataset` from your data or, in this case, generate some fake dataset.


```python
num_datapoints = 50
beta0 = 7
beta1 = 3
scale = 10
x = tf.random.normal((num_datapoints,), mean=0.0, stddev=10.0)
true_dist = tfd.Normal(loc=beta0 + x*beta1, scale=scale)
y = true_dist.sample()
```

The package provides a utility function `make_dataset_from_df` for easy construction of a dataset from a `pd.DataFrame`.


```python
data = pd.DataFrame(dict(x=x, y=y))
dataset = make_dataset_from_df(data, target_names=['y'], feature_names=['x'], format_features_as='dict')
```


```python
ax = data.plot(x='x', y='y', kind='scatter', title='Linear Regression Test Data')
```


    
![png](img/output_70_0.png)
    


### 1.2. Define a Bayesian Model

To define a Bayesian `Model`, simply provide an `collections.OrderedDict` of prior distributions as `tfp.distribution.Distirbution`s and a likelihood function, which takes the parameters and the features as inputs.


```python
# define priors
priors = collections.OrderedDict(
    beta_0 = tfd.Normal(loc=0., scale=10.),
    beta_1 = tfd.Normal(loc=0., scale=10.),
    scale = tfd.HalfNormal(scale=10.),
)

# define likelihood
def likelihood(beta_0, beta_1, scale, features):
    linear_response = beta_0 + beta_1*features['x']
    return tfd.Normal(loc=linear_response, scale=scale)
```


```python
model = Model(priors=priors, likelihood=likelihood)
```

### 1.3. Run Inference Algorithm of Choice

You can now again use either MCMC or variational inference to do Bayesian inference.

#### 1.3.1. Run MCMC (e.g. with Random Walk Metropolis)


```python
NUM_CHAINS = 2
NUM_SAMPLES = 4000
NUM_BURNIN_STEPS = 1000


kernel = RandomWalkMetropolis()

mcmc = MCMC(model=model, dataset=dataset, transition_kernel=kernel)

mcmc_result = mcmc.fit(
    num_chains=NUM_CHAINS, 
    num_samples=NUM_SAMPLES, 
    num_burnin_steps=NUM_BURNIN_STEPS,
    progress_bar=True,
)
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4689' class='' max='5000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  93.78% [4689/5000 00:01<00:00]
</div>



The `mcmc_result` is a wrapper containing the sample results `mcmc_result.samples` and additional traced metrics specific to the kernel `mcmc_result.trace`.

To visualize the results, one can use existing libraries like arviz (https://arviz-devs.github.io/arviz/index.html).

Later versions of this package should allow for a seamless integration with arviz, but for now we have to reformat the sample results to conform to arviz conventions.


```python
# adapt sample results to arviz required format --- later versions should allow for seamless integration with arviz
posterior_samples = to_ordered_dict(model.param_names, tf.nest.map_structure(lambda v: v.numpy().T, mcmc_result.samples))
```


```python
az.summary(posterior_samples)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>beta_0</th>
      <td>7.406</td>
      <td>1.303</td>
      <td>4.549</td>
      <td>9.655</td>
      <td>0.127</td>
      <td>0.090</td>
      <td>105.0</td>
      <td>105.0</td>
      <td>106.0</td>
      <td>160.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>beta_1</th>
      <td>2.808</td>
      <td>0.139</td>
      <td>2.559</td>
      <td>3.075</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>623.0</td>
      <td>614.0</td>
      <td>611.0</td>
      <td>588.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>9.705</td>
      <td>0.851</td>
      <td>8.384</td>
      <td>11.415</td>
      <td>0.052</td>
      <td>0.037</td>
      <td>270.0</td>
      <td>270.0</td>
      <td>268.0</td>
      <td>333.0</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = az.plot_trace(posterior_samples)
```


    
![png](img/output_82_0.png)
    


#### 1.3.2. Run Variational Inference (e.g. full rank ADVI)



```python
NUM_STEPS = 1000
SAMPLE_SIZE = 10
LEARNING_RATE = 1e-2

surrogate_posterior = ADVI(model=model, mean_field=True)

vi = VI(model=model, dataset=dataset, surrogate_posterior=surrogate_posterior, discrepancy_fn=tfp.vi.kl_reverse)

approx_posterior, meanfield_advi_losses = vi.fit(
    optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), 
    num_steps=NUM_STEPS, 
    sample_size=SAMPLE_SIZE, 
    progress_bar=True
)

```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:03<00:00 avg loss: 193.700]
</div>




```python
ax = plt.plot(meanfield_advi_losses)
```


    
![png](img/output_85_0.png)
    



```python
df = pd.DataFrame(approx_posterior.sample(5000))

df.describe(percentiles=[0.03, 0.5, 0.97]).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>3%</th>
      <th>50%</th>
      <th>97%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>beta_0</th>
      <td>5000.0</td>
      <td>7.412712</td>
      <td>1.148661</td>
      <td>2.865888</td>
      <td>5.230334</td>
      <td>7.409355</td>
      <td>9.530113</td>
      <td>11.454782</td>
    </tr>
    <tr>
      <th>beta_1</th>
      <td>5000.0</td>
      <td>2.827650</td>
      <td>0.125789</td>
      <td>2.361377</td>
      <td>2.593280</td>
      <td>2.827903</td>
      <td>3.058454</td>
      <td>3.290145</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>5000.0</td>
      <td>8.695953</td>
      <td>0.630038</td>
      <td>6.448833</td>
      <td>7.504722</td>
      <td>8.690381</td>
      <td>9.900940</td>
      <td>10.838046</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = df.plot(kind='kde')
```


    
![png](img/output_87_0.png)
    

