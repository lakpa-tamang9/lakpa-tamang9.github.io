---
layout: post
title: 'Variational Bayesian Inference'
date: 2024-05-18 02:41:00-0400
description: Using variational inference in bayesian networks..
tags: bayesian inference, ELBO
categories: blog-posts
related_posts: false
---
**Log likelihood** is a statistical measure that is used to quantify how well a given set of parameters explains the observed data under specific probabilistic model. It is especially popular in statistics and machine learning community, particularly in the context of Bayesian inference, which this post is about.
In simpler words, Likelihood : How probable is the observed data given a set of parameters?, and Log-likelihoo: The natural logarithm of the likelihood, transforming a product of proabilities into a sum of log-probabilities for ease of computation.

Now let's break down the log likelihood to more thorough understanding with an example:
Let us consider, we have some observed data points $$x_1, x_2, \cdots, x_n$$. Also, let's assume that these data points are generated from a probability distribution with parameters $$\theta$$. For example, the data might come from a normal distribution with mean $$\mu$$, and standard deviation $$\sigma$$ such that $$\theta = (\mu, \sigma)$$.

Now the likelihood function $$L(\theta)$$ is the probability of the observed data given the parameters $$\theta$$, or mathematically:

$$
L(\theta) = P(X \mid \theta)
$$

For $$n$$ independent and identically distributed (i.i.d) observations $$x_1, x_2, \cdots, x_n$$, the likelihood function is defined as the product of the probabilities of each individual observation. Mathematically,

$$
L(\theta) = \prod_{i = 1}^{n} P(x_i \mid \theta)
$$

Subsequently, the **Log-Likelihood Function** is now simply the natural logarithm of the above likelihood function.

$$
\mathcal{l}(\theta) = \sum_{i = 1}^n \log L(\theta) = log(\prod_{i = 1}^n)P(x_i \mid \theta)
$$

Using the properties of logarithm, the above equation can be expressed as sum. This transformation from product to sum is often done because mathematically sums are easier to work with compared to products, especially when taking gradients (eg: for optimization).

$$
\mathcal{l}(\theta) = \sum_{i = 1}^n \log P(x_i \mid \theta)
$$

Log-Likelihood for a Normal Distribution
======
Usually in scenarios when the data is modelled as a Normal (Gaussian) distribution with some mean $$\mu$$, and standard deviation $$\sigma$$, the conditional probability of some input $$x_i$$ is given by:

$$
P(x_i \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(\frac{-(x_i - \mu)^2}{2\sigma^2})
$$

The log-likelihood function for this normal distribution can be written as:

$$
\mathcal{l}(\mu, \sigma) = \sum_{i = 1}^n \log(\frac{1}{\sqrt{2\pi\sigma^2}}\exp(\frac{-(x_i - \mu)^2}{2\sigma^2}))
$$

$$
\mathcal{l}(\mu, \sigma) = \sum_{i = 1}^n(log(\frac{1}{\sqrt{2\pi\sigma^2}}) + \log(\exp(\frac{-(x_i - \mu)^2}{2\sigma^2})))
$$

$$
\mathcal{l}(\mu, \sigma) = \sum_{i = 1}^n(-\frac{1}{2}\log(2\pi\sigma^2) - \frac{-(x_i - \mu)^2}{2\sigma^2})
$$

Bayesian Linear Layer 
======
Now, let us create a Bayesian linear layer in Pytorch. The weights of the linear layer are sampled from a fully factorised Normal with learnable parameters. The likelihood
of the weight samples under the prior and the approximate posterior are returned with each forward pass in order to estimate the KL term in the ELBO.

In the following code, a Python Class created that inherits from nn.Module. For easier understanding of the process, I will try to break down and explain the code snippets in simple manner. 
- First, the layer is initialized with the number of input features `num_inp_feats`, the number of output features `num_out_feats`, a prior distribution class `prior_class`, and an optional bias term `with_bias` (default is `True`).
- `self.W_mu` and `self.W_p` are the mean and a parameter related to the standard deviation of the weights, respectively.
- `W_mu` is initialized uniformly between -0.1 and 0.1.
- `W_p` is initialized uniformly between -3 and -2.
- `self.b_mu` and `self.b_p` are the mean and a parameter related to the standard deviation of the biases, respectively, initialized similarly to the weights.

```
class BayesLinearNormalDist(nn.Module):

    def __init__(self, num_inp_feats, num_out_feats, prior_class, with_bias=True):
        super(BayesLinearNormalDist, self).__init__()
        self.num_inp_feats = num_inp_feats
        self.num_out_feats = num_out_feats
        self.prior = prior_class
        self.with_bias = with_bias

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(
            torch.Tensor(self.num_inp_feats, self.num_out_feats).uniform_(-0.1, 0.1)
        )
        self.W_p = nn.Parameter(
            torch.Tensor(self.num_inp_feats, self.num_out_feats).uniform_(-3, -2)
        )

        self.b_mu = nn.Parameter(torch.Tensor(self.num_out_feats).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.num_out_feats).uniform_(-3, -2))

```
The forward method below defines the forward pass of the layer.
- If `ifsample` is `False`, the method returns the deterministic output using the mean of the weights and biases.
- If `ifsample` is `True`, it proceeds to sample weights and biases from the approximate posterior distribution rather than using the mean values. This stochastic sampling is crucial for Bayesian neural networks, allowing the model to account for uncertainty in the parameters. 
```
    def forward(self, X, sample=0, local_rep=False, ifsample=True):

        if not ifsample:  # When training return MLE of w for quick validation
            # pdb.set_trace()
            if self.with_bias:
                output = torch.mm(X, self.W_mu) + self.b_mu.expand(
                    X.size()[0], self.num_out_feats
                )
            else:
                output = torch.mm(X, self.W_mu)
            return output, torch.Tensor([0]).cuda()
```
The weights and biases are expanded along a new dimension for sampling.
- `eps_W` and `eps_b` are standard normal noise samples.
- `std_w` and `std_b` are computed using the softplus function to ensure positivity.
- `W` and `b` are sampled from the approximate posterior.
- *Unsqueezing*: The unsqueeze method adds an extra dimension to the tensors. This is necessary to prepare the tensors for sampling.
    - `W_mu` and `W_p` are unsqueezed at dimension 1, transforming them from shape (`num_inp_feats`, `num_out_feats`) to (`num_inp_feats`, 1, `num_out_feats`).
    - `b_mu` and `b_p` are unsqueezed at dimension 0, transforming them from shape (`num_out_feats`) to (1, `num_out_feats`).

- *Repeating*: The repeat method replicates the tensors along specified dimensions.This ensures that each weight and bias parameter has multiple samples, one for each desired sample.

    - `W_mu` and `W_p` are repeated along the new dimension to match the number of samples, resulting in shape (`num_inp_feats`, `sample`, `num_out_feats`).
    - `b_mu` and `b_p` are repeated along the new dimension to match the number of samples, resulting in shape (`sample`, `num_out_feats`).

- *Sampling Noise*: `eps_W` and `eps_b` are tensors of the same shape as `W_mu` and `b_mu`, filled with samples from a standard normal distribution.
- *Computing Standard Deviation*: The standard deviations for weights and biases are computed using the *softplus* function applied to `W_p` and `b_p`. The softplus function ensures that the standard deviations are positive.
- The weights `W` and biases `b` are sampled from their approximate posterior distributions by adding the scaled noise (`std_w` * `eps_W` and `std_b` * `eps_b`) to their means (`W_mu` and `b_mu`).
```
        else:
            if not local_rep:
                # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
                # the same random sample is used for every element in the minibatch
                # pdb.set_trace()
                W_mu = self.W_mu.unsqueeze(1).repeat(1, sample, 1)
                W_p = self.W_p.unsqueeze(1).repeat(1, sample, 1)

                b_mu = self.b_mu.unsqueeze(0).repeat(sample, 1)
                b_p = self.b_p.unsqueeze(0).repeat(sample, 1)

                eps_W = W_mu.data.new(W_mu.size()).normal_()
                eps_b = b_mu.data.new(b_mu.size()).normal_()

                if not ifsample:
                    eps_W = eps_W * 0
                    eps_b = eps_b * 0

                # sample parameters
                std_w = 1e-6 + f.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + f.softplus(b_p, beta=1, threshold=20)

                W = W_mu + 1 * std_w * eps_W
                b = b_mu + 1 * std_b * eps_b
```
- `lqw` and `lpw` are the log-likelihoods of the weights and biases under the approximate posterior and prior distributions, respectively.
- **Log-Likelihood under the Posterior (lqw)**: This is computed for both weights and biases if `self.with_bias` is `True`. The function `isotropic_gauss_loglike` calculates the log-likelihood of the sampled parameters under the approximate posterior distribution.

- **Log-Likelihood under the Prior (lpw)**: This is computed using the provided prior class for both weights and biases if `self.with_bias` is `True`.

For log-likelihood under the posterior, the weight and bias parameter including the mean, and standard deviation of weights are passed as arguments to `isotropic_gauss_loglike` function. The `isotropic_gauss_loglike` function computes the log-likelihood of a set of samples `x` under an isotropic Gaussian (Normal) distribution with mean `mu` and standard deviation `sigma`. Within the `BayesLinearNormalDist` class, the `isotropic_gauss_loglike` function is used to calculate the log-likelihood of the sampled weights and biases under the approximate posterior distribution. This helps in computing the *Evidence Lower Bound (ELBO)* for variational inference. Specifically, the log-likelihood difference between the posterior and prior distributions (`lqw` - `lpw`) is used as part of the loss function to optimize the parameters of the model.Here is a detailed breakdown of the method:
```
def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    """Returns the computed log-likelihood

    Args:
        x (_type_): the sampled weights or biases
        mu (_type_): mean of gaussian distribution
        sigma (_type_): standard deviation of gaussian dist
        do_sum (bool, optional): _description_. a boolean indicating whether to sum the log-likelihoods
        over all elements or to take the mean.

    Returns:
        _type_: Gaussian Log likelihood
    """
    cte_term = -(0.5) * np.log(2 * np.pi)   # constant term
    det_sig_term = -torch.log(sigma)    # Determinant term
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner**2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term).mean()
    return out
```
- `cte_term` : Constant term, This term is a constant that is part of the Gaussian log-likelihood formula. It arises from the normalization constant of the Gaussian distribution.
- `det_sig_term`: Determinant term, This term accounts for the log of the determinant of the covariance matrix. In the isotropic case (where the covariance matrix is diagonal with equal entries), this term simplifies to the log of the standard deviation.
- dist_term: Distance term, This term measures the squared distance between the samples x and the mean mu, scaled by the standard deviation sigma. The result is then multiplied by -0.5, which is part of the Gaussian log-likelihood formula.
- The output is computed using the sampled weights and biases.

```
                if self.with_bias:
                    lqw = isotropic_gauss_loglike(
                        W, W_mu, std_w
                    ) + isotropic_gauss_loglike(b, b_mu, std_b)
                    lpw = self.prior.loglike(W) + self.prior.loglike(b)
                else:
                    lqw = isotropic_gauss_loglike(W, W_mu, std_w)
                    lpw = self.prior.loglike(W)

                # Reshaping weight to (num_inp_feats, num_out_feats) and biases to (num_out_feats)
                W = W.view(W.size()[0], -1)
                b = b.view(-1)

                if self.with_bias:
                    # wx + b
                    output = torch.mm(X, W) + b.unsqueeze(0).expand(
                        X.shape[0], -1
                    )  # (batch_size, num_out_featsput)
                else:
                    output = torch.mm(X, W)
```

If local representation `local_rep` is `True`, weights and biases are sampled for each data point in the batch. The process becomes slightly different, with weights and biases sampled for each data point individually rather than a shared sample across the batch. The output is computed using batch matrix multiplication (`torch.bmm`). The method returns the output and the difference between the log-likelihoods of the weights and biases under the posterior and prior distributions.

```
            else:
                W_mu = self.W_mu.unsqueeze(0).repeat(X.size()[0], 1, 1)
                W_p = self.W_p.unsqueeze(0).repeat(X.size()[0], 1, 1)

                b_mu = self.b_mu.unsqueeze(0).repeat(X.size()[0], 1)
                b_p = self.b_p.unsqueeze(0).repeat(X.size()[0], 1)
                # pdb.set_trace()
                eps_W = W_mu.data.new(W_mu.size()).normal_()
                eps_b = b_mu.data.new(b_mu.size()).normal_()

                # sample parameters
                std_w = 1e-6 + f.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + f.softplus(b_p, beta=1, threshold=20)

                W = W_mu + 1 * std_w * eps_W
                b = b_mu + 1 * std_b * eps_b

                # W = W.view(W.size()[0], -1)
                # b = b.view(-1)
                # pdb.set_trace()

                if self.with_bias:
                    output = (
                        torch.bmm(X.view(X.size()[0], 1, X.size()[1]), W).squeeze() + b
                    )  # (batch_size, num_out_featsput)
                    lqw = isotropic_gauss_loglike(
                        W, W_mu, std_w
                    ) + isotropic_gauss_loglike(b, b_mu, std_b)
                    lpw = self.prior.loglike(W) + self.prior.loglike(b)
                else:
                    output = torch.bmm(X.view(X.size()[0], 1, X.size()[1]), W).squeeze()
                    lqw = isotropic_gauss_loglike(W, W_mu, std_w)
                    lpw = self.prior.loglike(W)

            return output, lqw - lpw
```
