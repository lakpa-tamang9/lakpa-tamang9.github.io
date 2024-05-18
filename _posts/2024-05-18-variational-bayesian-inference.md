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