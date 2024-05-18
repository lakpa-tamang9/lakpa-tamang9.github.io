---
layout: post
title: Variational Bayesian Inference 
date: 2024-05-18 02:41:00-0400
description: Using variational inference in bayesian networks..
tags: bayesian inference, ELBO
categories: blog-posts
related_posts: false
---
**Log likelihood** is a statistical measure that is used to quantify how well a given set of parameters explains the observed data under specific probabilistic model. It is especially popular in statistics and machine learning community, particularly in the context of Bayesian inference, which this post is about.
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
l(\theta) = \sum_{i = 1}^n \log L(\theta) = log(\prod_{i = 1}^n)P(x_i \mid \theta)
$$
Using the properties of logarithm, the above equation can be expressed as sum. This transformation from product to sum is often done because mathematically sums are easier to work with compared to products, especially when taking gradients (eg: for optimization).
$$
l(\theta) = \sum_{i = 1}^n \log P(x_i \mid \theta)
$$

Log-Likelihood for a Normal Distribution
======
Usually in scenarios when the data is modelled as a Normal (Gaussian) distribution with some mean $$\mu$$, and standard deviation $$\sigma$$, the conditional probability of some input $$x_i$$ is given by:
$$
P(x_i \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(\frac{-(x_i - \mu)^2}{2\sigma^2})
$$
