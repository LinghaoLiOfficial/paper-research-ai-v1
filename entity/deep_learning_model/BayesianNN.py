import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


# BNN层，类似于BP网络的Linear层，与BP网络类似，一层BNN层由weight和bias组成，weight和bias都具有均值和方差
class BayesianLayer(nn.Module):

    def __init__(self, input_features, output_features, device, prior_var=1.0):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """

        # initialize layers
        super().__init__()

        self.device = device

        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0,prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        # 从标准正态分布中采样权重
        w_epsilon = Normal(0,1).sample(self.w_mu.shape)
        # 获得服从均值为mu，方差为delta的正态分布的样本
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon.to(self.device)

        # sample bias
        # 与sample weights同理
        b_epsilon = Normal(0,1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon.to(self.device)

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        # 计算log p(w)，用于后续计算loss
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        # 计算 log p(w|\theta)，用于后续计算loss
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        # 权重确定后，和BP网络层一样使用
        return F.linear(input, self.w, self.b)


class BayesianNN(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, device, noise_tol=0.1,  prior_var=1.0):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super(BayesianNN, self).__init__()

        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood

        self.encoder_bayesian1 = BayesianLayer(hidden_dim1, hidden_dim2, device, prior_var=prior_var)

        self.encoder_bayesian2 = BayesianLayer(hidden_dim2, hidden_dim3, device, prior_var=prior_var)

        self.encoder_bayesian3 = BayesianLayer(hidden_dim3, hidden_dim4, device, prior_var=prior_var)

        self.decoder_bayesian1 = BayesianLayer(hidden_dim4, hidden_dim3, device, prior_var=prior_var)

        self.decoder_bayesian2 = BayesianLayer(hidden_dim3, hidden_dim2, device, prior_var=prior_var)

        self.decoder_bayesian3 = BayesianLayer(hidden_dim2, hidden_dim1, device, prior_var=prior_var)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h = x.reshape(-1, x.shape[2])

        h = self.encoder_bayesian1(h)
        h = self.relu(h)

        h = self.encoder_bayesian2(h)
        h = self.relu(h)

        h = self.encoder_bayesian3(h)
        h = self.relu(h)

        h = self.decoder_bayesian1(h)
        h = self.relu(h)

        h = self.decoder_bayesian2(h)
        h = self.relu(h)

        h = self.decoder_bayesian3(h)
        h = self.relu(h)

        y = h.reshape(-1, x.shape[1], h.shape[1])

        return y

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    # 计算loss
    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples

        # 蒙特卡罗近似
        for i in range(samples):
            outputs[i] = self(input) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like

        return loss

