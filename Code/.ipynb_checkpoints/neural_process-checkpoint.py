import torch
import numpy as np


class NP(torch.nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, encoder_specs, decoder_specs, init_func):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.encoder_specs = encoder_specs
        self.decoder_specs = decoder_specs
        self.init_func = init_func
        self.softplus_act = torch.nn.Softplus()
        self.r_to_z_mean = torch.nn.Linear(self.r_dim, self.z_dim)
        self.r_to_z_logvar = torch.nn.Linear(self.r_dim, self.z_dim)

        
        # Create the encoder
        for i in range(len(self.encoder_specs)):
            if i == 0:    
                encoder_input_dim = self.x_dim + self.y_dim
                self.add_module('h_layer' + str(i), \
                        torch.nn.Linear(encoder_input_dim, self.encoder_specs[i][0]))

                if self.encoder_specs[i][1]:
                    self.add_module('h_layer' + str(i) + '_act', self.encoder_specs[i][1])

            else:
                self.add_module('h_layer' + str(i), \
                        torch.nn.Linear(self.encoder_specs[i-1][0], self.encoder_specs[i][0]))

                if self.encoder_specs[i][1]:
                    self.add_module('h_layer' + str(i) + '_act', self.encoder_specs[i][1]) 

        # Create the decoder
        for i in range(len(self.decoder_specs)):
            if i == 0:    
                decoder_input_dim = self.x_dim + self.z_dim
                self.add_module('g_layer' + str(i), \
                    torch.nn.Linear(decoder_input_dim, self.decoder_specs[i][0]))

                if self.decoder_specs[i][1]:
                    self.add_module('g_layer' + str(i) + '_act', self.decoder_specs[i][1])

            else:
                self.add_module('g_layer' + str(i), \
                    torch.nn.Linear(self.decoder_specs[i-1][0], self.decoder_specs[i][0]))

                if self.decoder_specs[i][1]:
                    self.add_module('g_layer' + str(i) + '_act', self.decoder_specs[i][1]) 
        
        if init_func:
            for layer_name,_ in self._modules.items():
                if layer_name.endswith('act') == False:
                    init_func(getattr(getattr(self, layer_name), 'weight'))
        


    def h(self, x, y):
        x_y = torch.cat([x, y], dim=1)
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('h'):
                x_y = layer_func(x_y)
        return x_y



    def aggregate(self, r):
        return torch.mean(r, dim=0)



    def xy_to_z_params(self, x, y):
        r = self.h(x, y)
        r = self.aggregate(r)

        mean = self.r_to_z_mean(r)
        logvar = self.r_to_z_logvar(r)

        return mean.unsqueeze(-1), logvar.unsqueeze(-1)
       


    def sample_z(self, z, how_many):
        """
        Returns a sample from z of size (z_dim, how_many)
        """
        mean, logvar = z
        std = torch.exp(0.5 * logvar)

        eps = torch.randn([self.z_dim, how_many])
        z_samples = mean + std * eps
        return z_samples



    def g(self, x, z):
        z_reshape =  z.t().unsqueeze(1).expand(-1, x.shape[0], -1)
        x_reshape = x.unsqueeze(0).expand(z_reshape.shape[0], x.shape[0], x.shape[1])
        x_z = torch.cat([x_reshape, z_reshape], dim=2)

        y_mean = x_z
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('g'):
                y_mean = layer_func(y_mean)

        return y_mean
    


    def forward(self, x_context, y_context, x_target, y_target):
        z_context = self.xy_to_z_params(x_context, y_context)
        print(self.training)
        if self.training:
            z_target = self.xy_to_z_params(x_target, y_target)
        else:
            z_target = z_context

        z_sample = self.sample_z(z_target, how_many=1)
        y_hat = self.g(x_target, z_sample)

        return y_hat, z_target, z_context
    
    
    
def KL_div(mu_q, logvar_q, mu_p, logvar_p):
    KL = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p) \
             - 1.0 \
             + logvar_p - logvar_q
    KL = 0.5 * KL.sum()
    return KL


def ELBO(y_hat, y, z_target, z_context):
    log_lik = torch.nn.functional.mse_loss(y_hat, y)
    KL = KL_div(z_target[0], z_target[1], z_context[0], z_context[1])
    return - log_lik + KL


# Log-likelihood
def MC_loglikelihood(inputs, outputs, decoder, z_mean, z_std, how_many, device=None):
    """
    Returns a Monte Carlo estimate of the log-likelihood
    z_mean: mean of the distribution from which to sample z
    z_std: std of the distribution from which to sample z
    how_many: number of monte carlo samples
    decoder: the decoder to be used to produce estimates of mean
    """
    # sample z
    if device:
        z_samples = sample_z(z_mean, z_std, how_many, device=device)
    else:
        z_samples = sample_z(z_mean, z_std, how_many)
    # produce the 10 estimated of the mean and std
    y_mean, y_std = decoder(inputs, z_samples)
    # define likelihood for each value of z
    likelihood = torch.distributions.Normal(y_mean, y_std)
        # for each value of z: 
            # evaluate log-likelihood for each data point 
            # sum these per-data point log-likelihoods
        # compute the mean
    log_likelihood = likelihood.log_prob(outputs).sum(dim=1).mean()
    return log_likelihood



# KL divergence
def KL_div(mean_1, std_1, mean_2, std_2):
    """Analytical KLD between 2 Gaussians."""
    KL = (torch.log(std_2) - torch.log(std_1) + \
        (std_1**2/ (2*std_2**2)) + \
        ((mean_1 - mean_2)**2 / (2*std_2**2)) - 1).sum()*0.5
    return KL



def predict(inputs, decoder, z_mean, z_std, how_many, numpy=True, device=None):
    """
    Generates prediction from the NP
    inputs: inputs to the NP
    decoder: the specific decoder employed
    z_mean: the mean of the latent variable distribution
    z_std: the mean of the latent variable distribution
    how_many: the number of functions to predict
    numpy: convert torch tensor to numpy array
    """
    if device:
        z = sample_z(z_mean, z_std, how_many, device=device) 
        y_pred, _ = decoder(inputs, z)
        if numpy:
            return y_pred.cpu().detach().numpy()
        else:
            return y_pred
    else:
        z = sample_z(z_mean, z_std, how_many) 
        y_pred, _ = decoder(inputs, z)
        if numpy:
            return y_pred.detach().numpy()
        else:
            return y_pred



