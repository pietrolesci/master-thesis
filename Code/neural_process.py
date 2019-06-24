import torch
import numpy as np



# class Swish(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.beta = torch.nn.Parameter(torch.ones(1, num_parameters))

#     def forward(self, input):
#         return input * torch.sigmoid(input * self.beta)



class Encoder(torch.nn.Module):
    def __init__(self, input_dim, encoder_specs, init_func=torch.nn.init.normal_):
        """input_dim: dimension of x_i + dimension of y_i, for i in Context"""
        super().__init__()        
        self.input_dim = input_dim
        self.encoder_specs = encoder_specs
        self.init_func = init_func
        for i in range(len(self.encoder_specs)):
            if i == 0:    
                self.add_module('h_layer' + str(i), torch.nn.Linear(self.input_dim, self.encoder_specs[i][0]))
                if self.encoder_specs[i][1]:
                    self.add_module('h_layer' + str(i) + '_act', self.encoder_specs[i][1])
            else:
                self.add_module('h_layer' + str(i), torch.nn.Linear(self.encoder_specs[i-1][0], self.encoder_specs[i][0]))
                if self.encoder_specs[i][1]:
                    self.add_module('h_layer' + str(i) + '_act', self.encoder_specs[i][1]) 
        if init_func:
            for layer_name,_ in self._modules.items():
                if layer_name.endswith('act') == False:
                    init_func(getattr(getattr(self, layer_name), 'weight'))
        
    def forward(self, x, y):   
        r_i = torch.cat([x, y], dim=1)
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('h'):
                r_i = layer_func(r_i)
        r = r_i.mean(dim=0)
        return r

    
    
# From r to z_mean, z_logvar
class Zparams(torch.nn.Module):
    def __init__(self, r_dim, z_dim):
        super().__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.r_to_mean = torch.nn.Linear(self.r_dim, self.z_dim)
        self.r_to_logvar = torch.nn.Linear(self.r_dim, self.z_dim)
        self.softplus = torch.nn.Softplus()

    def forward(self, r):
        z_mean = self.r_to_mean(r).unsqueeze(-1)
        z_logvar = self.softplus(self.r_to_logvar(r)).unsqueeze(-1)
        z_std = torch.exp(0.5 * z_logvar)
        return z_mean, z_std
    
    

class Decoder(torch.nn.Module):
    def __init__(self, input_dim, decoder_specs, init_func=torch.nn.init.normal_):
        """input_dim: number of features + dimesion of z"""
        super().__init__()
        self.input_dim = input_dim
        self.decoder_specs = decoder_specs
        for i in range(len(self.decoder_specs)):
            if i == 0:    
                self.add_module('g_layer' + str(i), torch.nn.Linear(self.input_dim, self.decoder_specs[i][0]))
                if self.decoder_specs[i][1]:
                    self.add_module('g_layer' + str(i) + '_act', self.decoder_specs[i][1])
            else:
                self.add_module('g_layer' + str(i), torch.nn.Linear(self.decoder_specs[i-1][0], self.decoder_specs[i][0]))
                if self.decoder_specs[i][1]:
                    self.add_module('g_layer' + str(i) + '_act', self.decoder_specs[i][1])    
        if init_func:
            for layer_name,_ in self._modules.items():
                if layer_name.endswith('act') == False:
                    init_func(getattr(getattr(self, layer_name), 'weight'))
        self.softplus = torch.nn.Softplus()
    
    def forward(self, x, z):
        z_reshape =  z.t().unsqueeze(1).expand(-1, x.shape[0], -1)
        x_reshape = x.unsqueeze(0).expand(z_reshape.shape[0], x.shape[0], x.shape[1])
        x_concat_z = torch.cat([x_reshape, z_reshape], dim=2)
        y_mean = x_concat_z
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('g'):
                y_mean = layer_func(y_mean)

        y_logvar = x_concat_z
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('g'):
                y_logvar = layer_func(y_logvar)
        y_logvar = self.softplus(y_logvar)
        y_std = torch.exp(0.5 * y_logvar)
        return y_mean, y_std
    
    


### UTILS ###

# Generate samples from z
def sample_z(z_mean, z_std, how_many, device=None):
    """
    Returns a sample from z of size (z_dim, how_many)
    """
    z_dim = z_std.shape[0]
    if device:
        eps = torch.randn([z_dim, how_many]).to(device)
    else:
        eps = torch.randn([z_dim, how_many])
    z_samples = z_mean + z_std * eps
    return z_samples



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



