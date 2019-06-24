import torch
import numpy as np


### h() ###
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, h_specs, init_func=torch.nn.init.normal_):
        """input_dim: number of features + labels per exaple"""
        super().__init__()        
        self.input_dim = input_dim
        self.h_specs = h_specs
        self.init_func = init_func
        for i in range(len(self.h_specs)):
            if i == 0:    
                setattr(self, 'h_layer' + str(i), torch.nn.Linear(self.input_dim, self.h_specs[i][0]))
                if self.h_specs[i][1]:
                    setattr(self, 'h_layer' + str(i) + '_act', self.h_specs[i][1])
            else:
                setattr(self, 'h_layer' + str(i), torch.nn.Linear(self.h_specs[i-1][0], self.h_specs[i][0]))
                if self.h_specs[i][1]:
                    setattr(self, 'h_layer' + str(i) + '_act', self.h_specs[i][1]) 
        
        if init_func:
            for layer_name,_ in self._modules.items():
                if layer_name.endswith('act') == False:
                    init_func(getattr(getattr(self, layer_name), 'weight'))
        
    def forward(self, xy_context):        
        r_i = xy_context
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
        return z_mean, z_logvar
    
    
### g() ###
class Decoder(torch.nn.Module):
    def __init__(self, input_dim, g_specs, init_func=torch.nn.init.normal_):
        """input_dim: number of features + dimesion of z"""
        super().__init__()
        self.input_dim = input_dim
        self.g_specs = g_specs
        for i in range(len(self.g_specs)):
            if i == 0:    
                setattr(self, 'g_layer' + str(i), torch.nn.Linear(self.input_dim, self.g_specs[i][0]))
                if self.g_specs[i][1]:
                    setattr(self, 'g_layer' + str(i) + '_act', self.g_specs[i][1])
            else:
                setattr(self, 'g_layer' + str(i), torch.nn.Linear(self.g_specs[i-1][0], self.g_specs[i][0]))
                if self.g_specs[i][1]:
                    setattr(self, 'g_layer' + str(i) + '_act', self.g_specs[i][1])    
                    
        if init_func:
            for layer_name,_ in self._modules.items():
                if layer_name.endswith('act') == False:
                    init_func(getattr(getattr(self, layer_name), 'weight'))
    
    def forward(self, xz):
        y_target_mean = xz
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('g'):
                y_target_mean = layer_func(y_target_mean)
                
        y_target_constant_std = 0.005
        return y_target_mean, y_target_constant_std
    

# Splits the dataset randomly
def randsplit_totensor(x, y, n_context):
    index = np.arange(x.shape[0])
    mask = np.random.choice(index, size=n_context, replace=False)
    x_context = torch.from_numpy(x[mask])
    y_context = torch.from_numpy(y[mask])
    x_target = torch.from_numpy(np.delete(x, mask, axis=0))
    y_target = torch.from_numpy(np.delete(y, mask, axis=0))
    return x_context, y_context, x_target, y_target


# Generate samples from z
def sample_z(z_mean, z_logvar, how_many):
    """
    Returns a sample from z of size (how_many, z_dim)
    """
    z_std = torch.exp(0.5 * z_logvar)
    for i in range(how_many):
        if i == 0:
            eps = torch.randn_like(z_std) 
            z_samples = eps.mul(z_std).add_(z_mean).t()
        else:
            eps = torch.randn_like(z_std) 
            z_sample = eps.mul(z_std).add_(z_mean).t()
            z_samples = torch.cat([z_sample, z_samples], dim=0)
    return z_samples


# Log-likelihood
def neg_loglik(mu, std, y_t):
    norm = torch.distributions.Normal(mu, std)
    NLL = -norm.log_prob(y_t.squeeze(-1)).mean(dim=0).sum()
    return NLL


# KL divergence
def KL_div(mu_q, std_q, mu_p, std_p):
    """Analytical KLD between 2 Gaussians."""
    qs2 = std_q**2 + 1e-16
    ps2 = std_p**2 + 1e-16
    
    return (qs2/ps2 + ((mu_q-mu_p)**2)/ps2 + torch.log(ps2/qs2) - 1.0).sum()*0.5

