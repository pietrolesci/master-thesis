import torch
import numpy as np

### ENCODER ###

# Self-Attention
class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, r_dim):
        """The three linear must be indepedent"""
        super().__init__()
        self.r_dim = r_dim
        self.q = torch.nn.Linear(input_dim, r_dim)
        self.k = torch.nn.Linear(input_dim, r_dim)
        self.v = torch.nn.Linear(input_dim, r_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        sdp = torch.mm(Q, K.t()) / np.sqrt(self.r_dim) # scaled dot product
        W = self.softmax(sdp)
        r_i = torch.matmul(W, V)
        return r_i
    

# Cross-Attention
class CrossAttention(torch.nn.Module):
    def __init__(self, context_indim, target_indim, r_dim):
        """The three linear must be indepedent"""
        super().__init__()
        self.r_dim = r_dim
        self.q = torch.nn.Linear(target_indim, r_dim)
        self.k = torch.nn.Linear(context_indim, r_dim)
        self.v = torch.nn.Linear(r_dim, r_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, query, keys, values):
        Q = self.q(query)
        K = self.k(keys)
        V = self.v(values)
        sdp = torch.mm(Q, K.t()) / np.sqrt(self.r_dim) # scaled dot product
        W = self.softmax(sdp)
        r = torch.matmul(W, V).mean(dim=0)
        return r
    
    
# Deterministic path
class R_encoder(torch.nn.Module):
    def __init__(self, r_dim, r_encoder_specs, init_func=torch.nn.init.normal_):
        """r_dim: number of features + labels per exaple"""
        super().__init__()        
        self.r_dim = r_dim
        self.r_encoder_specs = r_encoder_specs
        self.init_func = init_func
        for i in range(len(self.r_encoder_specs)):
            if i == 0:    
                self.add_module('h_layer' + str(i), torch.nn.Linear(self.r_dim, self.r_encoder_specs[i][0]))
                if self.r_encoder_specs[i][1]:
                    self.add_module('h_layer' + str(i) + '_act', self.r_encoder_specs[i][1])
            else:
                self.add_module('h_layer' + str(i), torch.nn.Linear(self.r_encoder_specs[i-1][0], self.r_encoder_specs[i][0]))
                if self.r_encoder_specs[i][1]:
                    self.add_module('h_layer' + str(i) + '_act', self.r_encoder_specs[i][1]) 
        if init_func:
            for layer_name,_ in self._modules.items():
                if layer_name.endswith('act') == False:
                    init_func(getattr(getattr(self, layer_name), 'weight'))
        
    def forward(self, xy_context):        
        r_i = xy_context
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('h'):
                r_i = layer_func(r_i)
        return r_i

    
    
# Latent path
class S_encoder(torch.nn.Module):
    def __init__(self, input_dim, s_encoder_specs, init_func=torch.nn.init.normal_):
        """input_dim: number of features + labels per exaple"""
        super().__init__()        
        self.input_dim = input_dim
        self.s_encoder_specs = s_encoder_specs
        self.init_func = init_func
        for i in range(len(self.s_encoder_specs)):
            if i == 0:    
                self.add_module('h_layer' + str(i), torch.nn.Linear(self.input_dim, self.s_encoder_specs[i][0]))
                if self.s_encoder_specs[i][1]:
                    self.add_module('h_layer' + str(i) + '_act', self.s_encoder_specs[i][1])
            else:
                self.add_module('h_layer' + str(i), torch.nn.Linear(self.s_encoder_specs[i-1][0], self.s_encoder_specs[i][0]))
                if self.s_encoder_specs[i][1]:
                    self.add_module('h_layer' + str(i) + '_act', self.s_encoder_specs[i][1]) 
        if init_func:
            for layer_name,_ in self._modules.items():
                if layer_name.endswith('act') == False:
                    init_func(getattr(getattr(self, layer_name), 'weight'))
        
    def forward(self, xy_context):        
        s_i = xy_context
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('h'):
                s_i = layer_func(s_i)
        return s_i
    
    
class Zparams(torch.nn.Module):
    def __init__(self, s_dim, z_dim):
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.s_to_mean = torch.nn.Linear(self.s_dim, self.z_dim)
        self.s_to_logvar = torch.nn.Linear(self.s_dim, self.z_dim)
        self.softplus = torch.nn.Softplus()

    def forward(self, s):
        z_mean = self.s_to_mean(s).unsqueeze(-1)
        z_logvar = self.softplus(self.s_to_logvar(s)).unsqueeze(-1)
        return z_mean, z_logvar
    
    
    

    
    
### DECODER ###
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
    
    def forward(self, xz):
        y_mean = xz
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('g'):
                y_mean = layer_func(y_mean)
        y_logvar = xz
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('g'):
                y_logvar = layer_func(y_logvar)
        y_logvar = self.softplus(y_logvar)
        return y_mean, y_logvar
    

    
    
    
    
### UTILS ###

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



# Negative log-likelihood
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

