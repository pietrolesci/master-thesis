import torch

### h() ###
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, h_specs):
        """input_dim: number of features + labels per exaple"""
        super().__init__()        
        self.input_dim = input_dim
        self.h_specs = h_specs
        for i in range(len(self.h_specs)):
            if i == 0:    
                setattr(self, 'h_layer' + str(i), torch.nn.Linear(self.input_dim, self.h_specs[i][0]))
                if self.h_specs[i][1]:
                    setattr(self, 'h_layer' + str(i) + '_act', self.h_specs[i][1])
            else:
                setattr(self, 'h_layer' + str(i), torch.nn.Linear(self.h_specs[i-1][0], self.h_specs[i][0]))
                if self.h_specs[i][1]:
                    setattr(self, 'h_layer' + str(i) + '_act', self.h_specs[i][1]) 
        
    def forward(self, xy_context):        
        r_i = xy_context
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('h'):
                r_i = layer_func(r_i)
        return r_i
    
    
### aggregate the r_i ###
def aggregate(r_i):
    r = torch.mean(r_i, dim=0)
    return r
       
    
### from r to z_mean, z_logvar
class Zparams(torch.nn.Module):
    def __init__(self, r_dim, z_dim):
        super().__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.r_to_mean = torch.nn.Linear(self.r_dim, self.z_dim)
        self.r_to_logvar = torch.nn.Linear(self.r_dim, self.z_dim)
        self.softplus = torch.nn.Softplus()

    def forward(self, r):
        z_mean = self.r_to_mean(r)
        z_logvar = self.softplus(self.r_to_logvar(r))
        return z_mean, z_logvar
    

### Generate samples from z ###
def sample_z(z_mean, z_logvar, how_many):
    for i in range(how_many):
        if i == 0:
            z_std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(z_std) 
            z_sample = eps.mul(z_std).add_(z_mean)
            z_samples = z_sample.unsqueeze(0)
        else:
            z_std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(z_std) 
            z_sample = eps.mul(z_std).add_(z_mean)
            z_samples = torch.cat([z_sample.unsqueeze(0), z_samples], dim=0)
    return z_samples
    
    
### g() ###
class Decoder(torch.nn.Module):
    def __init__(self, input_dim, g_specs):
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
    
    def forward(self, xz):
        y_target_mean = xz
        for layer_name, layer_func in self._modules.items():
            if layer_name.startswith('g'):
                y_target_mean = layer_func(y_target_mean)
                
        y_target_constant_std = 0.005
        return y_target_mean, y_target_constant_std
    
