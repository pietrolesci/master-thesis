# Architecture for the Neural Process

class NeuralProcess:
    
    def __init__():
        self.encoder_h_params = None
        self.decoder_g_params = None
        self.dim_r = None
        self.dim_z = None
        self.gp_mean_func = None
        self.gp_cov_func = None
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        NeuralProcessParams = namedtuple('NeuralProcessParams', 
                                 ['dim_r', 'dim_z', 'n_hidden_units_h', 'n_hidden_units_g'])




























# encoder h -- map inputs (x_i, y_i) to r_i
def encoder_h(context, dims):
    """
    context: tf.placeholder of x_c and y_c concatenated
    dims: list of the number of units for each layer; number of layers automatically inferred    
    """
    nn = tf.layers.dense(context, dims[0],
                         activation=tf.nn.relu,
                         name='encoder_layer_1',
                         reuse=tf.AUTO_REUSE,
                         kernel_initializer='normal')
    
    
    for n_layer, n_units in enumerate(dims[1:]):
        nn = tf.layers.dense(hidden_layer, 
                             n_units,
                             activation=tf.nn.relu,
                             name='encoder_layer_{}'.format(n_layer+2),
                             reuse=tf.AUTO_REUSE,
                             kernel_initializer='normal')
        
    return r


def encoder_h(context_xys: tf.Tensor, params: NeuralProcessParams) -> tf.Tensor:
    """Map context inputs (x_i, y_i) to r_i

    Creates a fully connected network with a single sigmoid hidden layer and linear output layer.

    Parameters
    ----------
    context_xys
        Input tensor, shape: (n_samples, dim_x + dim_y)
    params
        Neural process parameters

    Returns
    -------
        Output tensor of encoder network
    """
    
    
    hidden_layer = context_xys
    # First layers are relu
    for i, n_hidden_units in enumerate(params.n_hidden_units_h):
        hidden_layer = 

    # Last layer is simple linear
    i = len(params.n_hidden_units_h)
    r = tf.layers.dense(hidden_layer, params.dim_r,
                        name='encoder_layer_{}'.format(i),
                        reuse=tf.AUTO_REUSE,
                        kernel_initializer='normal')
    
    return r


x = tf.placeholder(shape=[None, 3], dtype=tf.float32)
nn = tf.layers.dense(x, 3, activation=tf.nn.sigmoid)
nn = tf.layers.dense(nn, 5, activation=tf.nn.sigmoid)
encoded = tf.layers.dense(nn, 2, activation=tf.nn.sigmoid)
nn = tf.layers.dense(encoded, 5, activation=tf.nn.sigmoid)
nn = tf.layers.dense(nn, 3, activation=tf.nn.sigmoid)

