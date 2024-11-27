import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

########################################################################
####################### WiFlexFormer architecture ######################
########## Paper: https://doi.org/10.48550/arXiv.2411.04224 ############
########################################################################

class WiFlexFormer(nn.Module):
    def __init__(self, 
                 input_dim: int=52, # number of subcarriers
                 num_channels:int=1, # 1 channel for amplitude features
                 feature_dim: int=32, # feature dimension
                 num_heads: int=16, # number of attention heads
                 num_layers: int=4,  # number of transformer encoder layers
                 dim_feedforward: int=64, # feedforward dimension
                 window_size: int=351, # number of wifi packets 
                 K=10, # number of Gaussian kernels
                 num_classes: int=3, # number of classes
                 ):
        super(WiFlexFormer, self).__init__()

        # Single-channel stem (for amplitude features)
        self.stem1D = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1), 
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1))
        
        # Multi-channel stem (e.g., for DFS features)
        self.stem2D = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU())
        
        # Class token embeddings
        self.class_token_embeddings = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Encoder
        self.pos_encoding = Gaussian_Position(feature_dim, window_size, K=K)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=0.3,batch_first=False) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,enable_nested_tensor=False)

        # Output linear layer
        self.class_token_output = nn.Linear(feature_dim, num_classes)

    def forward(self, x):

        if x.shape[1] == 1: # single-channel input path
            x = einsum(x, 'b c f t -> b f t')
            x = self.stem1D(x)
            x = rearrange(x, 'b f t -> b t f')

        else: # multi-channel input path
            x = self.stem2D(x)
            x = einsum(x, 'b c f t -> b f t')
            x = self.stem1D(x)
            x = rearrange(x, 'b f t -> b t f')
  
        # Gaussian positional encoding
        x = self.pos_encoding(x) 
        x = rearrange(x, 'b t f -> t b f')   
        
        # Add class token to the input sequence
        class_tokens = self.class_token_embeddings.expand(-1, x.size(1), -1)
        x = torch.cat((class_tokens, x), dim=0)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        return self.class_token_output(x[0, :, :])
    

# Source code from: https://github.com/windofshadow/THAT/blob/main/TransCNN.py
class Gaussian_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Position, self).__init__()
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)

        # Assume total_size corresponds to the sequence length in x
        self.total_size = total_size

        # Setup Gaussian distribution parameters
        positions = torch.arange(total_size).unsqueeze(1).repeat(1, K)
        self.register_buffer('positions', positions)
        
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(s)
            s += interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones([1, K], dtype=torch.float) * 50.0, requires_grad=True)

    def forward(self, x):
        # Ensure input x has shape [batch_size, seq_length, d_model]
        batch_size, seq_length, d_model = x.shape

        # Check if total_size matches seq_length, if not adjust positions or error out
        assert self.total_size == seq_length, "total_size must match seq_length of input x"

        # Calculate Gaussian distribution values
        M = normal_pdf(self.positions, self.mu, self.sigma)  # Assuming this function is defined correctly
        
        # Positional encodings
        pos_enc = torch.matmul(M, self.embedding)  # [seq_length, d_model]

        # Expand pos_enc to match the batch size in x
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_length, d_model]

        # Add position encodings to input x
        return x + pos_enc
        
def normal_pdf(pos, mu, sigma): 
    a = pos - mu
    log_p = -1*torch.mul(a, a)/(2*sigma) - torch.log(sigma)/2
    return F.softmax(log_p, dim=1)
    
    
