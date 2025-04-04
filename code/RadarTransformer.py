import torch
import torch.nn as nn
import math

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_transpose=False):
        super(ConvLayer, self).__init__()
        
        def calculate_conv_padding(kernel_size, stride):
            return  (- stride + kernel_size) // 2
        
        padding = calculate_conv_padding(kernel_size, stride)

        if is_transpose:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def chunkify_with_overlap(x, chunk_size=200, overlap=50):
    
    N, L, C = x.size()
    chunks = []
    
    step = chunk_size - overlap  
    start = 0
    
    while start < L:
        end = start + chunk_size
        if end > L:
            end = L  

        chunk_data = x[:, start:end, :]
        chunks.append((start, end, chunk_data))
        
        if end == L:
            break
        
        start += step
    
    return chunks


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        
    def forward(self, src):
       
        src = self.input_projection(src) 
        
        src = self.positional_encoding(src)
        
        tgt = src.clone()
        
        out = self.transformer(src, tgt)  # [N, L, d_model]

        out = self.output_projection(out)  # [N, L, 1]

        return out

class TimeSeriesBiLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.1):
        super(TimeSeriesBiLSTM, self).__init__()
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,    
            hidden_size=hidden_dim,  
            num_layers=num_layers,   
            dropout=dropout,
            batch_first=True,        
            bidirectional=True       
        )
        

    def forward(self, x):
        """
        x: [N, L, input_dim]
        out: [N, L, 1]
        """
        
        out, (hn, cn) = self.lstm(x)  
       
        return out
   
class TimeSeriesTransformerEncoder(nn.Module):
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, 
                 dim_feedforward=256, dropout=0.1):
        super(TimeSeriesTransformerEncoder, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers=num_layers
        )
                
  
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        
        x = self.input_projection(x)
        
        x = self.pos_encoder(x)
        
        out = self.transformer_encoder(x)
        
        out = self.output_projection(out)
        
        return out


class RadarModel(nn.Module):
    def __init__(self,input_length=10000):
        super(RadarModel, self).__init__()

        self.input_length = input_length

        kernel_size_1 = 32
        stride_1 = 10 
        self.conv1 = ConvLayer(8, 64, kernel_size_1, stride_1)
        
        kernel_size_2 = 64
        stride_2 = 2  
        self.conv2 = ConvLayer(64, 128, kernel_size_2, stride_2)

        kernel_size_3 = 32
        stride_3 = 2 
        self.conv3 = ConvLayer(128, 256, kernel_size_3, stride_3)

        kernel_size_4 = 32
        stride_4 = 2
        self.deconv1 = ConvLayer(256, 128, kernel_size_4, stride_4, is_transpose=True)

        kernel_size_5 = 64
        stride_5 = 2
        self.deconv2 = ConvLayer(128, 64, kernel_size_5, stride_5, is_transpose=True)

        kernel_size_6 = 9
        stride_6 = 1
        self.deconv3 = ConvLayer(64, 64, kernel_size_6, stride_6, is_transpose=True)

        self.transformer_encoder = TimeSeriesTransformerEncoder(
            input_dim=128,
            d_model=128,
            nhead=16,
            num_layers=3,
            dim_feedforward=1024,
            dropout=0.1
        )


        self.lstm = TimeSeriesBiLSTM(
            input_dim=64,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        )


    @staticmethod
    def normalize_amplitude(x, min_val=-1, max_val=1):
        
        batch_min = x.min(dim=2, keepdim=True)[0]  # [N,1,1]
        batch_max = x.max(dim=2, keepdim=True)[0]  # [N,1,1]
        
        denominator = (batch_max - batch_min)  # [N,1,1]
        
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        
        x_norm = (x - batch_min) / denominator
        
        x_norm = x_norm * (max_val - min_val) + min_val
        
        return x_norm

    def forward(self, x):
       
        x_conv1 = self.conv1(x)

        x_conv2 = self.conv2(x_conv1)

        x_conv3 = self.conv3(x_conv2)

        x_deconv1 = self.deconv1(x_conv3)

        x_add1 = torch.add(x_deconv1, x_conv2)

        x_deconv2 = self.deconv2(x_add1)

        x_add2 = torch.add(x_deconv2, x_conv1)

        x_deconv3 = self.deconv3(x_add2)

        x_perm = x_deconv3.permute(0, 2, 1)

        x_lstm = self.lstm(x_perm)
        
        x_transformer = self.transformer_encoder(x_lstm)

        x_final = x_transformer.permute(0, 2, 1)  # => [N, 1, L]
        
        return x_final


if __name__ == "__main__":
    
    model = RadarModel()
    
    model.eval()
    
    from torch.utils.tensorboard import SummaryWriter
    
    with torch.no_grad():
        writer = SummaryWriter('C:/TensorBoard/runs/radar_model')
        dummy_input = torch.randn(1, 8, 10000).cpu()
        
        traced_model = torch.jit.trace(model, dummy_input)
        writer.add_graph(traced_model, dummy_input)
        writer.close()