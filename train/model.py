import torch
import torch.nn as nn
import math
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 添加Layer Normalization
        self.ln = nn.LayerNorm(hidden_size)
        
        # 全连接层 - 包含一个隐藏层，增加非线性映射能力
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_size // 2, output_size)
            nn.Linear(hidden_size // 2, output_size)
            # 不再使用ReLU激活函数在最后一层，适用于回归任务
        )
        
        # 初始化LSTM权重
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM层前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 使用Layer Normalization
        out = self.ln(out[:, -1, :])
        
        # 全连接层前向传播
        out = self.fc(out)
        # out = out.view(out.size(0), 5, 10)  # Reshape output to (batch_size, 5, 10)    
        return out



class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(ImprovedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bi-directional LSTM for better context understanding
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Layer Normalization after LSTM
        self.ln = nn.LayerNorm(hidden_size * 2)  # because of bidirectional

        # Attention layer (optional but very useful for sequence focus)
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: [B, T, 2H]

        # Layer Normalization
        out = self.ln(out)  # normalize across hidden features

        # Attention mechanism: weighted sum of all time steps
        attn_weights = self.attn(out)  # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # [B, T, 1]
        out = torch.sum(out * attn_weights, dim=1)  # [B, 2H]

        # Fully connected layers
        out = self.fc(out)  # [B, output_size]
        return out
 

    


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 添加Layer Normalization
        self.ln = nn.LayerNorm(hidden_size)
        
        # 全连接层 - 包含一个隐藏层，增加非线性映射能力
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_size // 2, output_size)
            nn.Linear(hidden_size // 2, output_size)
            # 不再使用ReLU激活函数在最后一层，适用于回归任务
        )
        
        # 初始化LSTM权重
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM层前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 使用Layer Normalization
        out = self.ln(out[:, -1, :])
        
        # 全连接层前向传播
        out = self.fc(out)
        # out = out.view(out.size(0), 5, 10)  # Reshape output to (batch_size, 5, 10)    
        return out

# # Example usage:
# # model = LSTM(input_size=4, hidden_size=256, num_layers=3, output_size=10, dropout=0.1)
# # print(model)

class Attention(nn.Module):
    def __init__(self,input_size, d_model, seq_len, dropout):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(input_size, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2
        )
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6),
        )

    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)
        # out: (batch, n_output)
        out = self.pred_layer(stats)
        return out


class AttentionLSTM(nn.Module):
    def __init__(self,
                 d_channel,
                 d_temporal,
                 d_lstm_hidden,
                 lstm_num_layers,
                 window_length,
                 dropout):
        super().__init__()
        self.channelEncoder = nn.Linear(5, d_channel)
        self.channelwiseAttentionLayer = nn.TransformerEncoderLayer(
            d_model=window_length, dim_feedforward=512, nhead=8, dropout=dropout)
        self.lstm = nn.LSTM(input_size=d_channel,
                            hidden_size=d_lstm_hidden,
                            num_layers=lstm_num_layers)
        self.predLayer = nn.Sequential(
            nn.Linear(d_lstm_hidden, d_lstm_hidden),
            nn.ReLU(),
            nn.Linear(d_lstm_hidden, 6))

    def forward(self, sensors):
        """
        About Pytorch LSTM:
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
        args:
          sensors: (batch size, length, 5)
        return:
          out: (batch size, angles)
        """
        # input: (batch size, length, 5)
        # (batch size, length, d_channel)
        out = self.channelEncoder(sensors)
        # (d_channel, batch size, length)
        out = out.permute(2, 0, 1)
        # (d_channel, batch size, length)
        out = self.channelwiseAttentionLayer(out)   
        # (length， batch size, d_channel)
        out = out.permute(2, 1, 0)
        # (length, batch size, d_lstm_hidden)
        out, (hn, cn) = self.lstm(out)
        # (batch size, d_lstm_hidden)
        out = out[-1, :, :]
         # (batch size, num_angles)
        out = self.predLayer(out)                  

        return out
    

class PositionalEncoding(nn.Module):
    """位置编码模块，为序列添加位置信息"""
    def __init__(self, d_model, max_seq_length=80, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算sin和cos位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区而不是模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量，形状 [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class SensorToJointTransformer(nn.Module):
    """
    Transformer模型，用于将传感器数据映射到关节角度
    
    参数:
        input_size: 传感器数量
        output_size: 关节角度数量 (10)
        d_model: Transformer模型的维度
        nhead: 多头注意力中头的数量
        num_encoder_layers: Transformer编码器层数
        num_decoder_layers: Transformer解码器层数
        dim_feedforward: 前馈网络的隐藏层维度
        dropout: Dropout概率
        seq_length: 序列长度 (80)
    """
    def __init__(self, input_size, output_size=10, d_model=64, nhead=4, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=128, dropout=0.1, seq_length=80):
        super(SensorToJointTransformer, self).__init__()
        
        self.input_size = input_size  # 传感器数量
        self.output_size = output_size  # 关节角度数量
        self.d_model = d_model
        self.seq_length = seq_length
        
        # 输入映射层：将传感器数据(5个维度)映射到模型维度
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, seq_length, dropout)
        
        # Transformer编码器和解码器
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出映射层：将模型维度映射到关节角度
        self.output_mapping = nn.Linear(d_model, output_size)
        
        # 初始化解码器输入的参数(可学习)
        self.decoder_input = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, src):
        """
        参数:
            src: 输入的传感器数据, 形状 [batch_size, seq_length, input_size]
            
        返回:
            关节角度预测, 形状 [batch_size, output_size]
        """
        batch_size = src.size(0)
        
        # 将输入映射到模型维度
        src = self.input_embedding(src)  # [batch_size, seq_length, d_model]
        
        # 添加位置编码
        src = self.positional_encoding(src)  # [batch_size, seq_length, d_model]
        
        # 创建掩码
        src_mask = None
        src_key_padding_mask = None
        
        # 创建解码器输入（批量化）
        tgt = self.decoder_input.expand(batch_size, 1, self.d_model)
        tgt = tgt.repeat(1, self.seq_length, 1)  # [batch_size, seq_length, d_model]
        
        # 添加位置编码到解码器输入
        tgt = self.positional_encoding(tgt)
        
        # 创建解码器掩码（防止看到未来时刻）
        tgt_mask = self.transformer.generate_square_subsequent_mask(self.seq_length).to(src.device)
        tgt_key_padding_mask = None
        
        # 通过Transformer
        output = self.transformer(
            src, tgt, 
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch_size, seq_length, d_model]
        
        # 我们只关心最后一个时间步的输出
        output = output[:, -1, :]  # [batch_size, d_model]
        
        # 映射到关节角度
        joint_angles = self.output_mapping(output)  # [batch_size, output_size]
        
        return joint_angles
        
    def predict(self, sensor_data):
        """
        使用模型预测关节角度
        
        参数:
            sensor_data: 传感器数据, 形状 [batch_size, seq_length, input_size]
            
        返回:
            关节角度预测, 形状 [batch_size, output_size]
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            joint_angles = self.forward(sensor_data)
        return joint_angles