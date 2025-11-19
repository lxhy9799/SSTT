import math

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from torch.utils.data import DataLoader
from thop import profile
import numpy as np
from loader import ngsimDataset

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.use_cuda = args.use_cuda
        self.train_flag = args.train_flag
        self.use_maneuvers = args.use_maneuvers
        self.use_true_man = args.use_true_man
        # IO Setting
        self.in_length = args.in_length
        self.out_length = args.out_length
        self.num_lat_classes = args.num_lat_classes
        self.num_lon_classes = args.num_lon_classes
        self.num_features = args.num_features
        self.num_opt = args.num_opt

        # Sizes of network layers
        self.ff_hide_size = args.ff_hidden_size
        self.encoder_size = args.lstm_encoder_size
        self.decoder_size = args.decoder_size
        self.blocks = args.num_blocks
        self.n_head = args.num_heads
        self.input_embed_size = args.input_embed_size
        self.soc_conv_depth = args.soc_conv_depth
        self.conv_3x1_depth = args.conv_3x1_depth
        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
        self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        # Activations:
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.linear_motion = nn.Linear(self.num_features, self.input_embed_size)
        self.enc_lstm = nn.LSTM(self.input_embed_size, self.encoder_size)
        self.casual_sparse_temporal = nn.ModuleList()
        self.sparse_spatial = nn.ModuleList()
        self.addnorm = AddAndNorm(self.encoder_size * self.blocks)
        self.window_size = []
        for i in range(1, self.blocks):
            self.window_size.append(2 ** i)
        self.window_size.append(16)
        for b in range(self.blocks):
            self.casual_sparse_temporal.append(
                LocalGlobalTemporalTransformer(
                    window_size=self.window_size[b],
                    num_time=self.in_length, args=args))

        for b in range(self.blocks):
            self.sparse_spatial.append(
                SparseSpatialTransformer(args)
            )
        # decoding
        self.op_lat = torch.nn.Linear(
            self.encoder_size,
            self.num_lat_classes)
        self.op_lon = torch.nn.Linear(
            self.encoder_size,
            self.num_lon_classes)
        self.op_mean = torch.nn.Linear(
            self.encoder_size,
            2)
        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(
                self.encoder_size,
                self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(
                self.encoder_size * self.blocks,
                self.decoder_size)
        self.op = torch.nn.Linear(self.decoder_size, 5)
        self.mu_fc1 = nn.Linear(self.encoder_size * self.blocks, self.n_head * self.encoder_size)
        self.mu_fc = nn.Linear(self.n_head * self.encoder_size, self.encoder_size)
        self.mapping = torch.nn.Parameter(
            torch.Tensor(self.in_length, self.out_length, self.num_lat_classes + self.num_lon_classes))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)  # Glorot init
        self.activation = nn.ELU()
        self.normalize = nn.LayerNorm(self.encoder_size)
        self.dec_linear = nn.Linear(self.encoder_size * self.blocks + self.num_lat_classes + self.num_lon_classes,
                                    self.encoder_size)

    def forward(self, hist, nbrs, mask, va, nbrsva, cls, nbrscls, lat_enc, lon_enc):
        """
        :param hist: T b d
        :param nbrs: T count d
        :param mask: b 3 13 d
        :param lat_enc: b 3
        :param lon_enc: b 3
        :return:
        final_pred： T b 5
        lat_pred：b 3
         lon_pred：b 3
        """
        # hist:
        # input embedding
        hist = torch.cat((hist, cls, va), -1)
        nbrs = torch.cat((nbrs, nbrscls, nbrsva), -1)

        hist_enc, (hist_hidden_state, _) = self.enc_lstm(
            self.leaky_relu(
                self.linear_motion(hist)))  # hist_enc:T  b encoder_size  hist_hidden_state:b, encoder_size
        hist_enc = hist_enc.permute(1, 0, 2)  # b t d

        # Forward pass nbrs
        nbrs_enc, (nbrs_hidden_state, _) = self.enc_lstm(
            self.leaky_relu(
                self.linear_motion(nbrs)))  # nbrs_enc:T  count  encoder_size  nbrs_hidden_state:b, encoder_size

        # scatter grid embeding
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
        mask = repeat(mask, 'b g s -> t b g s', t=self.in_length)
        soc_enc = torch.zeros_like(mask).float()  # b 3 13 s
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_enc)
        soc_enc = soc_enc.permute(1, 0, 2, 3)
        ##nbr pool
        spatial_list = []
        casual_temporal_list = []
        for i in range(self.blocks):
            sparse_spatial_enc,spatial_attn_weight= self.sparse_spatial[i](
                hist_enc,
                soc_enc) #256 16 64
            casual_sparse_temporal ,temporal_attn_weight= self.casual_sparse_temporal[i](sparse_spatial_enc)
            spatial_list.append(sparse_spatial_enc)
            casual_temporal_list.append(casual_sparse_temporal)
        spatial_d_stack = torch.stack(spatial_list)
        casual_temporal_d_stack = torch.stack(casual_temporal_list)
        num_blocks, B, T, D = spatial_d_stack.shape
        sparse_spatial_value = spatial_d_stack.permute(1, 2, 0, 3).reshape(B, T, num_blocks * D)
        casual_temporal_value = casual_temporal_d_stack.permute(1, 2, 0, 3).reshape(B, T, num_blocks * D)
        enc = self.addnorm(casual_temporal_value,sparse_spatial_value)  # b 16 D
        # enc = self.addnorm(casual_temporal_value)  # b 16 D

        if self.use_maneuvers:
            ## Maneuver recognition:
            maneuver_state = enc[:, -1, :]  # b d
            maneuver_state = self.activation(self.mu_fc1(maneuver_state))
            maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
            # mean_pred = self.op_mean(maneuver_state)  # b 2
            lat_pred = F.softmax(self.op_lat(maneuver_state), dim=-1)  # b 3
            lon_pred = F.softmax(self.op_lon(maneuver_state), dim=-1)  # b 3
            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                lat_man = torch.argmax(lat_pred, dim=-1).detach().unsqueeze(1)  # b 1
                lon_man = torch.argmax(lon_pred, dim=-1).detach().unsqueeze(1)
                lat_enc_tmp = torch.zeros_like(lat_pred)
                lon_enc_tmp = torch.zeros_like(lon_pred)
                lat_man = lat_enc_tmp.scatter_(1, lat_man, 1)
                lon_man = lon_enc_tmp.scatter_(1, lon_man, 1)
                index = torch.cat((lat_man, lon_man), dim=-1).transpose(1, 0)  # 6 b
                mapping = F.softmax(torch.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)  # b 25 16
                dec = torch.matmul(mapping, enc).permute(1, 0, 2)  # 25 b D
                fut_pred = self.decode(dec, lat_enc, lon_enc) #拼接意图
                return fut_pred, lat_pred, lon_pred
            else:
                final_pred = []
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_pred)
                        lon_enc_tmp = torch.zeros_like(lon_pred)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        index = torch.cat((lat_enc_tmp, lon_enc_tmp), dim=-1).transpose(1, 0)
                        mapping = F.softmax(torch.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
                        dec = torch.matmul(mapping, enc).permute(1, 0, 2)
                        fut_pred = self.decode(dec, lat_enc_tmp, lon_enc_tmp)
                        final_pred.append(fut_pred)
                return final_pred, lat_pred, lon_pred

        else:
            fut_pred = self.decode(enc,lat_enc, lon_enc)
            return fut_pred

    def decode(self, dec, lat_enc, lon_enc):
        if self.use_maneuvers:
            lat_enc = lat_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            lon_enc = lon_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            dec = torch.cat((dec, lat_enc, lon_enc), -1)
            dec = self.dec_linear(dec)
        h_dec, _ = self.dec_lstm(dec)
        h_dec = h_dec.permute(1, 0, 2)  # b , T ,D
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)  # T , b ,D
        fut_pred = outputActivation(fut_pred)  # T , b ,D
        return fut_pred


class SparseSpatialTransformer(nn.Module):
    def __init__(self,
                 args):
        super(SparseSpatialTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.n_head = args.num_heads
        self.att_out_size = args.att_out_size
        self.dim_in = args.lstm_encoder_size
        self.ff_hidden_size = args.ff_hidden_size
        self.sparse_spatial_attn = SparseSpatialAttention(dim_in=self.dim_in, num_heads=self.n_head,
                                                          att_out_size=self.att_out_size,
                                                          )
        self.add_and_norm = AddAndNorm(hidden_layer_size=self.dim_in)
        self.ff = FeedForward(dim=self.dim_in, hidden_dim=self.ff_hidden_size)

    def forward(self, x, soc_x):
        # batch,N, self.input_embed_dim
        sparse_spatial_attn, attn_weight= self.sparse_spatial_attn(x, soc_x)
        x1 = self.add_and_norm(sparse_spatial_attn, x)
        x2 = self.ff(x1)
        out = self.add_and_norm(x2, x1)
        return out, attn_weight


class SparseSpatialAttention(nn.Module):
    def __init__(self, dim_in, num_heads, att_out_size):
        super(SparseSpatialAttention, self).__init__()
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.att_out_size = att_out_size
        self.linear_q = nn.Linear(self.dim_in, self.num_heads * self.att_out_size)
        self.linear_k = nn.Linear(self.dim_in, self.num_heads * self.att_out_size)
        self.linear_v = nn.Linear(self.dim_in, self.num_heads * self.att_out_size)
        self.in_length = 16
        self._norm_fact = 1 / self.dim_in
        self.fc_o = nn.Linear(self.num_heads * self.att_out_size, self.dim_in)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.in_length, out_channels=64, kernel_size=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Conv2d(in_channels=64, out_channels=self.in_length, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()
        self.kesi = 0.3
    def forward(self, x, soc_x):
        """
        :param x: (batch, T, d)
        :param soc_x: (batch, T,n, d)
        :return:
        """
        query = self.linear_q(x).unsqueeze(2)  # (batch, T, 1,d)
        _, _, _, embed_size = query.shape
        keys = self.linear_k(soc_x).transpose(2, 3)  # b t d,39
        values = self.linear_v(soc_x)  # b 16 39, d
        attn_weight = (query @ keys) * self._norm_fact  # b t 1 39
        attn_weight = F.softmax(attn_weight, dim=-1)  # b t 1 39
        conv_attn_weight = self.sigmoid(self.conv2d(attn_weight))  # b 16 1 39
        zeros = torch.zeros_like(conv_attn_weight)
        R_spatial = torch.where(conv_attn_weight < self.kesi, zeros,
                        conv_attn_weight)  # b 16 1 39
        sparse_att_weight = attn_weight * R_spatial  # b 16 1 39
        sparse_values = sparse_att_weight @ values  # b 16 1 d
        sparse_values = self.fc_o(sparse_values.squeeze(2))  # b 16 self.dim_in
        return sparse_values, attn_weight


class LocalGlobalTemporalTransformer(nn.Module):
    def __init__(self,
                 window_size,  # the size of local window
                 num_time,  # the number of time slot
                 args):
        super(LocalGlobalTemporalTransformer, self).__init__()
        self.dim_in = args.lstm_encoder_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, self.dim_in))
        self.layers = nn.ModuleList([])
        self.window_size = window_size
        self.n_head = args.num_heads
        self.att_out_size = args.att_out_size
        self.ff_hidden_size = args.ff_hidden_size
        self.sparse_temporal_attn = SparseTemporalAttention(dim_in=self.dim_in, num_heads=self.n_head,
                                                            att_out_size=self.att_out_size, window_size=window_size
                                                            )
        self.add_and_norm = AddAndNorm(hidden_layer_size=self.dim_in)
        self.ff = FeedForward(dim=self.dim_in, hidden_dim=self.ff_hidden_size)
    def forward(self, x):
        # batch,T, self.input_embed_dim
        x = x + self.pos_embedding
        sparse_temporal_attn, att_weight = self.sparse_temporal_attn(x)
        x1 = self.add_and_norm(sparse_temporal_attn, x)
        x2 = self.ff(x1)
        out = self.add_and_norm(x2, x1)
        return out, att_weight


class SparseTemporalAttention(nn.Module):
    def __init__(self, dim_in, num_heads, att_out_size, window_size=2):
        super(SparseTemporalAttention, self).__init__()
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.att_out_size = att_out_size
        self.linear_q = nn.Linear(self.dim_in, self.num_heads * self.att_out_size)
        self.linear_k = nn.Linear(self.dim_in, self.num_heads * self.att_out_size)
        self.linear_v = nn.Linear(self.dim_in, self.num_heads * self.att_out_size)

        self.fc_o = nn.Linear(self.num_heads * self.att_out_size, self.dim_in)
        self._norm_fact = 1 / self.dim_in
        self.window_size = window_size
        self.sigmoid = nn.Sigmoid()
        self.in_length = 16
        self.encoder_size = self.dim_in
        if self.window_size > 0:
            self.conv1d = nn.Sequential(
                nn.Conv1d(in_channels=self.window_size, out_channels=self.window_size * 2, kernel_size=1),
                nn.Conv1d(in_channels=self.window_size * 2, out_channels=self.window_size * 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.window_size * 2, out_channels=self.window_size * 2, kernel_size=1),
                nn.Conv1d(in_channels=self.window_size * 2, out_channels=self.window_size, kernel_size=1),
            )
        # else:
        # self.conv1d = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1),
        #     nn.Conv1d(in_channels=16, out_channels=4, kernel_size=1),
        #     nn.Conv1d(in_channels=4, out_channels=8, kernel_size=1),
        #     nn.Conv1d(in_channels=8, out_channels=self.in_length, kernel_size=1),
        # )
        self.kesi = 0.3

    def forward(self, x):
        # target: tensor of shape (batch,T, dim_in)
        batch_prev, _t_prev, dim_in_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(batch_prev*_t_prev//self.window_size, self.window_size, dim_in_prev)  # create local windows
        batch, _t, dim_in = x.shape

        query = self.linear_q(x)  # b,t,h*
        keys = self.linear_k(x).permute(0, 2, 1)
        values = self.linear_v(x)  # b,t,h*
        att_weight = (query @ keys) * self._norm_fact  # b t t
        att_weight = torch.softmax(att_weight, dim=-1)  # b t t
        conv_att_weight = self.sigmoid(self.conv1d(att_weight))  # b t t
        zeros = torch.zeros_like(conv_att_weight)
        R_temporal = torch.where(conv_att_weight < self.kesi, zeros,
                                             conv_att_weight)  # b t t
        sparse_att_weight = att_weight * R_temporal
        sparse_values = sparse_att_weight @ values  # b t h*
        sparse_values = self.fc_o(sparse_values).reshape(batch_prev, _t_prev, dim_in_prev)
        return sparse_values, att_weight


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2=None, x3=None):
        if x3 is not None:
            x = torch.add(torch.add(x1, x2), x3)
        elif x2 is not None:
            x=torch.add(x1, x2)
        else:
            x = x1
        return self.normalize(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def sparseSoftmax(A, sparse_dist):
    x_exp = torch.exp(A)  # m * n
    eps = 0.0001
    partition = x_exp * sparse_dist
    sum_partition = torch.sum(partition, dim=-1, keepdim=True)
    return x_exp * sparse_dist / (sum_partition + eps)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()  # 30,embed_dim
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)  # 30,1
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [N,batch_size, seq_len, d_model]
        '''
        x = x + self.pe[:, :x.size(2), :].unsqueeze(0)
        return x


def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out

