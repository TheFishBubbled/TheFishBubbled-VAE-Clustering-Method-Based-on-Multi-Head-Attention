import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder
# Decoder
class Encoder_rna(nn.Module):
    def __init__(self, in_feature):
        super(Encoder_rna, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        return self.encoder(x)


class Encoder_atac(nn.Module):
    def __init__(self, in_feature):
        super(Encoder_atac, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feature, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_rna(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Decoder_rna, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_feature // 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, out_feature),
        )

    def forward(self, x):
        return self.decoder(x)


class Decoder_atac(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Decoder_atac, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_feature // 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_feature),
        )

    def forward(self, x):
        return self.decoder(x)


class Encoder_both(nn.Module):
    def __init__(self, in_feature=512):
        super(Encoder_both, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feature, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, x):
        return self.encoder(x)

class VAE_MHA_V2(nn.Module):
    def __init__(self, input_dim_rna=2000, input_dim_atac=49, latten_feature=256, num_head=4):
        super(VAE_MHA_V2, self).__init__()
        # num_head 表示多头注意力机制中头的数目

        self.liner_rna_k = nn.Linear(in_features=latten_feature, out_features=latten_feature)
        self.liner_rna_q = nn.Linear(in_features=latten_feature, out_features=latten_feature)
        self.liner_rna_v = nn.Linear(in_features=latten_feature, out_features=latten_feature)

        self.liner_atac_k = nn.Linear(in_features=latten_feature, out_features=latten_feature)
        self.liner_atac_q = nn.Linear(in_features=latten_feature, out_features=latten_feature)
        self.liner_atac_v = nn.Linear(in_features=latten_feature, out_features=latten_feature)
        # 编码器部分
        self.encoder_rna = Encoder_rna(in_feature=input_dim_rna)
        self.encoder_atac = Encoder_atac(in_feature=input_dim_atac)
        self.encoder_both = Encoder_both(in_feature=latten_feature * 2)

        # 解码器部分
        self.decoder_rna = Decoder_rna(in_feature=latten_feature, out_feature=input_dim_rna)
        self.decoder_atac = Decoder_atac(in_feature=latten_feature, out_feature=input_dim_atac)
        # 多头注意力部分在forward部分实现

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, rna, atac):
        # encode
        rna_hidden = self.encoder_rna(rna)
        atac_hidden = self.encoder_atac(atac)

        rna_k = self.liner_rna_k(rna_hidden)
        rna_v = self.liner_rna_v(rna_hidden)
        rna_q = self.liner_rna_q(atac_hidden)

        U_rna = torch.mm(rna_q.t(), rna_k)
        A_rna = torch.sum(U_rna, dim=1)
        # 计算最大值和最小值
        max_value_rna = torch.max(A_rna)
        min_value_rna = torch.min(A_rna)
        # 进行最大最小标准化
        normalized_A_rna = (A_rna - min_value_rna) / (max_value_rna - min_value_rna)
        rna_attention_result = normalized_A_rna * rna_v

        atac_k = self.liner_atac_k(atac_hidden)
        atac_v = self.liner_atac_v(atac_hidden)
        atac_q = self.liner_atac_k(rna_hidden)

        U_atac = torch.mm(atac_q.t(), atac_k)
        A_atac = torch.sum(U_atac, dim=1)
        # 计算最大值和最小值
        max_value_atac = torch.max(A_atac)
        min_value_atac = torch.min(A_atac)
        # 进行最大最小标准化
        normalized_A_atac = (A_atac - min_value_atac) / (max_value_atac - min_value_atac)
        atac_attention_result = normalized_A_atac * atac_v

        fusion_feature = torch.cat((rna_attention_result, atac_attention_result), dim=1)
        encoded_both = self.encoder_both(fusion_feature)

        mu, log_var = torch.chunk(encoded_both, 2, dim=-1)
        z = self.reparameterize(mu, log_var)

        # decode
        rna_reconstructed = self.decoder_rna(z)
        atac_reconstructed = self.decoder_atac(z)
        return rna, atac, rna_reconstructed, atac_reconstructed, mu, log_var, encoded_both

if __name__ == '__main__':
    rna = torch.rand((5, 2000))
    atac = torch.rand((5, 49))
    vae = VAE_MHA_V2()

