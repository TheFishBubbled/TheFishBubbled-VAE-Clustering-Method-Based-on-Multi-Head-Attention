import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from VAE import VAE_MHA_V2


def kl_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
def reconstruction_loss(reconstructed_x, x):
    return F.mse_loss(reconstructed_x, x, reduction='sum')

def train(num_epochs, vae, dataloader, vae_optimizer):
    loss_kl_epoch = []
    loss_re_epoch = []
    total_loss_epoch = []
    loss_min = float('inf')
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        loss_kl = []
        loss_re = []
        total_loss = []
        for batch_idx, batch_data in progress_bar:
            vae_optimizer.zero_grad()
            rna, atac, rna_reconstructed, atac_reconstructed, mu, log_var, _ = vae(batch_data[0], batch_data[1])
            vae_reconstruction_loss = (reconstruction_loss(rna_reconstructed, rna) +
                                       reconstruction_loss(atac_reconstructed, atac))/2
            loss_re.append(vae_reconstruction_loss.data.item())
            total_kl_divergence = kl_divergence(mu, log_var)
            loss_kl.append(total_kl_divergence.data.item())
            # 计算总损失
            vae_gan_loss = vae_reconstruction_loss + total_kl_divergence
            # 反向传播和优化
            total_loss.append(vae_gan_loss.data.item())
            vae_gan_loss.backward()
            vae_optimizer.step()
            progress_bar.set_postfix({'Total Loss': vae_gan_loss.item()}, refresh=True)
        loss_kl_epoch.append(sum(loss_kl) / len(loss_kl))
        loss_re_epoch.append(sum(loss_re) / len(loss_re))
        total_loss_epoch.append(sum(total_loss) / len(total_loss))
        if total_loss_epoch[-1] < loss_min:
            loss_min = total_loss_epoch[-1]
            print('best model in epoch ', epoch, 'loss min is :', loss_min)
            best_net = vae.state_dict()
            torch.save(best_net, './TrainedModel/vaev2_4.pth')
    return loss_kl_epoch, loss_re_epoch, total_loss_epoch

def main():
    rna_data = pd.read_csv("data_for_final/scRNA_seq_for_final.tsv", sep="\t",
                           index_col=0).transpose()
    atac_data = pd.read_csv("data_for_final/scATAC_seq_for_final.tsv", sep="\t",
                            index_col=0).transpose()

    # 定义数据加载器
    rna_tensor = torch.tensor(rna_data.values, dtype=torch.float32)
    atac_tensor = torch.tensor(atac_data.values, dtype=torch.float32)

    combined_dataset = TensorDataset(rna_tensor, atac_tensor)
    # 定义一个DataLoader
    combined_dataloader = DataLoader(combined_dataset, batch_size=128, shuffle=True)

    # 定义输入维度和潜在表示维度
    input_dim_rna = rna_data.shape[1]  # 输入RNA数据的维度
    input_dim_atac = atac_data.shape[1]
    latent_dim = 256  # 潜在表示的维度

    vae_gan = VAE_MHA_V2(input_dim_rna=input_dim_rna, input_dim_atac=input_dim_atac, latten_feature=latent_dim)
    vae_gan.train()

    # 设置优化器
    vae_optimizer = optim.Adam(vae_gan.parameters(), lr=0.001)

    # 定义训练循环
    num_epochs = 100
    loss_kl, loss_re, total_loss = train(num_epochs, vae_gan,
                                           combined_dataloader,
                                           vae_optimizer)
    data = {
        'loss_kl': loss_kl,
        'loss_re': loss_re,
        'total_loss': total_loss
    }

    df = pd.DataFrame(data)

    # 将数据框保存到.xlsx文件
    df.to_excel('output.xlsx', index=False)

if __name__ == '__main__':
    main()
