import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import VAE
import umap
from sklearn.cluster import KMeans

# load model and pretrained
model = VAE.VAE_MHA_V2(input_dim_rna=2000, input_dim_atac=49, latten_feature=256)
model.load_state_dict(torch.load('TrainedModel/vaev2_4.pth'))
model.eval()

# load data
rna_data = pd.read_csv("data_for_final/scRNA_seq_for_final.tsv", sep="\t",
                       index_col=0).transpose()
atac_data = pd.read_csv("data_for_final/scATAC_seq_for_final.tsv", sep="\t",
                        index_col=0).transpose()
label_data = pd.read_csv("data_for_final/label_for_final.tsv", sep="\t", header=None,
                         names=["CellID", "Label"])
print('The scRNA data shape is : ', rna_data.shape)
print('The ATAC data shape is: ', atac_data.shape)
print('The label data shape is : ', label_data.shape)

rna_tensor = torch.tensor(rna_data.values, dtype=torch.float32)
atac_tensor = torch.tensor(atac_data.values, dtype=torch.float32)
with torch.no_grad():
    encode_data = model(rna_tensor, atac_tensor)[6].data.numpy()

# Using UMAP to Visualization
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(encode_data)
umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])
umap_df['Label'] = label_data['Label'].values



# Visualization
sns.set(style='white', palette='bright')
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Label', data=umap_df,
                palette="Set2", )
plt.title('UMAP')
plt.legend(loc='upper right', fontsize='small')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

n_clusters = len(label_data['Label'].unique())  # 根据标签确定聚类簇的数量,原标签行第二列标记为label
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_pred_1 = kmeans.fit_predict(umap_result)

ari_vae = adjusted_rand_score(label_data['Label'], labels_pred_1)
print('ari为:',ari_vae)

