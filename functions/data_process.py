from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scanpy as sc
import numpy as np
import random
import scipy
import os


def setup_seed(seed, need_torch=True):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if need_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def tfidf2(count_mat: sp.csc_matrix):
    if not sp.isspmatrix_csc(count_mat):
        count_mat = sp.csc_matrix(count_mat)

    mat_data = count_mat.data.copy().astype(np.float32)
    mat_indices = count_mat.indices.copy()
    mat_indptr = count_mat.indptr.copy()
    # tf
    cell_sum = count_mat.sum(axis=1).A.squeeze()
    for i in range(count_mat.size):
        mat_data[i] = 1.0 * count_mat.data[i] / cell_sum[mat_indices[i]] if cell_sum[mat_indices[i]] else 0
    # idf
    df = count_mat.sum(axis=0).A.squeeze().astype(np.float32)
    for i in range(len(df)):
        if df[i] != 0:
            df[i] = 1.0 * count_mat.shape[0] / df[i]
    col_ptr = 0
    for i in range(count_mat.size):
        while i >= count_mat.indptr[col_ptr + 1]:
            col_ptr += 1
        mat_data[i] = np.log(1 + 1e4 * mat_data[i] * df[col_ptr])
    return sp.csc_matrix((mat_data, mat_indices, mat_indptr), shape=count_mat.shape)


def merge_genes_group_score(annData : sc.AnnData, groupby: str, n_genes_all: int):
    class_list = annData.obs[groupby].cat.categories
    class_num = len(class_list)
    value_counts_list = annData.obs[groupby].value_counts()
    num_per_cell = n_genes_all // class_num
    peak_num_list = [num_per_cell for _ in range(class_num)]
    for i in range(n_genes_all - num_per_cell * class_num):
        peak_num_list[class_num - 1 - i] += 1
    cell_idx_dict = {class_list[i]: i for i in range(class_num)}

    idx_mat = np.array(annData.uns['selected_peaks'].tolist()).astype(int).T

    duplicated_set = set()
    col_list_list = [set() for _ in range(class_num)]

    for i in range(class_num - 1, -1, -1):
        cell_idx = cell_idx_dict[value_counts_list.index[i]]
        pointer = 0
        while len(col_list_list[cell_idx]) < peak_num_list[i]:
            deduplicationed_set = set(idx_mat[cell_idx, pointer : pointer + peak_num_list[i] - len(col_list_list[cell_idx])].tolist()) - duplicated_set
            col_list_list[cell_idx].update(deduplicationed_set)
            pointer += peak_num_list[i] - len(col_list_list[cell_idx])
        duplicated_set.update(col_list_list[cell_idx])

    for i in range(class_num):
        col_list_list[i] = list(col_list_list[i])

    col_list = []
    for i in range(class_num):
        col_list.extend(col_list_list[i])

    return col_list, peak_num_list


def process_peak_mat_to_posi_idx(data: np.ndarray, peak_num_list):
    class_num = len(peak_num_list)
    peak_num = min(peak_num_list)
    output = np.zeros((data.shape[0], class_num, peak_num))
    for i in range(data.shape[0]):
        ptr = 0
        for j in range(class_num):
            output[i, j] = data[i, ptr: ptr + peak_num]
            ptr = ptr + peak_num + 1 if peak_num_list[j] > peak_num else ptr + peak_num
    return output


def select_groups(adata: sc.AnnData, key: str):
    """Get subset of groups in adata.obs[key]."""
    groups_masks = np.zeros(
        (len(adata.obs[key].cat.categories), adata.obs[key].values.size), dtype=bool
    )
    for iname, name in enumerate(adata.obs[key].cat.categories):
        if adata.obs[key].cat.categories[iname] in adata.obs[key].values:
            mask = adata.obs[key].cat.categories[iname] == adata.obs[key].values
        else:
            mask = str(iname) == adata.obs[key].values
        groups_masks[iname] = mask

    return groups_masks


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", save_path=None, dpi=100, show_all_value=False):
    sc.set_figure_params(dpi=300)

    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true', labels=[i for i in range(len(label_name))])

    plt.figure(figsize=(3, 2), facecolor='white')
    plt.imshow(cm, cmap='Blues')
    plt.title(title, fontsize=6)
    plt.xlabel("Predict label", fontsize=6)
    plt.ylabel("Truth label", fontsize=6)

    label_name_show = [label_name[i] + f" ({(i + 1):02d})" for i in range(len(label_name))]
    label_idx_show = [f" ({(i + 1):02d})" for i in range(len(label_name))]
    plt.yticks(range(label_name.__len__()), label_name_show, fontsize=5)
    plt.xticks(range(label_name.__len__()), label_idx_show, rotation=45, fontsize=5)
    plt.tight_layout()
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=5)

    fontsize = 30 / len(label_name)
    if show_all_value:
        for i in range(label_name.__len__()):
            for j in range(label_name.__len__()):
                color = (1, 1, 1) if i == j else (0, 0, 0)
                value = float(format('%.2f' % cm[j, i]))
                plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=fontsize)
    else:
        for i in range(label_name.__len__()):
            color = (1, 1, 1)
            value = float(format('%.2f' % cm[i, i]))
            plt.text(i, i, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=fontsize)

    plt.grid(False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


def plot_and_save_umap(trans_output, Predicted_labels, True_labels, label_title, label_palette, save_label_path, pred_title, pred_palette,):
    adata_embedding = sc.AnnData(X=trans_output)
    adata_embedding.obs['Predicted_labels'] = Predicted_labels
    adata_embedding.obs['True_labels'] = True_labels

    sc.pp.neighbors(adata_embedding, use_rep='X')
    sc.tl.umap(adata_embedding)
    sc.set_figure_params(figsize=(2, 2), dpi=300, fontsize=8)
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=300)

    sc.pl.umap(
        adata_embedding,
        color='Predicted_labels',
        palette=pred_palette,
        size=8,
        title=pred_title,
        show=False,
        ax=axes[0],
        legend_loc=None
    )
    sc.pl.umap(
        adata_embedding,
        color='True_labels',
        palette=label_palette,
        size=8,
        title=label_title,
        show=False,
        ax=axes[1],
        legend_loc='right margin'
    )

    handles, labels = axes[1].get_legend_handles_labels()
    legend = axes[1].legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=6,
        markerscale=0.5,
        frameon=False,
    )

    if save_label_path:
        fig.savefig(save_label_path, bbox_inches='tight', dpi=1200)
    plt.show()
    plt.close()
