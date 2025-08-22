from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import scipy.sparse as sp
import scanpy as sc
import numpy as np
import random
import torch
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


def hitanno_preprocess_data(adata_train: sc.AnnData, adata_test: sc.AnnData, batch_size: int=4, peak_num_ct=None, seed=42):
    adata_train.var_names = [str(i) for i in range(adata_train.X.shape[1])]
    adata_test.var_names = [str(i) for i in range(adata_test.X.shape[1])]

    cell_type_num = len(adata_train.obs['label'].cat.categories.to_list())
    PEAK_NUM_CT = int(np.sqrt(9 / cell_type_num) * 2000) # for an 24G GPU
    peak_num_ct = peak_num_ct if peak_num_ct else PEAK_NUM_CT
    peak_num = peak_num_ct * len(adata_train.obs['label'].cat.categories.to_list())
    print(f"{peak_num_ct} peaks per cell type, {peak_num} peaks in total")
    
    idx_all = adata_train.obs.index
    labels = adata_train.obs["label"]
    idx_train, idx_eval = train_test_split(
        idx_all,
        test_size=0.25,
        stratify=labels,
        random_state=seed
    )

    atac_train = adata_train[idx_train]
    atac_test = adata_test
    atac_eval = adata_train[idx_eval]

    atac_train.layers["tfidf"] = tfidf2(atac_train.X)
    sc.tl.rank_genes_groups(atac_train, 'label', n_genes=6300, layer="tfidf", use_raw=False, method='t-test')
    atac_train.uns['selected_peaks'] = atac_train.uns['rank_genes_groups']['names']
    col_list, peak_num_list = merge_genes_group_score(atac_train, 'label', peak_num)

    sub_ATAC_train = atac_train[:, col_list]
    sub_ATAC_test = atac_test[:, col_list]
    sub_ATAC_eval = atac_eval[:, col_list]

    x_train = np.where(sub_ATAC_train.X.toarray(), 1, 0)
    y_train = atac_train.obs['label']
    x_test = np.where(sub_ATAC_test.X.toarray(), 1, 0)
    y_test = atac_test.obs['label']
    x_eval = np.where(sub_ATAC_eval.X.toarray(), 1, 0)
    y_eval = atac_eval.obs['label']

    x_train = process_peak_mat_to_posi_idx(x_train, peak_num_list)
    x_test = process_peak_mat_to_posi_idx(x_test, peak_num_list)
    x_eval = process_peak_mat_to_posi_idx(x_eval, peak_num_list)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.cat.codes, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.cat.codes, dtype=torch.long))
    eval_dataset = TensorDataset(torch.tensor(x_eval), torch.tensor(y_eval.cat.codes, dtype=torch.long))
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, eval_loader, peak_num, col_list, peak_num_list