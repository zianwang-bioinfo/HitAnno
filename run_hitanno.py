from hitanno.data_processing import *
from hitanno.visualization import *
from hitanno.trainer import *
from hitanno.model import *

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import warnings
import argparse
import torch
import os


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

SEED = 42
setup_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default = "./data/Domcke2020_adrenal_mini_train.h5ad",
                    help="Path to training dataset (h5ad file)")
parser.add_argument('--test_data', type=str, default = "./data/Domcke2020_adrenal_mini_test.h5ad",
                    help="Path to test dataset (h5ad file)")
parser.add_argument('--peak_num_ct', type=int, default = 0,
                    help="Peak number per cell type (the default parameters are set for a 24 GB GPU and can be adjusted dynamically according to the available memory)")
parser.add_argument('--output_path', type=str, default = "./outputs/",
                    help="Path of outputs")
args = parser.parse_args()

OUTPUT_DIREC = args.output_path
os.makedirs(OUTPUT_DIREC + "predictions/", exist_ok=True)
os.makedirs(OUTPUT_DIREC + "model/", exist_ok=True)
os.makedirs(OUTPUT_DIREC + "metrics/", exist_ok=True)
os.makedirs(OUTPUT_DIREC + "figures/", exist_ok=True)

print("#" * 80)
print("loading train dataset ...")
adata_train = sc.read_h5ad(args.train_data)
print(adata_train)

print("loading test dataset ...")
adata_test = sc.read_h5ad(args.test_data)
print(adata_test)

peak_num_ct = args.peak_num_ct
BATCH_SIZE = 4
class_list = adata_train.obs['label'].cat.categories.to_list()

print("#" * 80)
print("processing data ...")
train_loader, test_loader, eval_loader, \
    peak_num, col_list, peak_num_list = hitanno_preprocess_data(
        adata_train,
        adata_test,
        batch_size=BATCH_SIZE,
        peak_num_ct=peak_num_ct,
        seed=SEED,
    )

HitAnno_model = HitAnno(
    batch_size=BATCH_SIZE,
    peak_num_list=peak_num_list,
    peak_num_all=peak_num,
    output_size=len(class_list),
    device=device,
).to(device)

print("#" * 80)
print("training model ...")
hitanno_train(train_loader, 
              eval_loader, 
              HitAnno_model, 
              col_list, 
              device, 
              save_model_path=OUTPUT_DIREC + "model/" + f"hitanno_model.pth")

print("#" * 80)
print("evaluating model ...")
y_label, y_pred, \
    trans_output = hitanno_predict(
        test_loader, 
        HitAnno_model, 
        device, 
        class_list, 
        save_label_path=OUTPUT_DIREC + "predictions/" + f'hitAnno_predict_labels.csv')

pd.set_option('display.width', 300)
calculate_metrices(y_label, 
                   y_pred, 
                   index=pd.CategoricalIndex(class_list),
                   save_metirces_path=OUTPUT_DIREC + "metrics/" + "performance.json")

draw_confusion_matrix(label_true=y_label,
                      label_pred=y_pred,
                      label_name=pd.CategoricalIndex(class_list),
                      title=f"Confusion Matrix",
                      save_path=OUTPUT_DIREC + "figures/" + "confusion_matrix.png",
                      dpi=1200)

cmap_cell = {class_list[i]: plt.cm.tab20(i) for i in range(len(class_list))}
plot_and_save_umap(
    trans_output=trans_output,
    Predicted_labels=[class_list[i] for i in y_pred],
    True_labels=[class_list[i] for i in y_label],
    label_title="True Labels",
    pred_title="Predicted Labels",
    label_palette=cmap_cell,
    pred_palette=cmap_cell,
    save_label_path=OUTPUT_DIREC + "figures/" + "umap.png",
)