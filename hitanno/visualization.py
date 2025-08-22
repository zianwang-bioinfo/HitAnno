from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import scanpy as sc


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