from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
from .model import *

import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import time
import copy
import json


def hitanno_train(train_loader, eval_loader, hitanno_model: HitAnno, col_list, device, save_model_path=None):
    NUM_EPOCHS = 20
    PATIENCE = 3
    best_eval_loss = np.Inf
    patience_count = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(hitanno_model.parameters()), lr=0.0001)
    scaler = GradScaler()
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=steps_per_epoch * 3,
        num_training_steps=num_training_steps,
    )

    time_start = time.time()
    epoch_losses = []
    epoch_eval_acc_list = []
    best_eval_loss = np.Inf
    patience_count = 0
    best_model_params = None
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        hitanno_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            with autocast():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, outputs_trans = hitanno_model(inputs)
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        hitanno_model.eval()
        eval_epoch_loss = 0.0
        with torch.no_grad():
            correct_eval = 0
            total_eval = 0
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, outputs_trans = hitanno_model(inputs)
                loss = criterion(outputs, labels)
                eval_epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_eval += labels.size(0)
                correct_eval += (predicted == labels).sum().item()
        eval_epoch_loss /= len(eval_loader)

        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)
        epoch_eval_acc = correct_eval / total_eval
        epoch_eval_acc_list.append(epoch_eval_acc)
        print(f"Epoch {str(epoch + 1).zfill(2)}/{NUM_EPOCHS}, loss: {(epoch_loss):.3f}; loss on eval set: {(eval_epoch_loss):.3f}, accuracy on eval set: {(epoch_eval_acc):.3f};")

        if eval_epoch_loss < best_eval_loss:
            best_eval_loss = eval_epoch_loss
            patience_count = 0
            best_model_params = copy.deepcopy(hitanno_model.state_dict())
        else:
            patience_count += 1
        if (patience_count >= PATIENCE) and (epoch >= 4):
            print(f"early stopping at epoch {str(epoch + 1).zfill(2)}, best eval loss: {(best_eval_loss):.3f} at epoch {str(epoch - 2).zfill(2)}")
            break

    hitanno_model.load_state_dict(best_model_params)
    if save_model_path:
        hitanno_model.save_parameters(col_list, save_model_path)
    
    time_end = time.time()
    time_sum = time_end - time_start
    print(f"run time: {int(time_sum // 3600)}h {int((time_sum - time_sum // 3600 * 3600) // 60)}m {round(time_sum - time_sum // 60 * 60, 1)}s")

    return


def hitanno_predict(test_loader,
                    hitanno_model: HitAnno,
                    device,
                    class_list=None,
                    save_label_path=None):
    hitanno_model.eval()
    with torch.no_grad():
        y_label = None
        y_pred = None
        trans_output_list = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, outputs_trans = hitanno_model(inputs)
            trans_output_list.extend(outputs_trans.detach().cpu().tolist())
            _, predicted = torch.max(outputs.data, 1)
            if y_label == None:
                y_label = labels
                y_pred = predicted
            else:
                y_label = torch.cat((y_label, labels), 0)
                y_pred = torch.cat((y_pred, predicted), 0)
        trans_output = np.array(trans_output_list)

    if save_label_path:
        save_predict_label = pd.DataFrame({
            'True_labels': pd.Series([class_list[i] for i in y_label]).reset_index(drop=True),
            'Predicted_labels': pd.Series([class_list[i] for i in y_pred]).reset_index(drop=True),
        })
        save_predict_label.to_csv(save_label_path, index=False)

    return y_label.cpu(), y_pred.cpu(), trans_output


def calculate_metrices(y_label, y_pred, index, save_metirces_path):
    confusion_mat = pd.DataFrame(confusion_matrix(y_label, y_pred, labels=[i for i in range(len(index))]), index=index, columns=index)
    print(confusion_mat)

    acc = accuracy_score(y_label, y_pred)
    macro_f1_score = f1_score(y_label, y_pred, average='macro')
    kappa_score = cohen_kappa_score(y_label, y_pred)

    print("\n")
    print('accuracy: {}'.format(round(acc, 3)))
    print('macro-f1 score: {}'.format(round(macro_f1_score, 3)))
    print('kappa score: {}'.format(round(kappa_score, 3)))

    conf_mat_dict = confusion_mat.astype(int).to_dict()
    metrics_dict = {
        "accuracy": acc,
        "macro_f1": macro_f1_score,
        "kappa": kappa_score,
        "confusion_matrix": conf_mat_dict
    }
    with open(save_metirces_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)