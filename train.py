'''
Author: Zhang-Linhui
Date: 2024-09-23 06:53:09
LastEditors: Zhang-Linhui
LastEditTime: 2024-09-23 20:11:50
'''
'''
Author: Zhang-Linhui
Date: 2024-09-23 06:53:09
LastEditors: Zhang-Linhui
LastEditTime: 2024-09-23 07:30:32
'''
from data_load import load_data


import torch

from tensorboardX import SummaryWriter
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


import json
import csv
import os
from datetime import datetime

data = load_data()
model_name = 'Transformer'

def initialize_model(model_name, data):
    from model.gcn import GCN
    from model.graphsage import GraphSAGE
    from model.gat import GAT
    from model.chebnet import ChebNet
    from model.sgc import SGC
    from model.gin import GIN
    from model.appnp import APPNPNet as APPNP
    from model.mlp import MLP
    from model.cnn import CNN
    from model.rnn import RNN
    from model.lstm import LSTM
    from model.transformer_model import TransformerModel

    if model_name == 'GCN':
        return GCN(in_channels=data.num_features, out_channels=2)  # 2分类任务
    elif model_name == 'GraphSAGE':
        return GraphSAGE(in_channels=data.num_features, out_channels=2)  # 2分类任务
    elif model_name == 'GAT':
        return GAT(in_channels=data.num_features, out_channels=2)  # 2分类任务
    elif model_name == 'ChebNet':
        return ChebNet(in_channels=data.num_features, out_channels=2)  # 2分类任务
    elif model_name == 'SGC':
        return SGC(in_channels=data.num_features, out_channels=2)  # 2分类任务
    elif model_name == 'GIN':
        return GIN(in_channels=data.num_features, out_channels=2)  # 2分类任务
    elif model_name == 'APPNP':
        return APPNP(in_channels=data.num_features, out_channels=2)  # 2分类任务
    elif model_name == 'MLP':
        return MLP(in_channels=data.num_features, out_channels=2)  # 2分类任务
    elif model_name == 'CNN':
        return CNN(in_channels=1, out_channels=2)  # 假设输入是单通道图像
    elif model_name == 'RNN':
        return RNN(in_channels=data.num_features, hidden_size=64, num_layers=2, out_channels=2)  # 2分类任务
    elif model_name == 'LSTM':
        return LSTM(in_channels=data.num_features, hidden_size=64, num_layers=2, out_channels=2)  # 2分类任务
    elif model_name == 'Transformer':
        return TransformerModel(in_channels=data.num_features, num_heads=1, num_layers=2, out_channels=2)  # 2分类任务
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train(model, optimizer, data, criterion, writer, epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    pred = out[data.train_mask].max(1)[1]
    correct = pred.eq(data.y[data.train_mask]).sum().item()
    acc = correct / data.train_mask.sum().item()

    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(data.y[data.train_mask].cpu(), pred.cpu(), average='binary')
    auc = roc_auc_score(data.y[data.train_mask].cpu(), out[data.train_mask][:, 1].detach().cpu())

    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)
    writer.add_scalar('Precision/train', precision, epoch)
    writer.add_scalar('Recall/train', recall, epoch)
    writer.add_scalar('F1/train', f1, epoch)
    writer.add_scalar('AUC/train', auc, epoch)

    return loss.item(), acc, precision, recall, f1, auc

def validate(model, data, criterion, writer, epoch):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[data.test_mask], data.y[data.test_mask])

        pred = out[data.test_mask].max(1)[1]
        correct = pred.eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()

        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(data.y[data.test_mask].cpu(), pred.cpu(), average='binary')
        auc = roc_auc_score(data.y[data.test_mask].cpu(), out[data.test_mask][:, 1].detach().cpu())

        writer.add_scalar('Loss/val', loss.item(), epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1/val', f1, epoch)
        writer.add_scalar('AUC/val', auc, epoch)

    return loss.item(), acc, precision, recall, f1, auc


def save_results(model_name, train_metrics, val_metrics, final_train_metrics, final_val_metrics):
    # Create a results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    detailed_results = {
        'model_name': model_name,
        'train_metrics': [
            {'epoch': i+1, 'loss': m[0], 'accuracy': m[1], 'precision': m[2], 
             'recall': m[3], 'f1': m[4], 'auc': m[5]}
            for i, m in enumerate(train_metrics)
        ],
        'val_metrics': [
            {'epoch': i+1, 'loss': m[0], 'accuracy': m[1], 'precision': m[2], 
             'recall': m[3], 'f1': m[4], 'auc': m[5]}
            for i, m in enumerate(val_metrics)
        ],
        'final_train_metrics': {
            'loss': final_train_metrics[0], 'accuracy': final_train_metrics[1],
            'precision': final_train_metrics[2], 'recall': final_train_metrics[3],
            'f1': final_train_metrics[4], 'auc': final_train_metrics[5]
        },
        'final_val_metrics': {
            'loss': final_val_metrics[0], 'accuracy': final_val_metrics[1],
            'precision': final_val_metrics[2], 'recall': final_val_metrics[3],
            'f1': final_val_metrics[4], 'auc': final_val_metrics[5]
        }
    }

    json_filename = os.path.join(results_dir, f'{model_name}_results_{timestamp}.json')
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Save summary results as CSV
    csv_filename = os.path.join(results_dir, f'{model_name}_summary_{timestamp}.csv')
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Metric', 'Train', 'Validation'])
        metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        for i, metric in enumerate(metrics):
            writer.writerow([model_name, metric, final_train_metrics[i], final_val_metrics[i]])

    print(f"Detailed results saved to: {json_filename}")
    print(f"Summary results saved to: {csv_filename}")



# 初始化数据、模型和优化器

model = initialize_model(model_name, data)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 创建 TensorBoardX writer
writer = SummaryWriter(f'runs/{model_name}_experiment')

# 训练
epochs = 100
train_metrics = []
val_metrics = []

for epoch in range(epochs):
    train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train(model, optimizer, data, criterion, writer, epoch)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(model, data, criterion, writer, epoch)

    train_metrics.append((train_loss, train_acc, train_prec, train_rec, train_f1, train_auc))
    val_metrics.append((val_loss, val_acc, val_prec, val_rec, val_f1, val_auc))
    
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}')
    print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')
    print('-' * 80)


metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
fig, axes = plt.subplots(3, 2, figsize=(15, 20))
for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    ax.plot(range(1, epochs+1), [m[i] for m in train_metrics], label='Train')
    ax.plot(range(1, epochs+1), [m[i] for m in val_metrics], label='Validation')
    ax.set_title(f'{metric} vs. Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric)
    ax.legend()

plt.tight_layout()
plt.savefig(f'runs/{model_name}_metrics.png')
writer.add_figure('Training Metrics', fig)
# Print final results
print("\nFinal Results:")
print(f"Train - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
print(f"Val   - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

# Save results
final_train_metrics = train_metrics[-1]
final_val_metrics = val_metrics[-1]
save_results(model_name, train_metrics, val_metrics, final_train_metrics, final_val_metrics)

# Close writer
writer.close()
