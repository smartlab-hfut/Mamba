import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from Data_provider.data_factory import DataProvider
from utils.metrics import metric
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast:
    def __init__(self, args):
        """
        初始化实验实例。
        """
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        构建模型实例。
        """
        from Model.Mamba import Model

        # 创建模型实例
        model = Model(self.args).float()
        return model.to(self.device)

    def _get_data(self, flag):
        """
        获取数据集和数据加载器。
        """
        data_loader = DataProvider(self.args).get_data_loader(flag)
        return data_loader.dataset, data_loader

    def _select_optimizer(self):
        """
        定义优化器。
        """
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """
        定义多任务损失函数。
        """
        regression_criterion = nn.MSELoss()  # 回归损失
        classification_criterion = nn.CrossEntropyLoss()  # 分类损失

        def combined_loss(label_out, batch_label):
            """
            分类损失
            """
            classification_loss = classification_criterion(label_out, batch_label)  # 分类任务损失
            return  classification_loss

        return combined_loss

    def _move_to_device(self, batch_x, batch_label):
        """
        将输入和标签移动到指定设备。
        """
        return (
            batch_x.to(self.device, non_blocking=True),
            batch_label.to(self.device, non_blocking=True),
        )

    def vali(self, vali_loader, criterion):
        """
        验证集评估：计算分类损失和准确率
        """
        classification_losses = []
        vali_correct = 0
        vali_total = 0

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_label in vali_loader:
                batch_x, batch_label = self._move_to_device(batch_x, batch_label)

                label_out = self.model(batch_x, None)
                loss = criterion(label_out, batch_label)

                classification_losses.append(loss.item())

                preds = label_out.argmax(dim=1)
                vali_correct += (preds == batch_label).sum().item()
                vali_total += batch_label.size(0)

        # 平均损失 & 准确率
        vali_loss = np.mean(classification_losses)
        vali_acc = vali_correct / vali_total if vali_total > 0 else 0.0

        print(f"Cls Loss: {vali_loss:.4f},  Acc: {vali_acc:.4f}")

        return vali_loss, vali_acc  # ← 现在返回两个已定义的量

    def train(self):
        """
        模型训练过程: 同时打印回归/分类损失、以及训练/验证准确度，加入预测指标。
        """
        train_loader = self._get_data(flag='train')[1]
        vali_loader = self._get_data(flag='val')[1]

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = optim.lr_scheduler.StepLR(model_optim, step_size=100, gamma=0.9)  # 学习率调度器

        y_pred = None  # Initialize y_pred before training

        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_correct = 0
            train_total = 0


            self.model.train()
            epoch_start_time = time.time()

            for batch_x,  batch_label in train_loader:
                batch_x,  batch_label = self._move_to_device(batch_x,  batch_label)

                model_optim.zero_grad()
                label_out = self.model(batch_x,  y_pred)  # Pass y_pred to model

                classification_loss= criterion(label_out, batch_label)

                classification_loss.backward()
                model_optim.step()

                train_loss.append(classification_loss.item())

                # 统计分类准确度 (训练集)
                pred_class = torch.argmax(label_out, dim=1)
                train_correct += (pred_class == batch_label).sum().item()
                train_total += batch_label.size(0)

                # 收集训练集的预测和标签

            scheduler.step()  # 更新学习率

            # 计算本 epoch 的平均训练损失和准确度
            avg_train_loss = np.mean(train_loss)
            train_acc = train_correct / train_total if train_total > 0 else 0.0


            # 验证集评估 (同时得到验证集loss和acc)
            vali_loss, vali_acc = self.vali(vali_loader, criterion)

            print(f"Epoch {epoch + 1}/{self.args.train_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {vali_loss:.4f}, Val Acc: {vali_acc:.4f}, "
                  f"Time: {time.time() - epoch_start_time:.2f}s")

        return self.model

    def test(self):
        """
        测试模型，并保存预测结果与分类报告。
        """
        test_loader = self._get_data(flag='test')[1]
        self.model.eval()

        preds, trues = [], []
        all_labels, all_preds = [], []

        folder_path = './test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        y_pred = None  # Initialize y_pred for testing

        with torch.no_grad():
            for i, (batch_x, batch_label) in enumerate(test_loader):
                batch_x, batch_label = self._move_to_device(batch_x, batch_label)
                label_out = self.model(batch_x, y_pred)  # Use y_pred for next batch

                # 保存分类预测结果
                predicted = torch.argmax(label_out, dim=1)
                all_labels.extend(batch_label.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())


        # 打印分类报告
        report = classification_report(all_labels, all_preds, digits=4)
        print("Classification Report:\n", report)

        # 打印混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:\n", conf_matrix)

        # 保存分类报告
        with open(os.path.join(folder_path, 'classification_report.txt'), 'w') as f:
            f.write("Classification Report:\n" + report + "\n")
            f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n")


        # 保存预测值与真实值
        np.save(os.path.join(folder_path, 'preds.npy'), preds)
        np.save(os.path.join(folder_path, 'trues.npy'), trues)

        return preds, trues

    # def _visualize_results(self, pred, true, folder_path, index):
    #     """
    #     可视化预测结果与真实值，每个维度单独绘制为子图。
    #     """
    #     if torch.is_tensor(pred):
    #         pred = pred.cpu().numpy()
    #     if torch.is_tensor(true):
    #         true = true.cpu().numpy()
    #
    #     num_features = pred.shape[-1]  # 数据的特征维度
    #     fig, axes = plt.subplots(num_features, 1, figsize=(10, 6 * num_features))
    #
    #     for i in range(num_features):
    #         ax = axes[i] if num_features > 1 else axes
    #         ax.plot(true[:, i], label='Ground Truth', color='blue', linestyle='-')
    #         ax.plot(pred[:, i], label='Prediction', color='red', linestyle='--')
    #         ax.set_title(f'Dimension {i}')
    #         ax.set_xlabel('Time Steps')
    #         ax.set_ylabel('Values')
    #         ax.legend()
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(folder_path, f'result_{index}.png'))
    #     plt.close()

