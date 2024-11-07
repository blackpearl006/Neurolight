'''implementation from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py'''
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import *
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import f1_score, accuracy_score
import torch.distributed as dist
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader = None,
        val_data: DataLoader = None,
        test_data: DataLoader = None,
        x_data: DataLoader = None,
        early_stopping_patience: int = 10,
        save_every: int = 10,
        snapshot_path: str = '',
        best_snapshot_path: str = '',
        result_path: str = '',
        criterion = None, #: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler = None,
        logger = None,
        note: str = '',
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.x_data = x_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.save_every = save_every
        self.epochs_run = 0
        self.best_vloss = np.inf
        self.no_improvement = 0
        self.best_model = model 
        self.snapshot_path =snapshot_path
        self.best_snapshot_path=best_snapshot_path
        self.result_path =result_path
        self.early_stop_flag = torch.zeros(1).to(self.gpu_id)
        self.val_loss, self.train_loss = [], []
        if not early_stopping_patience: 
            self.early_stopping_patience = 1000
        else:
            self.early_stopping_patience = early_stopping_patience
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path, remove_module=True)
        self.model = DDP(model, device_ids=[self.gpu_id])
        if self.gpu_id == 0: self.logger.info(note)

    def _batch_normalise(self, inputs, labels):
        # labels = labels / 100 #for regression
        return inputs, labels

    def _run_train_batch(self, inputs, labels):
        inputs, labels = self._batch_normalise(inputs, labels)
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item() / inputs.size(0)
    
    def _run_val_batch(self, inputs, labels):
        inputs, labels = self._batch_normalise(inputs, labels)
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        return loss.item() / inputs.size(0)

    def _run_epoch(self, epoch):
        tloss, vloss, xloss = 0.0, 0.0, 0.0
        self.model.train()
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            targets = targets[0]
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            tloss += self._run_train_batch(source, targets)
        
        if self.scheduler:
            self.scheduler.step()
            # if self.gpu_id == 0: print(self.scheduler.get_last_lr()[0])

        vloss = self._run_test_epoch(self.val_data)

        sync_tloss = self._sync_tensor(tloss)
        sync_vloss = self._sync_tensor(vloss)

        self.train_loss.append(sync_tloss)
        self.val_loss.append(sync_vloss)
        return sync_tloss, sync_vloss, xloss
    
    def _sync_tensor(self, loss_values):
        tensor = torch.tensor(loss_values, device=self.gpu_id)
        gathered_tensor = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensor, tensor)
        mean_val = sum(gathered_tensor)/dist.get_world_size()
        return mean_val.item()

    def early_stopping(self, vloss, epoch=0):
        if self.gpu_id == 0:
            if vloss >= self.best_vloss:
                self.no_improvement += 1
            else:
                self.no_improvement = 0
                self.best_vloss = vloss
                self.logger.info(f"Value of self.no_improvement: {self.no_improvement}")
                self._save_snapshot(epoch=epoch, best=True)

            if self.no_improvement >= self.early_stopping_patience:
                print(f'Early stopping triggered at epoch {self.epochs_run}')
                self.logger.info(f'Early stopping triggered at epoch {epoch}')
                self.early_stop_flag += 1  # Set early stop flag
        torch.distributed.all_reduce(self.early_stop_flag, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.barrier()  # Ensure all processes are synchronized here

    def _run_test_epoch(self, test_data):
        self.model.eval()
        with torch.no_grad():
            loss = 0.0
            for source, targets in test_data:
                targets = targets[0]
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                loss += self._run_val_batch(source, targets)
        return loss
    
    def _plot_curve(self):
        plt.figure(figsize=(16,16))
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss value')
        plt.legend()
        plt.savefig(f'{self.result_path}/loss_curves.png',dpi=400)
        plt.close()

        # plt.figure(figsize=(16,16))
        # plt.plot(self.train_loss[20:], label='Training Loss')
        # plt.plot(self.val_loss[20:], label='Validation Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss value')
        # plt.legend()
        # plt.savefig(f'{self.result_path}/loss_curves_leaveout_first20.png',dpi=400)
        # plt.close()

    def _get_r2_mae(self, test_data, name='nothing'):
        if os.path.exists(self.best_snapshot_path):
            self.logger.info('Loaded from best snapshot path')
            self._load_snapshot(self.best_snapshot_path, remove_module=False)
        else:
            self._load_snapshot(self.snapshot_path)

        with torch.no_grad():
            test_loss, all_predicted, all_labels, all_datasets, all_ids = 0.0, [], [], [], []
            total, correct = 0, 0
            for inputs, labels in test_data:
                labels, dataset, ids = labels[0], labels[1], labels[2]
                inputs = inputs.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                inputs, labels = self._batch_normalise(inputs, labels)
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                test_loss += loss.item() / inputs.size(0)
                probability = torch.sigmoid(output)
                predicted = (probability >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect predictions, labels, datasets, and ids
                all_predicted.append(predicted.cpu().numpy().ravel())
                all_labels.append(labels.cpu().numpy().ravel())
                all_datasets.extend(dataset)  # Extend to capture dataset names
                all_ids.extend(ids)  # Extend to capture ids

        # Concatenate numpy arrays of all_predicted and all_labels
        all_predicted = torch.tensor(np.concatenate(all_predicted)).to(self.gpu_id)
        all_labels = torch.tensor(np.concatenate(all_labels)).to(self.gpu_id)

        gathered_predicted = [torch.empty_like(all_predicted) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.empty_like(all_labels) for _ in range(dist.get_world_size())]

        # Gather predictions and labels
        dist.all_gather(gathered_predicted, all_predicted)
        dist.all_gather(gathered_labels, all_labels)

        all_predicted = torch.cat(gathered_predicted).cpu().numpy().ravel()
        all_labels = torch.cat(gathered_labels).cpu().numpy().ravel()

        # Gather string-based dataset and ID lists across processes using all_gather_object
        gathered_datasets = [None for _ in range(dist.get_world_size())]
        gathered_ids = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(gathered_datasets, all_datasets)
        dist.all_gather_object(gathered_ids, all_ids)

        # Flatten the gathered lists
        all_datasets = [item for sublist in gathered_datasets for item in sublist]
        all_ids = [item for sublist in gathered_ids for item in sublist]

        test_f1_score = f1_score(all_labels, all_predicted)
        test_accuracy_score = accuracy_score(all_labels, all_predicted)

        # Create DataFrame with Labels, Predictions, Dataset, and ID
        df = pd.DataFrame({
            'Dataset': all_datasets,
            'ID': all_ids,
            'True Sex': all_labels,
            'Predicted Sex': all_predicted
        })

        csv_path = f'{self.result_path}/test_predictions_{name}.csv'
        os.makedirs(self.result_path, exist_ok=True)
        df.to_csv(csv_path, index=False)

        return test_loss, test_f1_score, test_accuracy_score


    def _save_snapshot(self, epoch, best=False):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSSES": self.train_loss,
            "VAL_LOSSES": self.val_loss,
            "BEST_VAL_LOSS": self.best_vloss
        }
        if self.gpu_id == 0: 
            if best:
                torch.save(snapshot, self.best_snapshot_path)
                print(f"Epoch {epoch} | Best Training snapshot saved at {self.best_snapshot_path}")
            else: 
                torch.save(snapshot, self.snapshot_path)
                print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self, snapshot_path, remove_module=False):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc, weights_only=True)
        if remove_module:
            new_state_dict = {}
            for key, value in snapshot["MODEL_STATE"].items():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_loss = snapshot["TRAIN_LOSSES"]
        self.val_loss = snapshot["VAL_LOSSES"]
        self.best_vloss = snapshot["BEST_VAL_LOSS"]
        if self.gpu_id == 0:
            print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
            self.logger.info(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            train_loss, val_loss, x_vloss = self._run_epoch(epoch)

            if self.gpu_id == 0:
                if epoch % self.save_every == 0:
                    self._plot_curve()
                    self._save_snapshot(epoch)
                print(f"Epoch [{epoch+1}/{max_epochs}]: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
                self.logger.info(f"Epoch [{epoch+1}/{max_epochs}]: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

            if epoch > 100:
                self.early_stopping(val_loss, epoch)
            # print(self.early_stop_flag)
            if self.early_stop_flag.item() != 0:
                print(f"Early stopping at epoch {epoch}")
                break

    def test(self):
        returnable_maes = []
        if self.gpu_id == 0: 
            print('Testing !!')
            self.logger.info('Testing !!')
        if self.x_data is not None:
            for data, name in zip([self.x_data, self.test_data], ['X_DATA', 'TEST_DATA']):
                loss, mae, test_r2_score = self._get_r2_mae(test_data=data, name=name)
                returnable_maes.append(mae)
                if self.gpu_id == 0:
                    print(f"ALL {name}: loss: {loss:.4f}, F1 score: {mae:.4f}, Accuracy: {test_r2_score:.4f}")
                    self.logger.info(f"ALL {name}: loss: {loss:.4f}, F1 score: {mae:.4f}, Accuracy: {test_r2_score:.4f}")
        else:
            loss, test_f1_score, test_accuracy_score = self._get_r2_mae(test_data=self.test_data, name='TEST_DATA')
            if self.gpu_id == 0:
                    print(f"Loss: {loss:.4f}, F1 score: {test_f1_score:.4f}, Accuracy: {test_accuracy_score:.4f}")
                    self.logger.info(f"Loss: {loss:.4f}, F1 score: {test_f1_score:.4f}, Accuracy: {test_accuracy_score:.4f}")
        # return returnable_maes