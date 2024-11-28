from data.dataset import *
from data.dataloder import *
from models.sfcn import SFCN

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, random_split

VAL_RATIO = 0.2
TEST_RATIO = 0.2
TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO
DATA_PARALELL = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = 'HCP'
PHASE = 'TRAIN'
K_FOLDS = 5

TASK = 'classification'
ROOT_DIR = '/data/ninad/fieldmaps/AD'
LABEL_DIR='/data/ninad/fieldmaps/HCPCN2mm.csv'
LOG_DIR = '/home/neelamlab/ninad/DWI/logs'
timestamp = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
RESULT_DIR = f'/home/neelamlab/ninad/DWI/results/run_{timestamp}'
os.makedirs(RESULT_DIR,exist_ok=True)
TEST_BATCH_SIZE = 2

BATCH_SIZE=16
LEARNING_RATE=0.0001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5

NP_SEED = 42
TORCH_SEED = 36

np.random.seed(NP_SEED)
torch.manual_seed(TORCH_SEED)

def setup_logger(logs_dir=LOG_DIR,dataset=None,phase=None):
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger('RunLogger')
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        timestamp = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        file_handler = logging.FileHandler(os.path.join(logs_dir, f'{phase}_{dataset}_{timestamp}.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

LOGGER = setup_logger(logs_dir=LOG_DIR,phase=PHASE,dataset=DATASET)

dataset = Fieldmapdata(root_dir=ROOT_DIR,label_dir=LABEL_DIR,task=TASK)
train_dataset, test_ds = random_split(dataset, [1- TEST_RATIO, TEST_RATIO])
test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False)

LOGGER.info('Data split completed !!')

model = SFCN() #generate_model_resnet(50,n_classes=1, n_input_channels=1)
LOGGER.info(model)

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=NP_SEED)
fold_train_losses, fold_val_losses = [], []
fold_train_accuracies, fold_val_accuracies = [], []
# LOGGER.info(f"Weighted BCE loss : Total subjects: 899 Number of males : 395, number of females: 504, pos_weight = 1.2759493670886075")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}/{K_FOLDS}')
    LOGGER.info(f'Fold {fold+1}/{K_FOLDS}')
    
    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(train_dataset, val_idx)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SFCN() #generate_model_resnet(50, n_classes=1, n_input_channels=1).to(DEVICE)
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_model = SFCN() #generate_model_resnet(50,n_classes=1, n_input_channels=1)
    
    # criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.28,dtype=torch.float32)) # Total subjects: 899 Number of males : 395, number of females: 504, pos_weight = 1.2759493670886075
    criterion2 = nn.BCEWithLogitsLoss()

    best_val_loss = np.inf
    epochs_without_improvement = 0

    # Train the model for each fold
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0

        for inputs, labels in train_dl:
            inputs = inputs.to(DEVICE)
            # mean = inputs.mean()
            # std = inputs.std()
            # inputs = (inputs - mean) / (std + 1e-8)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            tloss = criterion2(outputs, labels)
            tloss.backward()
            optimizer.step()
            train_loss += tloss.item() / inputs.size(0)
            probability = torch.sigmoid(outputs)
            predicted = (probability >= 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        fold_train_accuracies.append(train_accuracy)
        fold_train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs = inputs.to(DEVICE)
                # mean = inputs.mean()
                # std = inputs.std()
                # inputs = (inputs - mean) / (std + 1e-8)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                vloss = criterion2(outputs, labels)
                val_loss += vloss.item() / inputs.size(0)
                probability = torch.sigmoid(outputs)
                predicted = (probability >= 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            val_accuracy = 100 * val_correct / val_total
            fold_val_accuracies.append(val_accuracy)
            fold_val_losses.append(val_loss)

            print(f"Fold [{fold+1}/{K_FOLDS}], Epoch [{epoch+1}/{NUM_EPOCHS}]: Train Loss: {train_loss:.4f} Train Accu: {train_accuracy:.4f} Val Loss: {val_loss:.4f} Val Accu: {val_accuracy:.4f}")
            LOGGER.info(f"Fold [{fold+1}/{K_FOLDS}], Epoch [{epoch+1}/{NUM_EPOCHS}]: Train Loss: {train_loss:.4f} Train Accu: {train_accuracy:.4f} Val Loss: {val_loss:.4f} Val Accu: {val_accuracy:.4f}")
        
        # Early stopping
        if epoch > 30:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0 

                new_state_dict = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace("module.", "")
                    new_state_dict[new_key] = value
                best_model.load_state_dict(new_state_dict)

                torch.save(model.state_dict(), f'{RESULT_DIR}/resnet50_finetuned_fold{fold+1}_epoch{epoch}.pt')
                LOGGER.info(f"Validation loss improved for fold {fold+1} at epoch {epoch+1}")
            else:
                epochs_without_improvement += 1
                LOGGER.info(f"No improvement in validation loss for {epochs_without_improvement} epochs for fold {fold+1}")

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered for fold {fold+1} at epoch {epoch+1}")
                LOGGER.info(f"Early stopping triggered for fold {fold+1} at epoch {epoch+1}")
                break

avg_train_loss = np.mean(fold_train_losses)
avg_val_loss = np.mean(fold_val_losses)
avg_train_accuracy = np.mean(fold_train_accuracies)
avg_val_accuracy = np.mean(fold_val_accuracies)

print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accu: {avg_train_accuracy:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Avg Val Accu: {avg_val_accuracy:.4f}")
LOGGER.info(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accu: {avg_train_accuracy:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Avg Val Accu: {avg_val_accuracy:.4f}")

# torch.save(model.state_dict(),'resnet50_finetuned_7sept.pt')
best_model.to(DEVICE)
best_model.eval()
test_loss = 0.0
test_total = 0
test_correct = 0

with torch.no_grad():
    all_predicted = []
    all_labels = []
    for inputs, labels in test_dl:
        inputs = inputs.to(DEVICE)
        # mean = inputs.mean()
        # std = inputs.std()
        # inputs = (inputs - mean) / (std + 1e-8)
        labels = labels.to(DEVICE).unsqueeze(1)

        outputs = best_model(inputs)
        te_loss = criterion2(outputs, labels)
        test_loss += te_loss.item()/inputs.size(0)
        probability = torch.sigmoid(outputs)
        predicted = (probability >= 0.5).float()
        all_predicted.extend(predicted.cpu().numpy().ravel().tolist())
        all_labels.extend(labels.cpu().numpy().ravel().tolist())
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total

print(f"TEST LOSS: {test_loss}, TEST ACCURACY: {test_accuracy}")
LOGGER.info(f"TEST LOSS: {test_loss}, TEST ACCURACY: {test_accuracy}")
accuracy = accuracy_score(all_labels, all_predicted)
sensitivity = recall_score(all_labels, all_predicted)
print(f'Accuracy: {accuracy}, Sensitivity (Recall): {sensitivity}')
LOGGER.info(f'Accuracy: {accuracy}, Sensitivity (Recall): {sensitivity}')

cm = confusion_matrix(all_labels, all_predicted)
print('CM: ',cm)
LOGGER.info(f"CM : {cm}")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')