from models.wiflexformer import WiFlexFormer
import datasets as data
import torch
import torch.nn as nn
import numpy as np
import sys
import time
import os.path
import wandb
import argparse
from tqdm import tqdm

# supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Precision metric
def precisionMetric(confusion_matrix):
    precision = confusion_matrix.diag() / confusion_matrix.sum(0)
    precision = precision[~torch.isnan(precision)].mean()
    return precision.item()

# Recall metric
def recallMetric(confusion_matrix):
    recall = confusion_matrix.diag() / confusion_matrix.sum(1)
    recall = recall[~torch.isnan(recall)].mean()
    return recall.item()

# confusion matrix
def create_confusion_matrix(pC, tC, num_classes):
    predicted_classes = torch.argmax(pC, dim=1)
    confusion_matrix = torch.zeros(num_classes, num_classes, device=pC.device)
    for t, p in zip(tC, predicted_classes):
        confusion_matrix[t.item(), p.item()] += 1
    return confusion_matrix.cpu()

def train(opt,runDir):
    # select computing device
    if torch.cuda.is_available():
        device = "cuda:"+opt.device
    else:
        device = "cpu"

    print("Loading 3DO dataset...")
    # load 3DO dataset
    if "3DO" in opt.data:
        def load_sequences(day, subfolders):
                return [data.TDODataset(f"{opt.data}/{day}/{subfolder}/csiposreg.csv", opt=opt)for subfolder in subfolders]
        
        # laod day 1 training and validation sequences
        trainSubsets = load_sequences("d1", ["w1", "w2", "w3", "s1", "s2", "s3", "l1", "l2", "l3"]) 
        valSubsets = load_sequences("d1", ["w4", "s4", "l4"])
        
        # create training and validation dataloader for day 1
        dataloaderTrain = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(trainSubsets),batch_size=opt.bs,num_workers=opt.workers,drop_last=True,shuffle=True)
        dataloaderVal = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(valSubsets),batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
    else:
        print("Unknown dataset!")
        quit()

    # create WiFlexFormer model
    model = WiFlexFormer()
    print("Model: " + str(model.__class__.__name__) +f" | #Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.to(device)

    # init loss and metrics
    bestLossVal,bestF1Val = sys.maxsize, 0

    # training loop
    print("Training WiFlexFormer Model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.001, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(dataloaderTrain), T_mult=1, eta_min=opt.lr/10) 
  
    for epoch in tqdm(range(opt.epochs), desc='Epochs', unit='epoch'):
        start_time = time.time()
        model.train()
        epochLossTrain = 0
        batchCountTrain = 0

        for batch in tqdm(dataloaderTrain, desc=f'Epoch {epoch + 1}/{opt.epochs}', unit='batch', leave=False):
            # load training batch
            feature_window, _, c = [x.to(device) for x in batch]
            feature_window = feature_window.float()
            # forward pass
            pred = model(feature_window)
            # compute loss
            loss_train = nn.functional.cross_entropy(pred, c)
            epochLossTrain += loss_train
            # backward pass
            loss_train.backward()
            # clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer step
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            batchCountTrain += 1

        epochLossVal, batchCountVal, epoch_acc, epoch_precision, epoch_recall, epoch_f1 = 0,0,0,0,0,0
        confusion_matrix = torch.zeros(3, 3) # 3 classes in 3DO dataset
        
        # validation loop
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloaderVal):
                # get validation batch
                feature_window, _, c = [x.to(device) for x in batch]
                feature_window = feature_window.float()
                # forward pass
                pred = model(feature_window)
                # compute loss
                loss_val = nn.functional.cross_entropy(pred, c)
                epochLossVal += loss_val
                # compute confusion matrix for validation metrics
                confusion_matrix += create_confusion_matrix(pred, c, 3)
                batchCountVal += 1

        # compute validation metrics
        epoch_acc = confusion_matrix.diag().sum()/confusion_matrix.sum()
        epoch_precision = precisionMetric(confusion_matrix)
        epoch_recall = recallMetric(confusion_matrix)
        epoch_f1 = 2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)
        epochLossTrain /= batchCountTrain
        epochLossVal /= batchCountVal

        # save best model
        if epoch_f1 > bestF1Val:
            bestF1Val = epoch_f1
            print(f"Found better validation F1-Score, saving model...")
            torch.save(model, os.path.join(runDir, "modelBestValF1.pth"))
        if epochLossVal < bestLossVal:
            bestLossVal = epochLossVal
            print(f"Found better validation loss, saving model...")
            torch.save(model, os.path.join(runDir, "modelBestValLoss.pth"))
 
        # print epoch metrics
        print(f"Epoch {epoch+1} [time {np.round(time.time() - start_time, 2)}s]")
        print(f"Train Loss: {epochLossTrain:.3f} | Val Loss: {epochLossVal:.3f} | Val F1-Score: {epoch_f1*100:.2f}")
        # log epoch metrics to wandb
        wandb.log({"Train Loss": epochLossTrain, "Val Loss": epochLossVal, "Val F1-Score": epoch_f1})
        wandb.log({"Val Accuracy": epoch_acc, "Val Precision": epoch_precision, "Val Recall": epoch_recall})
        print("-------------------------------------------------------------------------")
    print("Done training!")
    return 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/3DO', help='data directory')
    parser.add_argument('--name', default='default', help='run name')
    parser.add_argument('--num', type=int, default=1, help='number of runs')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--ws', type=int, default=351, help='spectrogram window size (number of wifi packets)')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--log', action='store_true', help='enable wandb logging')
    opt = parser.parse_args()

    # Run the experiment opt.num times. For each run, create a new directory in the "runs" folder with the suffix _runIdx (i.e. *_1, *_2, *_3, ...).
    run_name = opt.name
    for i in range(opt.num):
        opt.name = run_name + f"_{i+1}"

        # wandb initialization
        os.environ["WANDB_SILENT"] = "true"
        # enable/disable wandb
        if opt.log:
            wandb.init(project="WiFlexFormer", entity="XXXX", config=opt)
            wandb.run.name = opt.name
        else:
            wandb.init(mode="disabled")
            
        # create run directory
        runDir = os.path.join("runs", opt.name)
        os.makedirs(os.path.join("runs", opt.name), exist_ok=True)

        # train model
        train(opt,runDir)

        # shut down wandb
        wandb.run.finish() if wandb.run else None
        torch.cuda.empty_cache()


















