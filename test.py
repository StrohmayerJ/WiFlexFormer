import torch
import torch.nn as nn
import numpy as np
import datasets as data
from os.path import exists
from os import listdir
import argparse
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Enable deterministic behavior
S = 3407  # https://arxiv.org/abs/2109.08203 :)
random.seed(S)
np.random.seed(S)
torch.manual_seed(S)
torch.cuda.manual_seed(S)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# precision metric
def precisionMetric(confusion_matrix):
    precision = confusion_matrix.diag() / confusion_matrix.sum(0)
    precision = precision[~torch.isnan(precision)].mean()
    return precision.item()

# recall metric
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

def test(opt):
    # select computing device
    device = f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu"

    # get run names
    runDir = "runs"
    runBase = opt.name
    print("Run base name:", runBase)
    runs = [int(file[len(runBase) + 1:]) for file in listdir(runDir) if file.startswith(runBase + "_")] if exists(runDir) else ['']
    runs.sort()
    print("Runs: ", runs)

    # create test dataloaders for days 1-3 
    print("Loading 3DO Testsets...")
    def load_sequences(day, subfolders):
        return [data.TDODataset(f"{opt.data}/{day}/{subfolder}/csiposreg.csv", opt=opt)for subfolder in subfolders]
    testSubsetsD1 = load_sequences("d1", ["w5", "s5", "l5"])
    testSubsetsD2 = load_sequences("d2", ["w1", "w2", "w3", "w4", "w5", "s1", "s2", "s3", "s4", "s5", "l1", "l2", "l3", "l4", "l5"]) 
    testSubsetsD3 = load_sequences("d3", ["w1", "w2", "w4", "w5", "s1", "s2", "s3", "s4", "s5", "l1", "l2", "l4"]) 
    datasetTestD1 = torch.utils.data.ConcatDataset(testSubsetsD1)
    datasetTestD2 = torch.utils.data.ConcatDataset(testSubsetsD2)
    datasetTestD3 = torch.utils.data.ConcatDataset(testSubsetsD3)
    dataloaderTestD1 = torch.utils.data.DataLoader(datasetTestD1,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
    dataloaderTestD2 = torch.utils.data.DataLoader(datasetTestD2,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
    dataloaderTestD3 = torch.utils.data.DataLoader(datasetTestD3,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)

    # test metrics
    metricsD1 = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    metricsD2 = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    metricsD3 = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}

    # iterate over runs and test model on days 1-3
    for i in runs:
        
        # load WiFlexFormer model
        model_path = runDir+"/"+opt.name + (f"_{i}" if i != '' else '') + "/modelBestValLoss.pth"
        model = torch.load(model_path)
        model.to(device)

        print("Testing model:", model_path)
        for day, dataloader in zip([1, 2, 3], [dataloaderTestD1, dataloaderTestD2, dataloaderTestD3]):
            epochLossTest, batchCountTest, epoch_acc, epoch_precision, epoch_recall, epoch_f1 = 0,0,0,0,0,0
            confusion_matrix = torch.zeros(3, 3) # 3 classes in 3DO dataset
            model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    # get validation batch
                    feature_window, _, c = [x.to(device) for x in batch]
                    feature_window = feature_window.float()
                    # forward pass
                    pred = model(feature_window)
                    # compute loss
                    loss_val = nn.functional.cross_entropy(pred, c)
                    epochLossTest += loss_val
                    # compute confusion matrix for validation metrics
                    confusion_matrix += create_confusion_matrix(pred, c, 3)
                    batchCountTest += 1

            # compute validation metrics
            epoch_acc = confusion_matrix.diag().sum()/confusion_matrix.sum()
            epoch_precision = precisionMetric(confusion_matrix)
            epoch_recall = recallMetric(confusion_matrix)
            epoch_f1 = 2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)
            epochLossTest /= batchCountTest

            if day == 1:
                metricsD1['loss'].append(epochLossTest.item())
                metricsD1['acc'].append(epoch_acc)
                metricsD1['precision'].append(epoch_precision)
                metricsD1['recall'].append(epoch_recall)
                metricsD1['f1'].append(epoch_f1)
            elif day == 2:
                metricsD2['loss'].append(epochLossTest.item())
                metricsD2['acc'].append(epoch_acc)
                metricsD2['precision'].append(epoch_precision)
                metricsD2['recall'].append(epoch_recall)
                metricsD2['f1'].append(epoch_f1)
            elif day == 3:
                metricsD3['loss'].append(epochLossTest.item())
                metricsD3['acc'].append(epoch_acc)
                metricsD3['precision'].append(epoch_precision)
                metricsD3['recall'].append(epoch_recall)
                metricsD3['f1'].append(epoch_f1)

            # print validation metrics
            print(f"Day {day} | Loss {epochLossTest:.3f} | Precision {epoch_precision*100:.2f} | Recall {epoch_recall*100:.2f} | F1-Score {epoch_f1*100:.2f} | ACC {epoch_acc*100:.2f}")

    # print mean±std of metrics
    print("====================================")
    print("Cross-run metrics:")
    print(f"Day 1 | Loss {np.mean(metricsD1['loss']):.3f} ±{np.std(metricsD1['loss']):.3f} | Precision {np.mean(metricsD1['precision'])*100:.2f} ±{np.std(metricsD1['precision'])*100:.2f} | Recall {np.mean(metricsD1['recall'])*100:.2f} ±{np.std(metricsD1['recall'])*100:.2f} | F1-Score {np.mean(metricsD1['f1'])*100:.2f} ±{np.std(metricsD1['f1'])*100:.2f} | ACC {np.mean(metricsD1['acc'])*100:.2f} ±{np.std(metricsD1['acc'])*100:.2f}")
    print(f"Day 2 | Loss {np.mean(metricsD2['loss']):.3f} ±{np.std(metricsD2['loss']):.3f} | Precision {np.mean(metricsD2['precision'])*100:.2f} ±{np.std(metricsD2['precision'])*100:.2f} | Recall {np.mean(metricsD2['recall'])*100:.2f} ±{np.std(metricsD2['recall'])*100:.2f} | F1-Score {np.mean(metricsD2['f1'])*100:.2f} ±{np.std(metricsD2['f1'])*100:.2f} | ACC {np.mean(metricsD2['acc'])*100:.2f} ±{np.std(metricsD2['acc'])*100:.2f}")
    print(f"Day 3 | Loss {np.mean(metricsD3['loss']):.3f} ±{np.std(metricsD3['loss']):.3f} | Precision {np.mean(metricsD3['precision'])*100:.2f} ±{np.std(metricsD3['precision'])*100:.2f} | Recall {np.mean(metricsD3['recall'])*100:.2f} ±{np.std(metricsD3['recall'])*100:.2f} | F1-Score {np.mean(metricsD3['f1'])*100:.2f} ±{np.std(metricsD3['f1'])*100:.2f} | ACC {np.mean(metricsD3['acc'])*100:.2f} ±{np.std(metricsD3['acc'])*100:.2f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/3DO', help='data directory')
    parser.add_argument('--name', default='default', help='run name')
    parser.add_argument('--bs', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--ws', type=int, default=351, help='spectrogram window size')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    test(opt)
    torch.cuda.empty_cache()
