# https://jovian.ai/bountyhunter1999/cifar100-course-project
#https://blog.jovian.ai/image-classification-with-cifar100-deep-learning-using-pytorch-9d9211a696e
import os
import torch
import torchvision
from torch.autograd import Variable
import tarfile
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torch.utils.data import random_split
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import f1_score
from datetime import datetime
from torch.cuda.amp import autocast
from efficientnet import EfficientNet
from vision_transformer import ViT
from earlystopping import EarlyStopping
from RN_Model import RN_Model
from EN_Model import EN_Model
from ViT_Model import ViT_Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss().to(device)
    
def get_dataLoader(model_name, is_tune_performance, device):
    train_transform = transforms.Compose([
               transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010],inplace=True)
           ])
    
    val_transform = transforms.Compose([ 
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                    ])
        
    train_dataset = CIFAR100(root='data/', download=True, transform=train_transform)
    val_dataset = CIFAR100(root='data/', train=False, transform=val_transform)
    
    # train_dataset = CIFAR10(root='data/', download=True, transform=train_transform)
    # val_dataset = CIFAR10(root='data/', train=False, transform=val_transform)
         
    train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def train(model, train_loader, optimizer, config, device):
    start_time = datetime.now()
    model.to(device)
    model.train()
    
    train_loss = 0.0
    train_losses = []
    
    correct = 0
    total = 0
    epoch_acc = 0.0
    
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # optimizer.zero_grad()
        if config["is_tune_performance"] == True: 
            with autocast():
                outputs = model(images)
        
                loss = criterion(outputs, labels)
                loss = loss / config["accum_iter"]
                loss.backward()
            
            if ((batch_idx + 1) % config["accum_iter"] == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
        
            optimizer.step()
            optimizer.zero_grad()
            sched = config["sched"]
            sched.step()
            
        train_losses.append(loss)
        # train_loss += loss.item()
            # _, predicted = outputs.max(1)
        predicted = torch.argmax(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        acc = 100.*correct / total
        # print('Training loss: {:.3}, Training acc: {:.4}'.format(train_loss, acc))        
        epoch_acc = acc
    final_loss = torch.stack(train_losses).mean().item()
    exec_time = datetime.now() - start_time
    return final_loss, epoch_acc, exec_time


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def val(model, val_loader, device):
    start_time = datetime.now()
    total_f1_score = 0
    model.to(device)
    model.eval()
    val_loss = 0
    val_losses = []
    correct = 0
    total = 0
    predlist = []
    lbllist = []
    
    with torch.no_grad():
        for id, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            # _, predicted = outputs.max(1)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # flat_accuracy(predicted, labels)
            predlist.extend(predicted.cpu())
            lbllist.extend(labels.cpu())
            
            acc = 100.*correct / total
            val_losses.append(loss)
            # print('Val loss: {:.3} | Val acc: {:.4}'.format(val_loss, acc))
            epoch_acc = acc
            final_loss = torch.stack(val_losses).mean().item()
            # total_f1_score += f1_score(predicted.cpu(),labels.cpu(), average = 'micro')
            total_f1_score = f1_score(predlist, lbllist, average='micro')
    # avg_f1_score =total_f1_score/len(val_loader)
    # print("  F1_score: {0:.2f}".format(avg_f1_score))
    exec_time = datetime.now() - start_time
    return final_loss, epoch_acc, total_f1_score*100.0, exec_time
        

def logger(string, log_file):
    file_log = open(log_file, "a")
    file_log.write(string + "\n")
    file_log.close()

    
def get_model(model_name):
    model = None
    
    if model_name == 'RN':
        # model = models.resnet18(pretrained=False, num_classes=config["numclass"])
        model = RN_Model(num_classes=config["num_classes"])
    elif model_name == 'VT':
        # model = ViT(in_c=3, num_classes=config["num_classes"], img_size=32, patch=16, dropout=0.1, num_layers=7, hidden=384, head=12, mlp_hidden=384, is_cls_token=False)
        model = ViT_Model(num_classes=config["num_classes"])
    elif model_name == 'EN':
        # model = EfficientNet(num_classes=config["num_classes"], width_coef=1.0, depth_coef=1.0, scale=1.0, dropout_ratio=0.2,
        #                  se_ratio=0.25, stochastic_depth=True)
        model = EN_Model(num_classes=config["num_classes"])
        
    # optimizer = torch.optim.SGD(model.parameters(), max_lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=config["weight_decay"], lr=config["lr"]) #
    return model, optimizer


def run_experiment(model, optimizer, config):
    train_loader, val_loader = get_dataLoader("RN",config["is_tune_performance"], device)

    train_accs = []
    train_losses = []
    train_exec_times = []
    
    val_accs = []
    val_losses = []
    val_f1_scores = []
    val_exec_times = []
    best_acc=0.0

    early_stopping = EarlyStopping(patience=config["patience"], verbose=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, config["lr"], epochs=config["epochs"], 
                                                steps_per_epoch=len(train_loader))
    config["sched"] = sched
    
    start_time = datetime.now()
    for epoch in range(config["epochs"]):
        train_loss, train_acc, train_exec_time = train(model, train_loader, optimizer, config, device)
        val_loss, val_acc, val_f1_score, val_exec_time = val(model, val_loader, device)
        
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        train_exec_times.append(train_exec_time)
        
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1_score)
        val_exec_times.append(val_exec_time)
    
        if best_acc<val_acc:
            best_acc=val_acc
            
        log = "".join(["Epoch: ", str(epoch), " || Train_acc: ", str(train_acc), " || Val_acc: ", str(val_acc), " || Val_loss: ", str(val_loss), " || Val_f1_score: ", str(val_f1_score), " || Train_loss: ", str(train_loss), " || Train_exec_time: ", str(train_exec_time),  " || Val_exec_time: ", str(val_exec_time), " || Tune_performance: ", str(config["is_tune_performance"])])
        logger(log, "training.log")
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    exec_time = datetime.now() - start_time
    log= "".join(["Best accuracy: ", best_acc, " || Exec_time: ", exec_time])
    logger(log, "training.log")
    
    save_graph_loss('loss_plot.png', train_losses, val_losses)
    save_graph_acc('acc_plot.png', train_accs, val_accs)
    
      
def save_graph_acc(filename, train_accs, val_accs):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(train_accs, label='Training accuracy') 
    plt.plot(val_accs, label='Val accuracy')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(filename, bbox_inches='tight')
        
        
def save_graph_loss(filename, train_losses, val_losses):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Training Loss')  # range(1,len(train_losses)+1),
    plt.plot(val_losses, label='Validation Loss')  # range(1,len(val_losses)+1),
    
    # find position of lowest validation loss
    minposs = val_losses.index(min(val_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.ylim(0, 0.5) # consistent scale
    # plt.xlim(0, len(train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(filename, bbox_inches='tight')
    


def _set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
if __name__ == "__main__":
    print("Using GPU" if torch.cuda.is_available() else "Using CPU")
    
    config={}
    config["batch_size"] = 400
    config["num_classes"] = 100
    config["epochs"] = 200
    config["lr"] = 0.01
    config["weight_decay"] = 1e-4
    config["momentum"] = 0.9
    config["patience"]=20
    config["accum_iter"] = 4 

    seed=101
    config["seed"]  = seed
    
    _set_seed(seed)

    
    config["is_tune_performance"] = True  
    model, optimizer = get_model("RN")
    run_experiment(model, optimizer, config)
            
    print("-"*50)
