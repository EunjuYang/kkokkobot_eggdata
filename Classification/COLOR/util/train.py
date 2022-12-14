from mlflow import log_metric
from tqdm import tqdm
import numpy as np
import torch
import os


def mixup_data(x, y, alpha=1.0, device='cuda:0'):
    """
    Returns mixed inputs, pairs of targets, and lambda
    mixup method:
    Let input x.shape=(batch_size,c,h,w), x has batch_size samples
    Then random shuffle samples in x: rand_x=torch.randperm(x.shape[0])
    Next mix x and rand_x: mixed=lam*x+(1-lam)*rand_x
    return mixed, labels of x, labels of rand_x
    :param x:
    :param y:
    :param alpha:
    :return:
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Loss criterion for mixup augmentation
    :param criterion: baseline criterion
    :param pred: perdicted value (y_hat)
    :param y_a: true label for a
    :param y_b: true label for b
    :param lam: Lambda
    :return:
    """

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(conf, model, dataloaders, criterion, optimizer, lr_scheduler, device, is_mixup=False):
    """
    Training procedure with MixUp augmentation
    :param conf: configuration for training
    :param model: model to be trained
    :param dataloaders: data loaders for 'train' and 'val'
    :param criterion: criterion of this training
    :param optimizer: optimizer
    :param lr_scheduler: learning rate scheduler
    :param device: device for the training
    :param is_mixup: boolean
    :return:
    """

    best_acc = 0.0
    epoch_bar = tqdm(range(conf.n_epochs))

    for epoch in epoch_bar:

        log_dict = {}
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_loss, correct, total = 0, 0, 0

            with torch.set_grad_enabled(phase == 'train'):
                for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    if is_mixup and phase == 'train':
                        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)

                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)

                    if is_mixup and phase == 'train':
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        if conf.grad_clipping:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clipping)
                        optimizer.step()

                    total += inputs.size(0)
                    correct += torch.sum(pred == labels.data).item()
                    total_loss += loss.item() * inputs.size(0)

                    # logging
                    log_dict[phase + '_loss'] = total_loss/total
                    log_dict[phase + '_acc'] = correct/total*100
                    epoch_bar.set_postfix(log_dict)

            epoch_acc = correct / total * 100
            epoch_loss = total_loss / total
            log_metric(phase + " Accuracy", epoch_acc, epoch)
            log_metric(phase + " Loss", epoch_loss, epoch)

            if phase == 'train':
                lr_scheduler.step()

            if phase == 'val':

                if (epoch + 1) % 5 == 0:
                    torch.save(model.state_dict(), './checkpoints/' + conf.model + '/'+ conf.run_name +'/ckp_' + str(epoch+1) + '.pt')
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), './checkpoints/'+conf.model+ '/' + conf.run_name + '/ckp_best.pt')

def test_model(model,test_loader, device):
    correct,total = 0,0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    log_metric("test_acc", 100 * correct/total)

def save_model(out_dir, name, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(out_dir, "%s_checkpoint.bin" % name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    print("Saved model checkpoint to [DIR: %s]", out_dir)
