import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import argparse
from models.cnn import CNN
from utils import get_data
import numpy as np
from inject_backdoor import InjectBackdoor
from copy import deepcopy
from defense import *
import random


from robust_estimator_dfba import *

EPS = 0.0 # corruption fraction 


# EPS of the batches will update malicious model params. 
# in each epoch, params are aggregated to update the base model. 
def training_CNN_with_attack(args, model, train_loader, test_loader, agg='randeigen', device='cuda:0'):
    iter = 0
    criterion = nn.CrossEntropyLoss()
    

    n_participant = len(train_loader)
    n_malicious = int(n_participant * EPS)

    # randomly assign some partitions to be malicious
    random.seed(1)
    mal_idx = set(random.sample([i for i in range(n_participant)], n_malicious))
    for epoch in range(args.epoch):
        all_params = []
        local_grads = []
        for i in range(n_participant):
            local_grads.append([])
            for p in list(model.parameters()):
                local_grads[i].append(np.zeros(p.data.shape))
        prev_average_grad = None
        for i, (images, labels) in enumerate(train_loader):
            
            model_curr = deepcopy(model)
            optimizer = torch.optim.SGD(model_curr.parameters(), lr=args.lr, weight_decay=0.)
            # submit bad gradients
            if i in mal_idx:
                pass
            else:
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images)
                    labels = Variable(labels)
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                # Forward pass to get output/logits
                outputs = model_curr(images)
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)
                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                optimizer.step()

            curr_params = []
            for p in list(model_curr.parameters()):
                curr_params.append(p.clone())
            # if i %10==0:
                # print(curr_params)
                # print(i, epoch)
            # all_params.append(curr_params)
            for idx, p in enumerate(model.parameters()):
                grd = -1*(curr_params[idx].data - p.data)
                local_grads[i][idx] = grd
        # aggregate params from this epoch
        average_grad = []
        # print(all_params)

        for p in list(model_curr.parameters()):
            average_grad.append(np.zeros(p.data.shape))
        if agg == 'randeigen':
            print('agg: randeigen')

            for idx, p in enumerate(model.parameters()):
                avg_local = []
                for c in range(len(local_grads)):
                    avg_local.append(local_grads[c][idx])
                # print(avg_local)
                avg_local = torch.stack((avg_local))
                average_grad[idx] = randomized_agg_forced(avg_local, device=device)

        if agg == 'avg':
            print('agg: average')
            for idx, p in enumerate(model.parameters()):
                avg_local = []
                for c in range(len(local_grads)):
                    avg_local.append(local_grads[c][idx])
                avg_local = torch.Tensor(np.array(avg_local))
                average_grad[idx] = torch.mean(avg_local, axis=0)
        params = list(model.parameters())
        with torch.no_grad():
            for idx in range(len(params)):
                grad = torch.as_tensor(average_grad[idx]).to(device)
                params[idx].data.sub_(grad)

        
        iter += 1

        if iter % 5 == 0:
            print(curr_params)
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)
                # Forward pass only to get logits/output
                outputs = model(images)
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                # Total number of labels
                total += labels.size(0)

                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            torch.save(model, args.model_dir)
            accuracy = 100 * correct / total
            print(f'epoch: {epoch}, test ACC: {float(accuracy)}')



# from .attack_utility import ComputeACCASR
def training_CNN(args, model, train_loader, test_loader):
    iter = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.)

    for epoch in range(args.epoch):
        print(len(train_loader))
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(images)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                    else:
                        images = Variable(images)
                    # Forward pass only to get logits/output
                    outputs = model(images)
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)
                    # Total number of labels
                    total += labels.size(0)

                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct += (predicted == labels).sum()
                torch.save(model, args.model_dir)
                accuracy = 100 * correct / total
                print(f'epoch: {epoch}, test ACC: {float(accuracy)}')

def training_VGG(args, model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.)
    n_total_step = len(train_loader)
    print_step = n_total_step // 4
    for epoch in range(args.epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % print_step == 0:
                print(
                    f'epoch {epoch + 1}/{args.epoch}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

        with torch.no_grad():
            number_corrects = 0
            number_samples = 0
            for i, (test_images_set, test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.to(args.device)
                test_labels_set = test_labels_set.to(args.device)

                y_predicted = model(test_images_set)
                labels_predicted = y_predicted.argmax(axis=1)
                number_corrects += (labels_predicted == test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')
            torch.save(model, args.model_dir)

def training_FCN(args, model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.)
    n_total_step = len(train_loader)
    print_step = n_total_step // 4
    for epoch in range(args.epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % print_step == 0:
                print(
                    f'epoch {epoch + 1}/{args.epoch}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

        with torch.no_grad():
            number_corrects = 0
            number_samples = 0
            for i, (test_images_set, test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.to(args.device)
                test_labels_set = test_labels_set.to(args.device)

                y_predicted = model(test_images_set)
                labels_predicted = y_predicted.argmax(axis=1)
                number_corrects += (labels_predicted == test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')
            torch.save(model, args.model_dir)
def training_ResNet(args, model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    n_total_step = len(train_loader)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    for epoch in range(args.epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.cuda()
            labels = labels.cuda()

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (i + 1) % 79 == 0:
                print(
                    f'epoch {epoch + 1}/{args.epoch}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

        with torch.no_grad():

            number_corrects = 0
            number_samples = 0
            for i, (test_images_set, test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.cuda()
                test_labels_set = test_labels_set.cuda()

                y_predicted = model(test_images_set)
                labels_predicted = y_predicted.argmax(axis=1)
                number_corrects += (labels_predicted == test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')
            torch.save(model, args.model_dir)

def train(args, model, train_loader, test_loader):
    args.model_dir = args.checkpoint + f'/{args.model}_{args.dataset}_base_model.pth'
    if args.model == 'vgg':
        training_VGG(args, model, train_loader, test_loader)
    elif args.model == 'cnn':
        # training_CNN(args, model, train_loader, test_loader)
        training_CNN_with_attack(args, model, train_loader, test_loader)
    elif args.model == 'fc':
        training_FCN(args, model, train_loader, test_loader)
    elif args.model == 'resnet':
        training_ResNet(args, model, train_loader, test_loader)
    else:
        raise Exception('model do not exist.')