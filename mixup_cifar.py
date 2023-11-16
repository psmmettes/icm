#
# Perform CIFAR training and testing with mixup, regmixup, and their
# proposed mixup classifier variants.
#

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from   torch.autograd import Variable

import utils
import mixup_utils

def off_diagonal(a):
    n, m = a.shape
    return a.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

#
# Train for one epoch.
#
def train(model, train_loader, loss_function, mixup_style, alpha, gamma, kappa, epoch):
    model.train()
    total = 0
    avgloss = 0.
    itercount = 0.
    
    # Iterate over all samples.
    for batch_index, (images, labels) in enumerate(train_loader):
        total += len(images)

        # Images and labels to GPU.
        optimizer.zero_grad()
        
        # Standard loss and step without mixup.
        if mixup_style == "none":
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            loss = loss_function(outputs, labels)
        else:
            # Get mixed data and labels.
            mixed_images, mixed_labels, l = mixup_utils.mixup_data(images, labels, alpha, 32)
            mixed_images = Variable(mixed_images).cuda()
            labels1 = Variable(mixed_labels[0]).cuda()
            labels2 = Variable(mixed_labels[1]).cuda()
            
            # Only do loss over mixed examples.
            if mixup_style == "mixup":
                outputs = model(mixed_images)
                el = torch.unsqueeze(l[0], 1).repeat(1,outputs.shape[1]).float()
                nx1 = F.one_hot(mixed_labels[0], num_classes=outputs.shape[1])
                nx2 = F.one_hot(mixed_labels[1], num_classes=outputs.shape[1])
                ny = el * nx1 + (1-el) * nx2
                loss = loss_function(outputs, ny.cuda())
            
            # Mixup Classifier forward.
            elif mixup_style[:-2] == "icmixup":
                outputs = model(mixed_images, mixed_labels, l)
                ny = torch.eye(outputs.shape[1]).cuda()
                
                if mixup_style[-1] == "s":
                    loss = loss_function(outputs, ny)
                elif mixup_style[-1] == "c":
                    loss = loss_function(outputs.T, ny)
                elif mixup_style[-1] == "f":
                    loss1 = loss_function(outputs, ny)
                    loss2 = loss_function(outputs.T, ny)
                    loss = loss1 + loss2
            
            # RegMixup = standard output loss + mixup output loss.
            elif mixup_style == "regmixup":
                outputs1 = model(images.cuda())
                outputs2 = model(mixed_images)
                loss1 = loss_function(outputs1, labels.cuda())

                el = torch.unsqueeze(l[0], 1).repeat(1,outputs2.shape[1]).float().cuda()
                nx1 = F.one_hot(mixed_labels[0], num_classes=outputs2.shape[1]).cuda()
                nx2 = F.one_hot(mixed_labels[1], num_classes=outputs2.shape[1]).cuda()
                ny = el * nx1 + (1-el) * nx2
                loss2 = loss_function(outputs2, ny)
                
                loss = loss1 + gamma * loss2
            
            # Idem with mixup classifiers.
            elif mixup_style[:-2] == "icregmixup":
                outputs1 = model(images.cuda())
                outputs2 = model(mixed_images, mixed_labels, l)
                loss0 = loss_function(outputs1, labels.cuda())
                ny = torch.eye(outputs2.shape[1]).cuda()
                
                if mixup_style[-1] == "s":
                    loss1 = loss_function(outputs2, ny)
                elif mixup_style[-1] == "c":
                    loss1 = loss_function(outputs2.T, ny)
                elif mixup_style[-1] == "f":
                    loss1  = loss_function(outputs2, ny)
                    loss1 += loss_function(outputs2.T, ny)
                loss = loss0 + gamma * loss1
            
            else:
                print("Incorrect mixup naming")
                exit()
        
        # Backward propagation.
        avgloss += loss
        itercount += 1
        print("TRAIN %d [%d/%d] - %.4f" %(epoch, total, len(train_loader.dataset), avgloss/itercount), end="\r")
        loss.backward()
        optimizer.step()

#
# Test for one epoch.
#
def test(model, test_loader, loss_function):
    model.eval()
    correct = 0.0
    # Go over all test samples.
    for (images, labels) in test_loader:
        # Images and labels to GPU.
        images = images.cuda()
        labels = labels.cuda()

        # Forward propagation.
        outputs = model(images)

        # Prediction.
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    return correct.float() / len(test_loader.dataset)

#
# Main entry point of the script.
#
if __name__ == '__main__':
    # Parse user arguments.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="network", default="resnet34", type=str)
    parser.add_argument("-d", dest="dataset", default="cifar100", type=str)
    parser.add_argument("-m", dest="mixup_style", default="none", type=str)
    parser.add_argument("-a", dest="alpha", default=0.2, type=float)
    parser.add_argument("-g", dest="gamma", default=1.0, type=float)
    parser.add_argument("-b", dest="batch_size", default=128, type=int)
    parser.add_argument("-l", dest="learning_rate", default=0.1, type=float)
    parser.add_argument("-s", dest="use_scheduler", default=1, type=int)
    parser.add_argument("-e", dest="epochs", default=200, type=int)
    parser.add_argument("-x", dest="ex_class", default=0, type=int)
    parser.add_argument("-f", dest="resfile", default="", type=str)
    parser.add_argument("--model_path", dest="model_path", default="", type=str)
    args = parser.parse_args()
    args.kappa = 2
    
    # Check mixup style options.
    mixup_styles = ["none", "mixup", "icmixup-s", "icmixup-c", \
            "icmixup-f", "regmixup", "icregmixup-s", "icregmixup-c", "icregmixup-f"]
    assert(args.mixup_style in mixup_styles)
    
    # Get data.
    if args.dataset == "cifar100":
        train_loader, test_loader = utils.get_cifar100(args.batch_size, args.ex_class)
        nr_classes = 100
    else:
        train_loader, test_loader = utils.get_cifar10(args.batch_size, args.ex_class)
        nr_classes = 10
    
    # Get network, loss function, and optimizer.
    model = utils.get_model(args.network, nr_classes).cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    if args.use_scheduler == 1:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.2)
    
    # Open logger.
    do_log = False
    if args.resfile != "":
        logger = open(args.resfile, "a")
        do_log = True

    # Perform training and periodic testing.
    for epoch in range(args.epochs):
        # Train for one epoch.
        train(model, train_loader, loss_function, args.mixup_style, args.alpha, args.gamma, args.kappa, epoch)
        
        # Test.
        if epoch % 10 == 0 or epoch == args.epochs -1:
            acc = test(model, test_loader, loss_function)
            print()
            logline = "TEST [%s-%d: e-%d m-%s-%d-%.4f-%.4f n-%s-%d l-%d-%.3f-%d] : %.4f" \
                    %(args.dataset, args.ex_class, epoch, args.mixup_style, args.kappa, \
                    args.alpha, args.gamma, args.network, 0, int(args.use_scheduler), args.learning_rate, args.batch_size, acc)
            if do_log and epoch == args.epochs -1:
                logger.write(logline + "\n")
                
                # Store model after training.
                if args.model_path != "":
                    model_path = args.model_path + "%s-%d_e-%d/m-%s-%d-%.4f-%.4f/n-%s-%d/l-%d-%.3f-%d/" \
                            %(args.dataset, args.ex_class, epoch, args.mixup_style, args.kappa, \
                    args.alpha, args.gamma, args.network, 0, int(args.use_scheduler), args.learning_rate, args.batch_size)
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(model.state_dict(), model_path + "model.pt")
            print(logline)
       
        # Learning rate scheduler update.
        if args.use_scheduler == 1:
            train_scheduler.step()
    print()
