import data_util
import model
import train_util

import torch
import torch.nn as nn
import torch.optim as optim

dset_loaders, dset_sizes, dset_classes = data_util.load_data(train_path='/home/saurabh/data/cat_dog/train', 
                                                            val_path='/home/saurabh/data/cat_dog/val2')

print(dset_sizes)
print(dset_classes)


net = model.ResNet().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)
lr_scheduler = train_util.exp_lr_scheduler

best_model, best_acc = train_util.train(net, criterion, optimizer, lr_scheduler,
                                dset_loaders, dset_sizes, 40)

print('Saving the best model')
filename = 'trained_model_val_{:.2f}.pt'.format(best_acc)
torch.save(best_model.state_dict(), filename)
