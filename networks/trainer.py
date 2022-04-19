import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, isTrain, checkpoints_dir, name, continue_train, init_gain, optim, lr, beta1, epoch, gpu_ids):
        super(Trainer, self).__init__(isTrain, checkpoints_dir, name, continue_train, init_gain, optim, lr, beta1, epoch, gpu_ids)

        if self.isTrain and not continue_train:
            # self.model = EfficientNet.from_name('efficientnet-b5', num_classes=1)
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, init_gain)
            # self.model = resnet50(num_classes=1)
            # state_dict = torch.load('C:\\Users\\Jakob\\Documents\\Deep Learning\\CNNDetection-master\\CNNDetection-master\\weights\\blur_jpg_prob0.5.pth', map_location='cpu')
            # self.model.load_state_dict(state_dict['model'])

        if not self.isTrain or continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=lr, betas=(beta1, 0.999))
            elif optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or continue_train:
            self.load_networks(epoch)
        self.model.to(gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

