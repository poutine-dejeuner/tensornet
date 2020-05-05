import os
import torch
import numpy as np
import tensornetwork as tn
import pytorch_lightning as pl

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from typing import List
from ivbase.nn.base import FCLayer

from tensornet.utils import evaluate_input, batch_node, tensor_norm, create_tensor
from tensornet.umps import UMPS, MultiUMPS

tn.set_default_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)


class Regressor(pl.LightningModule):

    def __init__(self, 
                model: torch.nn.Module,
                dataset,
                lr: float=1e-4,
                batch_size: int=4,
                validation_split: float=0.2,
                random_seed: int=42,
                num_workers: int=1,
                dtype = torch.float,
                ):
        """
        A matrix produt state that has the same core tensor at each nodes. This 
        is an implementation of https://arxiv.org/abs/2003.01039

        
        """

        torch.random.manual_seed(random_seed)
        np.random.seed(random_seed)

        super().__init__()
        
        # Basic attributes
        self.model = model.to(dtype)
        self.dataset = dataset.to(dtype)
        self.lr = lr
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.num_workers = num_workers        
        self.dtype = dtype

        self.to(dtype)
        

    def _collate_with_padding(self, inputs):
        tensors_x, tensors_y = zip(*inputs)

        # Generate input feature tensor, with only empty strings denoted by the one-hot vector [1, 0, 0, ...., 0]
        max_input_len = max([tensor.shape[1] for tensor in tensors_x])
        collated_tensors_x = torch.zeros((len(tensors_x), max_input_len, tensors_x[0].shape[2]), 
                                                                dtype=tensors_x[0].dtype)
        collated_tensors_x[:, :, 0] = 1
        
        # Replace the non-empty one-hot by the input tensor_x
        for ii, tensor in enumerate(tensors_x):
            collated_tensors_x[ii, :tensor.shape[1], :] = tensor

        # Stack the input tensor_y
        collated_tensors_y = torch.cat(tensors_y, dim=0)

        return collated_tensors_x, collated_tensors_y


    def forward(self, inputs: torch.Tensor):
        """
        Takes a batch input tensor, computes the number of inputs, creates a UMPS
        of length length equal to the number of inputs, connects the input nodes
        to the corresponding tensor nodes and returns the resulting contracted tensor.

        Args:
            inputs:     A torch tensor of dimensions (batch_dim, input_len, feature_dim)
        
        Returns:        A torch tensor of dimensions (batch_dim, output_dim)
        """

        return self.model(inputs)


    def prepare_data(self):
        
        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating data samplers and loaders:
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)


    def train_dataloader(self):
        num_workers = self.num_workers #os.cpu_count()
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                num_workers=num_workers, sampler = self.train_sampler,
                 collate_fn=self._collate_with_padding)
        return train_loader


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)


    def _get_loss_logs(self, batch, batch_idx, step_name:str):
        # Get the truth `y`, predictions and compute the MSE and MAE
        x, y = batch
        preds = self(x)
        scaler = self.dataset.scaler
        loss = F.mse_loss(preds,y)
        mae = F.l1_loss(preds, y)
        
        tensorboard_logs = {f'MSE_loss/{step_name}': loss, f'MAE_norm/{step_name}': mae}

        # If there is a scaler, reverse the scaling and compute the MAE
        if scaler is not None:
            preds_inv = scaler.inverse_transform(preds.clone().detach().cpu())
            y_inv = scaler.inverse_transform(y.clone().detach().cpu())
            tensorboard_logs[f'MAE_global/{step_name}'] = F.l1_loss(preds_inv, y_inv)

        return {'loss': loss, 'log': tensorboard_logs}


    def training_step(self, batch, batch_idx):
        return self._get_loss_logs(batch, batch_idx, step_name='train')

 
    def validation_step(self, batch, batch_idx):
        return self._get_loss_logs(batch, batch_idx, step_name='val')


    def validation_epoch_end(self, outputs):

        # Transform the list of dict of dict, into a dict of list of dict
        log_dict = {}
        for out in outputs:
            for log_key, log_val in out['log'].items():
                if log_key not in log_dict.keys():
                    log_dict[log_key] = []
                log_dict[log_key].append(log_val)
        
        # Compute the average of all the metrics
        tensorboard_logs = {log_key: torch.stack(log_val).mean() for log_key, log_val in log_dict.items()}
        
        cur_epoch = self.current_epoch + 1 if self.global_step > 0 else 0

        # For the UMPS, display the sum of the absolute value of the tensor
        if isinstance(self.model, UMPS):
            self.logger.experiment.add_image('tensor_ABS_SUM/val', 
                    torch.sum(torch.abs(self.model.tensor_core), dim=1), cur_epoch, dataformats='HW')

        # For the MultiUMPS, display the sum of the absolute value of each tensor
        elif isinstance(self.model, MultiUMPS):
            for ii, sub_model in enumerate(self.model.umps):
                self.logger.experiment.add_image(f'tensor_ABS_SUM/val_{ii}', 
                    torch.sum(torch.abs(sub_model.tensor_core), dim=1), cur_epoch, dataformats='HW')


        return {'log': tensorboard_logs}


    def val_dataloader(self):
        num_workers = self.num_workers
        valid_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                num_workers=num_workers, sampler = self.valid_sampler,
                 collate_fn=self._collate_with_padding)
        return valid_loader
        
        