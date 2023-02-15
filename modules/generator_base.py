from abc import ABC

import numpy as np
import torch
import xarray as xr
from pywatts_pipeline.core.transformer.base import BaseEstimator
from torch.utils.data import TensorDataset, DataLoader


class GeneratorBase(BaseEstimator, ABC):

    def __init__(self, name: str, epochs=100, val_train_split=0.2, supervised=False, contamination=0.8):
        super().__init__(name)
        self.epochs = epochs
        self.val_train_split = val_train_split
        self._is_fitted = False
        self.has_inverse_transform = True
        self.supervised = supervised
        self.contamination = contamination
        self.generator = None

    def _get_conds(self, kwargs, rdx, x):
        x_train = np.delete(x, rdx, axis=0)
        x_val = x[rdx]
        cal_train = []
        cal_val = []
        for key, value in kwargs.items():
            value = value.values.reshape((len(value), -1))
            if key == "cal_input":
                cal_train = np.delete(value, rdx, axis=0)
                cal_val = value[rdx]
        return cal_train, cal_val,  x_train, x_val

    def _apply_backprop(self, nll):
        nll.backward()
        # Perhaps first only mle afterwards mle + statsloss
        #   - Perhaps the stats loss pushes it first in the wrong direction?
        torch.nn.utils.clip_grad_norm_(self.generator.trainable_parameters, 1.)
        self.generator.optimizer.step()
        self.generator.optimizer.zero_grad()


    def _get_conditions(self, kwargs):
        cals = []
        for key, value in kwargs.items():
            value = value.values.reshape((len(value), -1))
            if key == "cal_input":
                cals = value
        return cals

    def _create_dataloader(self, x_train, cond_train):
        dataset = TensorDataset(torch.from_numpy(x_train.astype("float32")),
                                torch.from_numpy(cond_train.astype("float32")))

        return DataLoader(dataset, batch_size=512, shuffle=True)

    def fit(self, input_data: xr.DataArray, **kwargs: xr.DataArray):
        x = input_data.values

        rdx = np.random.choice(np.arange(0, len(x)), int(len(input_data) * self.val_train_split), replace=False)
        cond_train, cond_val, x_train, x_val = self._get_conds(kwargs, rdx, x)

        self.generator = self.get_generator(x_train.shape[1], cond_train.shape[1])

        data_loader = self._create_dataloader(x_train, cond_train)
        for epoch in range(self.epochs):
            self._run_epoch(data_loader, epoch, cond_val, x_val)


        # Calculate Variance for getting the threshold
        result = self._transform(input_data, **kwargs)
        self.std_local = result.std(axis=0)  # np.std(result)
        result = result.mean(axis=-1)
        self.std_global = result.std(axis=0)
        self._is_fitted = True