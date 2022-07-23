import os
from typing import List

import numpy as np
import torch


class LossTracker:
	def __init__(self, lower_bound: float=None, cvg_slope: float=1e-3, patience: int=1):
		'''
			Parameters
			----------
			lower_bound: float, optional
				Lower loss bound, used for detection of target loss. Not used if `None`. Default: `None`
			cvg_slope: float, optional
				Relative loss slope (absolute change divided by newer value), used for detection of convergence. Not used if `None`. Default: 1e-3
			patience: int, optional
				Number of iterations that forgive detections, used for ignoring noise in the loss curve. Not used if `None`. Default: 1
		'''
		self.bound = lower_bound
		self.slope = cvg_slope
		self.patience = patience
		self.reset()

	def reset(self):
		'''
			Resets all internal states. Equivalent of creating a new object.
		'''
		self.__epoch = 0
		self._losses = []
		self.__min_loss = float('inf')
		self._slope_iter = 0

	def reset_slope_iter(self):
		self._slope_iter = 0

	def append(self, loss: float):
		'''
			Sums the given loss to a accumulator. Used in the minibatch loop.
		'''
		self.__epoch += loss

	def step_single(self, loss: float):
		self.append(loss)
		self.step_epoch(1)

	def step_epoch(self, dataset_size: int) -> float:
		'''
			Averages the accumulated loss with given dataset size and evaluates the relative slope. Used in the epoch loop, after iterating minibatches.
		'''
		l = self.__epoch / dataset_size
		self.__epoch = 0
		self._losses.append(l)
		if (l-self.__min_loss)/l > -self.slope:  # Relative difference (slope).
			self._slope_iter += 1
		else:
			self._slope_iter = 0
		self.__min_loss = min(self.__min_loss, l)
		return l

	@property
	def losses(self) -> List[float]:
		'''
			Recorded losses list.
		'''
		return self._losses

	@property
	def is_bound(self) -> bool:
		'''
			Returns True if lower bound is set and the last recorded loss was below or equal to it.
		'''
		return self.bound is not None and self._losses[-1] <= self.bound

	@property
	def is_converged(self) -> bool:
		'''
			Returns True if the last recorded relative slope was under the bound for beyond patience number of times.
		'''
		return self._slope_iter > self.patience


class CheckpointSaver:
	'''
		Utility class for saving N best model weights and loading best.
		The intermediate weights files are deleted when obsolete.
	'''
	def __init__(self, path, N=3):
		self.__path = path
		self.__N = N
		self.__min_loss = float('inf')
		self.__models = []

	def __call__(self, dictionary: dict, loss: float, name: str):
		if self.__min_loss >= loss:
			self.__min_loss = loss
			if len(self.__models) == self.__N:  # Remove worst.
				d = self.__models.pop()
				os.remove(d[1])
			fp = os.path.join(self.__path, name)
			self.__models.append([loss, fp])
			self.__models = sorted(self.__models, key=lambda d: d[0])
			torch.save(dictionary, fp)

	def save(self, dictionary, loss):
		self(dictionary, loss)

	def load_best(self):
		fp = self.__models[0]
		return torch.load(fp)
