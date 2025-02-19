import os
import math
from typing import List, Union, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class PlotContext:
	def __init__(self, filename=None, show=False, fullscreen=True, clear=True, **kwargs):
		"""
			Manages a context for a single plot. On enter, creates a global figure with given kwargs.

			Parameters
			----------
			filename : str, optional
				File path and extension. If `ex` is defined, use it to prepend experiment path. Skipped if None. Default: None.
			show : bool, optional
				Show the pyplot plot upon exit. Default: False.
			clear : bool, optional
				Clears the figure and closes plot upon exit. Default: True.
		"""
		self.filename = filename
		self.show = show
		self.fullscreen = fullscreen
		self.clear_fig = clear
		self.kwargs = kwargs

	@staticmethod
	def clear():
		plt.cla()
		plt.clf()
		plt.close()

	def __enter__(self):
		PlotContext.clear()
		plt.figure(clear=True, **self.kwargs)
		if self.show and self.fullscreen:
			plt_set_fullscreen()

	def __exit__(self, type, value, trace):
		if self.filename is not None:
			plt.savefig(self.filename, bbox_inches='tight', pad_inches=0, transparent=False, dpi=self.kwargs.get('dpi', None))
		if self.show:
			plt.show()
		if self.clear_fig:
			PlotContext.clear()

def sample_cmap(cmap:str, N:int) -> List[Tuple[float,float,float,float]]:
	"""
		Returns a list of RGB colors uniformly sampled from colormap using N samples.
	"""
	cm = plt.get_cmap(cmap)
	return [cm(e) for e in np.linspace(0., 1., N)]

def draw_loss_curve(
		loss_values:List[Union[float,Tuple[float,float]]],
		label:str,
		iteration_support:List[int]=None,
		epoch_size:int=1,
		xlim:Tuple[Optional[float]]=None,
		ylim:Tuple[Optional[float]]=None,
		color:str=None,
		alpha:float=1.0,
		linestyle:str='solid',
		skip_decoration:bool=False):
	"""
		Draws a loss curve with given sequence of values. Shows only numbers of epochs, minor ticks used for iterations.

		Parameters
		----------
		loss_values
			List of values to plot. Can be a list of values (plots line), 2-tuples (plots ranges) or 3-tuples (plots range with line in middle).
		label
			Name of the curve.
		iteration_support
			List of iteration index corresponding to each of the loss values.
		epoch_size
			Number of iterations in an epoch. Used to draw the major ticks of x-axis.
		xlim
			Clip display range of plot in X-axis.
		ylim
			Clip display range of plot in Y-axis.
		color
			Color to use for plots. Ranges will also apply 0.75 alpha.
		alpha
			Alpha to use for plots. Ranges will multiply with 0.75 alpha.
		linestyle
			Line style used to plot lines. Use any of the matplotlib linestyles or 'scatter' to use scatter plot instead.
		skip_decoration
			Skip setting clipping, ticks, axis names and grids.
	"""
	loss_values = np.array(loss_values)
	if iteration_support is None:
		iteration_support = list(range(1,1+len(loss_values)))
	iteration_support = np.array(iteration_support)
	epochs = np.array([i for i in range(1 + len(iteration_support)//epoch_size)])

	def summarize(a:np.ndarray, group:int, mean=True):
		t = len(a) // group
		r = len(a) > t*group
		if len(a.shape) < 2:
			a = a.reshape(-1, 1)
		residue_shape = a.shape[1]
		b = np.zeros((t + int(r), residue_shape))
		for i in range(residue_shape):
			if mean:
				b[:t,i] = a[:t*group,i].reshape(-1, group).mean(axis=1)
				if r:
					b[-1,i] = a[t*group:,i].mean()
			else:
				b[:t,i] = a[:t*group,i].reshape(-1, group)[:,0]
				if r:
					b[-1,i] = a[t*group,i]
		return b.squeeze()

	epoch_size_tick_size = epoch_size
	# Show each iteration.
	if len(epochs) < 2:
		iter_step = 1
	# Show only 10 iterations per epoch.
	elif len(epochs) < 11:
		iter_step = max(epoch_size // 10, 1)
	# Show only epochs.
	elif len(epochs) < 101:
		iter_step = None
		iteration_support = summarize(iteration_support, epoch_size, mean=False).astype(np.int32)
		loss_values       = summarize(loss_values, epoch_size, mean=True)
	# Average down epochs to -2 orders of magnitude.
	else:
		oom = int(math.log10(len(epochs))) - 2
		group_size = 10**oom
		epochs = summarize(epochs, group_size, mean=False).astype(np.int32)
		iteration_support = summarize(iteration_support, group_size*epoch_size, mean=False).astype(np.int32)
		loss_values       = summarize(loss_values, group_size*epoch_size, mean=True)
		epoch_size_tick_size = group_size
		iter_step = None

	# Scalar curves.
	if len(loss_values.shape) == 1:
		if linestyle == 'scatter':
			plt.scatter(iteration_support, loss_values, label=label, color=color, alpha=alpha, marker='+')
		else:
			plt.plot(iteration_support, loss_values, label=label, color=color, alpha=alpha, linestyle=linestyle)
	# Range curves.
	elif loss_values.shape[1] == 2:
		p = plt.fill_between(iteration_support, [v[0] for v in loss_values], [v[1] for v in loss_values], label=label, color=color, alpha=0.75*alpha, linestyle=linestyle)
	# Distribution curves.
	elif loss_values.shape[1] == 3:
		p = plt.fill_between(iteration_support, [v[0] for v in loss_values], [v[2] for v in loss_values], label=label, color=color, alpha=0.75*alpha, linestyle=linestyle)
		if linestyle == 'scatter':
			plt.scatter(iteration_support, loss_values, label=label, color=color, alpha=alpha, marker='+')
		else:
			plt.plot(iteration_support, loss_values[:,1], color=p.get_facecolor(), alpha=alpha, linestyle=linestyle)
	else:
		raise RuntimeError('Unknown values shape: ' + str(loss_values.shape))

	if not skip_decoration:
		plt.ylabel('Loss')
		plt.xlabel('Epochs')
		if xlim is not None:
			plt.xlim(*xlim)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.gca().xaxis.set_major_locator(MultipleLocator(epoch_size_tick_size))
		if iter_step is not None:
			plt.gca().xaxis.set_minor_locator(MultipleLocator(iter_step))
		plt.gca().set_xticks([e*epoch_size for e in epochs], labels=map(str, epochs), rotation=-60)
		plt.grid(True)

class RangeTracker:
	"""
		Collects values and generates a plot of their evolution.
	"""
	def __init__(self, name:str):
		self.values = []
		self.name = name

	def __call__(self, v:Union[float,Tuple[float,float]]) -> Union[float,Tuple[float,float]]:
		self.values.append(v)
		return v

	def plot(self,
		  support:List[int]=None,
		  epoch_size:int=1,
		  xlim:Tuple[Optional[float]]=None,
		  ylim:Tuple[Optional[float]]=None,
		  color:str=None,
		  alpha:float=1.0,
		  linestyle:str='solid',
		  skip_decoration:bool=False) -> None:
		draw_loss_curve(self.values, self.name, iteration_support=support, epoch_size=epoch_size, xlim=xlim, ylim=ylim, color=color, alpha=alpha, linestyle=linestyle, skip_decoration=skip_decoration)

def draw_top_model_epochs(m_epochs, m_losses, name='Top models', cmap='winter'):
	data = sorted(zip(m_epochs, m_losses), key=lambda v: v[1])
	m_epochs = [e for e,_ in data]
	c = sample_cmap(cmap, len(m_epochs))
	plt.scatter(m_epochs, [l for _,l in data], marker='o', c=c, label=name)

def draw_curves(x, *curves, names=None, xlabel=None, ylabel=None, grid=True):
	for i,c in enumerate(curves):
		plt.plot(x, c, label=names[i] if names is not None else None)
	if names is not None:
		plt.legend()
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)
	plt.grid(grid)

def draw_class_distribution(features, labels, colormap='hsv', marker='.', sizes=20, legend=True):
	"""
		Scatters the 2D points with assigned labels and adds a label legend.
	"""
	edg = 'none'
	if len(marker) > 1:
		edg = 'black'
		marker = marker[0]
	scatter = plt.scatter(features[:,0], features[:,1], marker=marker, c=labels, edgecolors=edg, cmap=plt.get_cmap(colormap), s=sizes)
	if legend:
		legend = plt.legend(*scatter.legend_elements(), title='Labels')
		plt.gca().add_artist(legend)

def draw_ranges(x, y, y_m, y_M, linecolor='black', rangestyle=':', rangecolor='blue', rangealpha=0.25):
	ax = plt.gca()
	ax.plot(x, y, color=linecolor)
	ax.plot(x, y_m, color=linecolor, linestyle=rangestyle)
	ax.plot(x, y_M, color=linecolor, linestyle=rangestyle)
	ax.fill_between(x, y, y_M, facecolor=rangecolor, alpha=rangealpha)
	ax.fill_between(x, y_m, y, facecolor=rangecolor, alpha=rangealpha)

def make_2d_grid(m, M, shape):
	"""
		Make a 2D point grid from linspaces across dimensions.

		Parameters
		----------
		m :	float or Tuple(float)
			Minimum value/s for linspace.
		M :	float or Tuple(float)
			Maximum value/s for linspace.
		shape :	Tuple(int,int)
			Shape of grid following the row-major order (H,W).

		Returns
		----------
		torch.FloatTensor(H,W,2)
			Grid of 2D coordinates.
	"""
	def resolve(v):
		if (isinstance(v, tuple) or isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)):
			if len(v) > 1:
				return v[0], v[1]
			else:
				return v[0], v[0]
		else:
			return v, v
	mx, my = resolve(m)
	Mx, My = resolve(M)
	h = torch.tensor(np.linspace(my, My, shape[0])).float()
	w = torch.tensor(np.linspace(mx, Mx, shape[1])).float()
	gw, gh = torch.meshgrid(h, w)
	return torch.stack([gh, gw], dim=2)

def resolve_grid_shape(N, aspect=1, force_rows=None):
	"""
		Resolves grid shape based on constraints of aspect ratio and dimension forcing.

		Parameters
		----------
		N : int
			Number of cells.
		aspect : float, optional
			Aspect ratio (width/height). Default: 1
		force_rows : int, optional
			Fixed number of rows in the grid with width filled left-to-right. If `None` is ignored. Default: `None`
	"""
	if force_rows is None:
		W = max(round(math.sqrt(N) * aspect), 1)
		H = max(math.ceil(N / W), 1)
	else:
		H = force_rows
		W = max(math.ceil(N / H), 1)
	if N < W:
		W = N
		H = 1
	return W, H

def mask_to_rgb(mask:np.ndarray) -> np.ndarray:
	"""
		Take a segmentation mask and convert it to RGB image.
	"""
	if mask.shape[2] == 1:  # To RGB.
		return mask.repeat(3, axis=2)
	elif mask.shape[2] == 3:  # Keep RGB.
		return mask
	else:
		raise RuntimeError('Mask shape not supported yet (only 1 and 3 channels supported): ' + str(mask.shape))
	# else:
	# 	mask_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
	# 	int_mask = np.argmax(mask, axis=2)
	# 	for c in range(1,mask.shape[2]):
	# 		col = np.array(get_unique_color_from_hsv(c, mask.shape[2]-1)) / 255.
	# 		mask_img[int_mask == c] = col
	# 	return mask_img

def mask_image(img:Union[np.ndarray,torch.Tensor], mask:Union[np.ndarray,torch.Tensor], color:List[float]=[1,0,0], alpha:float=0.5) -> np.ndarray:
	"""
		Takes HWC numpy or CHW torch image and a matching mask.
		Returns their lerp in original format.

		Parameters
		----------
		img : Union[np.ndarray,torch.Tensor]
			Background image to overlay mask on.
		mask : Union[np.ndarray,torch.Tensor]
			Single-channel mask to overlay. Gets multiplied by the color, so it can be grayscale.
		color : List[float]
			RGB color to use for the mask. Default: [1,0,0]
		alpha : float
			Percentage of influence of mask on the image. Default: 0.5
	"""
	is_tensor = isinstance(img, torch.Tensor)
	if is_tensor:
		# CHW to HWC
		if len(img.shape) == 3:
			img = img.permute(1,2,0)
		if len(mask.shape) == 3:
			mask = mask.permute(1,2,0)
		# Use numpy for processing.
		img = img.numpy()
		mask = mask.numpy()

	# Ensure channels dimension.
	if len(img.shape) == 2:
		img = img[:,:,None]
	if len(mask.shape) == 2:
		mask = mask[:,:,None]

	# Ensure RGB image.
	if img.shape[2] == 1:
		img = img.repeat(3, axis=2)

	# Colorize mask to RGB.
	color = np.array(color).reshape(1,1,3)
	mask = mask_to_rgb(mask) * alpha
	img = (1-mask) * img + mask * color

	if is_tensor:  # Tensorify and return to CHW.
		return torch.tensor(img).permute(2,0,1)
	return img

def plt_set_fullscreen():
	backend = str(plt.get_backend()).upper()
	mgr = plt.get_current_fig_manager()
	if backend == 'TKAGG':
		if os.name == 'nt':
			mgr.window.state('zoomed')
		else:
			mgr.resize(*mgr.window.maxsize())
	elif backend == 'WXAGG':
		mgr.frame.Maximize(True)
	elif backend == 'QT5AGG' or backend == 'QT4AGG' or backend == 'QTAGG':
		mgr.window.showMaximized()
	else:
		print('--> Cannot maximize pyplot window, unknown backend: ', backend)

class PlotImageGridContext:
	def __init__(self, num_of_images, margins=0.01, quiet=True, aspect=16/9, colorbar=False, ticks=False, title=None, force_rows=None):
		self.margins = margins
		self.quiet = quiet
		self.aspect = aspect
		self.colorbar = colorbar
		self.ticks = ticks
		self.title = title
		self.N = num_of_images
		self.force_rows = force_rows

	def __enter__(self):
		self.i = 0
		self.W, self.H = resolve_grid_shape(self.N, self.aspect, self.force_rows)
		return self

	def add_image(self, img, name=None, fontsize=16, **kwargs):
		if img is not None:
			if not self.quiet and (isinstance(img, np.ndarray) or isinstance(img, torch.Tensor)):
				if name is None:
					print(f'Image {self.i+1} of shape {img.shape} has range: [{img.min()},{img.max()}]')
				else:
					print(f'{name} of shape {img.shape} has range: [{img.min()},{img.max()}]')
			plt.subplot(self.H, self.W, self.i+1)
			if name is not None:
				plt.title(name, fontdict={'fontsize': fontsize})
			kwargs.setdefault('cmap', 'gray')
			im = plt.imshow(img, **kwargs)
			plt.gca().axes.xaxis.set_visible(self.ticks)
			plt.gca().axes.yaxis.set_visible(self.ticks)
			if self.colorbar:
				cbar = plt.colorbar(im)
				cbar.ax.tick_params(labelsize=fontsize)
		self.i += 1

	def __exit__(self, type, value, trace):
		plt.gcf().subplots_adjust(
			top=1-self.margins,
			bottom=self.margins,
			left=self.margins,
			right=1-self.margins,
			hspace=self.margins,
			wspace=self.margins
		)
		if self.title is not None:
			plt.suptitle(self.title)

def show_images(*imgs, names=None, fullscreen=True, margins=0.01, quiet=True, aspect=16/9, colorbar=False, ticks=False, title=None, force_rows=None, **kwargs):
	with PlotContext(show=True, fullscreen=fullscreen):
		draw_images(*imgs, names=names, margins=margins, quiet=quiet, aspect=aspect, colorbar=colorbar, ticks=ticks, title=title, force_rows=force_rows, **kwargs)

def draw_images(*imgs, names=None, margins=0.01, quiet=True, aspect=16/9, colorbar=False, ticks=False, title=None, force_rows=None, **kwargs):
	with PlotImageGridContext(len(imgs), margins, quiet, aspect, colorbar, ticks, title, force_rows) as pc:
		for i,img in enumerate(imgs):
			if names is not None:
				n = names[i]
			else:
				n = None
			pc.add_image(img, n, **kwargs)

def draw_summarization_curves(x, y, x_label='', y_label='', chosen=None, original_points=None, class_names=None, title=None, grid=True, padding=0.01, markersize=2):
	assert len(x) == len(y), f'Lengths should match, but got: X={len(x)} and Y={len(y)}'
	# Generate class names.
	C = len(x)
	if class_names is None:
		class_names = [f'Class {c+1}' for c in range(C)]
	# Draw plots per-class.
	for c, name in enumerate(class_names):
		x_c, y_c = x[c], y[c]
		plt.step(x_c, y_c, where='post', marker='o', markersize=markersize, label=name)  # Step plot.
		if original_points is not None:
			print(original_points[0][:,c].shape)
			plt.scatter(original_points[0][:,c], original_points[1][:,c], marker='.', s=markersize/4)  # Locations for all thresholds.
		if chosen is not None:
			plt.scatter(chosen[0][c], chosen[1][c], marker='o', c='red')  # Best threshold.
		plt.xlabel(x_label)
		plt.ylabel(y_label)
	plt.legend()
	plt.gca().set_aspect(1)
	plt.grid(grid)
	plt.xlim(-padding,1+padding)
	plt.ylim(-padding,1+padding)
	if title is not None:
		plt.title(title)
