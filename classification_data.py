import glob
import cv2
import numpy as np
import os

#ler o diretorio(caminho) das imagens
def readpath(filepath):
	filenames = []
	i = 0
	for filename in glob.glob(filepath):
		if ('segmentation' not in filename) and filename.find('.jpg') != -1 and i < 3500:
			filenames.append(filename)
			i += 1
		else:
			os.remove(filename)
	return filenames

#lendo as imagens
def read_files(imgs):
	readed_imgs = []
	for img in imgs:
		readed_imgs.append(cv2.imread(img))
	return readed_imgs

#alterando tamanho das imagens
def resize_img(imgs,size):
	new_imgs = []
	for img in imgs:
		new_imgs.append(cv2.resize(img,size))

	return new_imgs

def started_values(x_values: list,y_values: list):
	x_values = np.concatenate(x_values,axis=0)
	y_values = np.array(y_values)
	y_values = y_values.reshape(-1)
	x_values = x_values.reshape(len(y_values),-1)

	return x_values,y_values

def result(test):
	if test == 1 or test == 2 or test == 3:
		return 'CNH'
	elif test == 4 or test == 5:
		return 'CPF'
	elif test == 7 or test == 6 or test == 8:
		return 'RG'
	else: return None
