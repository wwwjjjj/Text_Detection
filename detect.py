import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from models.East import EAST
from dataset.custom_dataset import get_rotate_mat
import numpy as np
import lanms
import matplotlib.pyplot as plt
import os
import cv2

from matplotlib import pyplot as plt

def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(path,valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	print(valid_pos)
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y

		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(path,score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score       : score map from models <numpy.ndarray, (1,row,col)>
		geo         : geo map from models <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	score1=(score>score_thresh)
	print(score.shape)
	plt.imshow(score1)
	plt.imshow(geo[0,:,:])

	plt.title(path.split('/')[-1])
	plt.show()
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	#print(path.split('/')[-1],valid_geo,valid_pos)
	polys_restored, index = restore_polys(path,valid_pos, valid_geo, score.shape)
	#print(polys_restored,index)
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]

	temp = np.zeros((score.shape[:2]), np.uint8)
	for box in boxes:
		cv2.fillPoly(temp, [np.array(
			[[int(box[0]), int(box[1])], [int(box[2]), int(box[3])], [int(box[4]), int(box[5])],
			 [int(box[6]), int(box[7])]])], 255)
	cv2.imshow("img", np.array(img))
	cv2.imshow("temp", temp)
	cv2.waitKey(0)
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	#print(boxes)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(path,img, model, device):
	'''detect text regions of img using models
	Input:
		img   : PIL Image
		models : detection models
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	img, ratio_h, ratio_w = resize_img(img)
	plt.imshow(img)
	plt.show()
	score, geo = model(load_pil(img).to(device))

	#print("***")
		#print(score)
	boxes = get_boxes(path,score.squeeze(0).cpu().detach().numpy(), geo.squeeze(0).cpu().detach().numpy())
	plot_boxes(img, boxes)
	return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img_, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img_
	img=np.array(img_)
	#print(img.shape)
	#draw = ImageDraw.Draw(img_)

	temp=np.zeros((img.shape[:2]),np.uint8)
	#print(temp.shape)
	for box in boxes:
		#print(boxes,box)
		#print(np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]],np.int32))
		cv2.fillPoly(temp,[np.array([[int(box[0]), int(box[1])], [int(box[2]), int(box[3])], [int(box[4]), int(box[5])],[int(box[6]), int(box[7])]])],255)
		#draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	cv2.imshow("img",np.array(img))
	cv2.imshow("temp",temp)
	cv2.waitKey(0)

	return img_


def detect_dataset(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		models        : detection models
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')
		boxes = detect(img_file,Image.open(img_file), model, device)
		#print(boxes)
		seq = []
		if boxes is not None:
			seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
		with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
			f.writelines(seq)


if __name__ == '__main__':
	import os
	img_path    ="/home/pei_group/jupyter/Wujingjing/dataset/ICDAR2015/ch4_test_images/"
	model_path  = 'exp/EAST/checkpoint/east_82.pth'
	res_path     = 'res'
	if not os.path.exists(res_path):
		os.mkdir(res_path)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST().to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	files=os.listdir(img_path)
	for file in files:
		#print(file)
		path=os.path.join(img_path,file)
		img = Image.open(path)
		boxes = detect(path,img, model, device)
		#plot_img = plot_boxes(img, boxes)
		#print("ooooo")

		#plot_img.save(os.path.join(res_path,file))


