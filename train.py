import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset.custom_dataset import custom_dataset
from models.East import EAST
from models.Hourglass import get_large_hourglass_net
from loss.EAST_Loss import EAST_Loss
from loss.Mutli_Task_Loss import CtdetLoss
import os
import time
import argparse
import yaml
from tensorboardX import SummaryWriter
from tools.Progbar import Progbar
import matplotlib.pyplot as plt
from dataset.total_text import TotalText
from tools.augmentation import Augmentation,BaseTransform,Resize
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def display(id,image,gt_map,pred_map):
    plt.imshow(image.cpu().detach().numpy()[0][0])
    plt.title("{}".format(id))
    plt.show()
    plt.imshow((gt_map['hm']).cpu().detach().numpy()[0][0] * gt_map['tr'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((pred_map['hm']).cpu().detach().numpy()[0][0]*pred_map['tr'].cpu().detach().numpy()[0])
    plt.show()



    plt.imshow((pred_map['hm_b']).cpu().detach().numpy()[0][0])
    plt.show()


    plt.imshow((pred_map['hm_l']).cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((pred_map['hm_r']).cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['tr']).cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['o_tr']).cpu().detach().numpy()[0][0])
    plt.show()

    #print(pred_map['tr'].cpu().detach().numpy()[0].shape)
    plt.imshow(pred_map['tr'].cpu().detach().numpy()[0])
    plt.show()
    '''

    plt.imshow((pred_map['hm_b']).cpu().detach().numpy()[0][0])
    plt.show()'''





    '''plt.imshow((gt_map['hm_t']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['hm_b']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['hm_l']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['hm_r']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()
'''

def valid(config,writer,model,valid_loader,epoch):
	model.eval()
	progbar_val = Progbar(
		len(valid_loader), stateful_metrics=["epoch", "config", "lr"]
	)
	valid_loss = 0.0

	nums = 0
	i = 0
	for i, ret in enumerate(valid_loader):
		image = ret['input'].cuda()
		gt_map = {'hm': ret['hm'].cuda(),  # 'tr_m':ret['tr_m'].cuda(),
				  'o_tr':ret['o_tr'].cuda(),
				  'hm_t': ret['hm_t'].cuda(), 'hm_b': ret['hm_b'].cuda(),
				  'hm_l': ret['hm_l'].cuda(), 'hm_r': ret['hm_r'].cuda(),
				  'trm': ret['trm'].cuda(), 'tr': ret['tr'].cuda(),
				  # 'dense_wh': ret['dense_wh'].cuda( ), 'dense_wh_mask': ret['dense_wh_mask'].cuda( ),
				  'center_points': ret['center_points'].cuda(),
				  'off_mask': ret['off_mask'].cuda(), 'offsets': ret['offsets'].cuda()}
		# print(gt_map['hm_t'].dtype,gt_map['hm_t'].shape)

		output = model(image)

		loss, loss_stas, pred_map = criterion(output, gt_map)

		valid_loss += loss.item()

		writer.add_scalars(config['modelname'],
						   {"val_loss": loss.item(),
							"val_hm_loss": loss_stas['hm_loss'],
							"val_geo_loss": loss_stas['geo_loss'],
							"val_tr_loss": loss_stas['tr_loss'],
							"val_off_loss": loss_stas['off_loss']
							}, i)

		progbar_val.add(1, values=[("epoch", epoch),
								   ("val_loss", loss.item()),
								   ("val_hm_loss", loss_stas['hm_loss'].item()),
								   ("val_tr_loss", loss_stas['tr_loss'].item()),
								   ("val_geo_loss", loss_stas['geo_loss'].item()),
								   ("val_off_loss", loss_stas['off_loss'].item())
								   ])
	return valid_loss

def train(config, model,train_loader,valid_loader,optimizer,scheduler):
	writer = SummaryWriter(logdir=config['path']['writer_path'])
	#scheduler.step()
	best_valid_loss=937.324348449707
	for epoch in range(config['train']['max_epoch']):

		progbar_train = Progbar(len(train_loader), stateful_metrics=["epoch", "config", "lr"])
		model.train()

		epoch_loss = 0
		num=0
		for i, gt_maps in enumerate(train_loader):

			if config['modelname']=='EAST_res18':
				img=gt_maps['img']
				gt_score=gt_maps['score_map']
				gt_geo=gt_maps['geo_map']
				ignored_map=gt_maps['ignored_map']
				img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
				pred_score, pred_geo = model(img)
				loss,loss_stas = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			elif config['modelname']=='Hourglass':
				image = gt_maps['input'].cuda()
				gt_map = {'hm': gt_maps['hm'].cuda(),  # 'tr_m':ret['tr_m'].cuda(),
						  'o_tr': gt_maps['o_tr'].cuda(),
						  'hm_t': gt_maps['hm_t'].cuda(), 'hm_b': gt_maps['hm_b'].cuda(),
						  'hm_l': gt_maps['hm_l'].cuda(), 'hm_r': gt_maps['hm_r'].cuda(),
						  'trm': gt_maps['trm'].cuda(), 'tr': gt_maps['tr'].cuda(),
						  'center_points': gt_maps['center_points'].cuda(),
						  'off_mask': gt_maps['off_mask'].cuda(), 'offsets': gt_maps['offsets'].cuda()}

				output = model(image)
				loss, loss_stas, pred_map = criterion(output, gt_map)
				if i == 0:
					display('epoch {}'.format(epoch), image, gt_map, pred_map)


			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()


			if num%5==0:
				writer.add_scalars(config['modelname'],
							   {"loss": loss.item(),
								"hm_loss": loss_stas['hm_loss'],
								"geo_loss": loss_stas['geo_loss'],
								"tr_loss": loss_stas['tr_loss'],
								"off_loss": loss_stas['off_loss']
								}, num)
				progbar_train.add(min(5,156-num), values=[("epoch", epoch),
										 ("loss", loss.item()),
										 ("hm_loss", loss_stas['hm_loss'].item()),
										 ("geo_loss", loss_stas['geo_loss'].item()),
										 ("tr_loss", loss_stas['tr_loss'].item()),
										 ("off_loss", loss_stas['off_loss'].item())
										 ])#,
			num+=1
		print('EPOCH<', epoch, '>: train loss:', epoch_loss/ num if num > 0 else epoch_loss
			  )
		#valid_loss=valid(config,writer,model,valid_loader,epoch)
		scheduler.step()
		if config['train']['step_save'] == True and (epoch + 1) % config['train']['save_step'] == 0:

			state_dict = model.module.state_dict() if config['train']['data_parallel'] else model.state_dict()
			torch.save(state_dict, os.path.join(config['path']['model_dir'],
												'{}_epoch_{}_{}.pth'.format(config['modelname'], epoch+42 ,epoch_loss/ num)))

			print("{} save done!".format(epoch))
			'''if best_valid_loss > valid_loss:
				best_valid_loss=valid_loss
				torch.save(state_dict, os.path.join(config['path']['model_dir'],
													'{}_shrink_best.pth'.format(config['modelname'])))
				print("{} best updated!".format(epoch))'''


if __name__ == '__main__':
	# ---------------- ARGS AND CONFIGS ----------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="Hourglass_52")
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
	parser.add_argument("--gpu_id", default=0)
	parser.add_argument('--gpu_num', type=int, default=1)
	opt = parser.parse_args()

	# print args in train.sh
	print("--- TRAINING ARGS ---")
	print(opt)


	if not os.path.exists("configs/%s.yaml" % opt.config):
		print("*** configs/%s.yaml not found. ***" % opt.config)
		exit()

	# read yaml configs
	f = open("configs/%s.yaml" % opt.config, "r", encoding="utf-8")
	config = yaml.load(f.read(), Loader=yaml.FullLoader)
	print("--- CONFIG ---")
	print(config)

	model_savedir = os.path.join(config["path"]["model_dir"], config["modelname"])
	print(model_savedir)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	os.makedirs(model_savedir, exist_ok=True)
	if config['modelname']=='EAST_res18':
		model=EAST()
		if config['train']['resume']:
			model.load_state_dict(torch.load(config['path']['resume_path']))
		model.to(device)
		'''data_parallel = False'''
		file_num = len(os.listdir(config['path']['train_data_path']))
		trainset = custom_dataset(
			config['path']['train_data_path'],
			config['path']['train_gt_path'])
		train_loader = data.DataLoader(trainset,
									   batch_size=config['train']['batchsize'], \
									   shuffle=True, num_workers=opt.workers, drop_last=True)
		# if torch.cuda.device_count() > 1:
		# model = nn.DataParallel(model)
		# data_parallel = True
		optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
		epoch_iter=int(file_num/2)
		scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter // 2], gamma=0.1)
		criterion = EAST_Loss()
	elif config['modelname']=='Hourglass':
		data_train = TotalText(
			config,
			data_root=config['path']['train_data_path'],
			# 'D:/python_Project/data/totaltext',#'/home/pei_group/jupyter/Wujingjing/data/totaltext/',
			ignore_list=None,
			is_training=True,
			transform=Augmentation(size=512, mean=config['train']['means'],
									std=config['train']['stds']),
			map_transform=Resize(size=config['train']['input_size']// config['train']['downsample']) if config['train']['downsample']> 0 else None,#BaseTransform(size=config['train']['input_size']// config['train']['downsample'], mean=config['train']['means'],
							#			std=config['train']['stds']) if config['train']['downsample']> 0 else None,
			map_size=config['train']['input_size'] if config['train']['downsample'] == 0 else (config['train']['input_size']  // config['train']['downsample'])
		)
		train_loader = data.DataLoader(
			data_train,
			batch_size=config['train']['batchsize'],
			pin_memory=True,
			shuffle=True,
			drop_last=True)
		data_valid = TotalText(
			config,
			data_root=config['path']['test_data_path'],
			# '/home/pei_group/jupyter/Wujingjing/data/totaltext/',#'D:/python_Project/data/totaltext',#'/home/pei_group/jupyter/Wujingjing/data/totaltext/',
			ignore_list=None,
			is_training=False,
			transform=BaseTransform(size=512, mean=config['train']['means'],
									std=config['train']['stds']),
			map_transform=Resize(size=config['train']['input_size'] // config['train']['downsample']) if
			config['train']['downsample'] > 0 else None,
			#map_transform=BaseTransform(size=config['train']['input_size']// config['train']['downsample'], mean=config['train']['means'],
			#							std=config['train']['stds']) if config['train']['downsample']> 0 else None,
			map_size=config['train']['input_size'] if config['train']['downsample'] == 0 else config['train']['input_size']  // config['train']['downsample']
		)

		valid_loader = data.DataLoader(
			data_valid,
			batch_size=config['train']['batchsize'],
			pin_memory=True,
			shuffle=False,
			drop_last=True)
		modelfile=None
		if config['train']['resume']:
			modelfile=torch.load(config['path']['resume_path'])

		model=get_large_hourglass_net(18, config['heads'], 64,modelfile).cuda()
		criterion=CtdetLoss(config)
		optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
		epoch_iter = int(len(train_loader) / config['train']['batchsize'])
		#scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.94)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)

		train(config,model,train_loader,valid_loader,optimizer,scheduler)
