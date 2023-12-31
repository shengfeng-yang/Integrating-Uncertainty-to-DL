from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from PIL import Image
import numpy as np
##########################################################################################
# Average test loss: 0.0639, MAE: 4.1275 
# Average train loss: 0.0604, MAE: 3.8158 
##########################################################################################
class MechMNISTDataset(Dataset):
	""" mechanical MNIST data set"""
	def __init__(self, args, train='Train',transform=None, target_transform=None):
		self.train = train
		self.args = args
		character = num_to_string(self.args.predict_character)
		if self.train == 'Train':
			self.data = np.load('./data/image_bi.npy')
			self.data = self.data.reshape((self.data.shape[0],28,28)).astype(float)
			self.targets = np.load('./data/ten_' + character + '.npy').reshape(-1,1).astype(float)
		elif self.train == 'Val':
			self.data = np.load('./data/image_bi_test.npy')
			self.data = self.data.reshape((self.data.shape[0],28,28)).astype(float)
			self.targets = np.load('./data/ten_' + character + '_test.npy').reshape(-1,1).astype(float)
		else:
			self.data = np.load('./data/image_bi_val.npy')
			self.data = self.data.reshape((self.data.shape[0],28,28)).astype(float)
			self.targets = np.load('./data/ten_' + character + '_test_test.npy').reshape(-1,1).astype(float)
		self.transform = transform
		self.target_transform = target_transform
		
	def __len__(self):
		return self.data.shape[0]
	
	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		img = self.data[idx,:,:]
		lab = self.targets[idx]
		
		#img = Image.fromarray(img.numpy(), mode='L')

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			lab = self.target_transform(lab)
		
		sample = (img,lab)
		
		return sample 

##########################################################################################
# train nn
##########################################################################################
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(784, 1568*4)
		self.fc2 = nn.Linear(1568*4,1568)
		self.fc3 = nn.Linear(1568,784)
		self.fc4 = nn.Linear(784, 1)

		self.flattens = nn.Flatten()
		self.fc1s = nn.Linear(1568, 128)
		self.fc5s = nn.Linear(128,16)
		self.fc4s = nn.Linear(16, 1)

	def forward(self, x):
		x0 = x.detach()
		x = self.flatten(x).float()
		x = F.leaky_relu(self.fc1(x))
		x1 = F.dropout(x,0.40)
		x = F.leaky_relu(self.fc2(x1))
		x2 = F.dropout(x,0.40)
		x3 = F.leaky_relu(self.fc3(x2))
		x = F.dropout(x3,0.40)
		x = F.leaky_relu(self.fc4(x))

		xs = self.flattens(x2).float()
		xs = F.leaky_relu(self.fc1s(xs))
		xs = F.dropout(xs,0.40)
		xs = F.leaky_relu(self.fc5s(xs))
		xs = F.dropout(xs,0.40)
		xs = F.leaky_relu(self.fc4s(xs))
		logstd = xs
		return x,logstd


def  num_to_string(num):
	numbers = {
		1:'yeild_stress',
		2:'yeild_strain',
		3:'UTS',
		4:'UTS_strain',
	}
	return numbers.get(num, None)


def prob_loss(mean,log_std,target):
	# loss = -torch.mean(-(torch.div(torch.pow(targets-output[0],2),2*torch.pow(torch.exp(output[1]),2))) - 0.5*np.log(2*np.pi) - 0.5*2.*output[1])
	loss = -torch.mean(-(torch.div(torch.pow(target-mean,2),2*torch.pow(torch.exp(log_std),2))) - 0.5*np.log(2*np.pi) - 0.5*2.*log_std)
	return loss


def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	loss_fcn = nn.MSELoss()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		mean,logstd = model(data)
		# print(logstd)
		loss = prob_loss(mean,logstd,target)
		# loss = loss_fcn(output,target.float()) #REGcha
		loss.backward()
		torch.nn.utils.clip_grad_norm(model.parameters(),5)
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader,is_train):
	model.eval()
	loss_fcn = nn.MSELoss()
	loss_fcn_l1 = nn.L1Loss()
	test_loss = 0
	correct = 0
	MAE = 0 
	counter = 0
	percent_error = 0
	predict_list = []
	std_list = []
	target_list = []
	prob_list = 0
	count = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output,logstd = model(data)
			for i in range(50):
				predict_list.append(output.cpu()[i])
				std_list.append(logstd[i].cpu().item())
				target_list.append(target.cpu()[i])
				if torch.abs(output.cpu()[i]-target.cpu()[i])<2*torch.exp(logstd[i].cpu()):
					count += 1
			# for i in range(1):
				# print([output[i].item(),target[i].item()])
			test_loss += loss_fcn(output, target.float()) # sum up batch loss
			prob_list += prob_loss(output,logstd,target)
			pred = output
			counter += 1 
			MAE += loss_fcn(output, target.float()).item()
			percent_error += np.abs(target.detach().cpu().numpy()[0][0] - output.detach().cpu().numpy()[0][0])/np.abs(target.detach().cpu().numpy()[0][0])

	test_loss /= counter

	prob_list /= counter

	# MAE = MAE/counter
	MAE = test_loss
	percent_error = percent_error/counter

	print('\nTest set: Average loss: {:.4f}, MAE: {:.4f}, percentage_error: {:.4f} \n'.format(test_loss, MAE,percent_error))
	print (count)
	character = num_to_string(args.predict_character)
	if is_train == 'Test':
		np.savetxt('fnn_predict_8000_prob_'+ character + '_test.csv',np.array(predict_list),delimiter=',')
		np.savetxt('fnn_MD_8000_prob_'+ character + '_test.csv',np.array(target_list),delimiter=',')
		np.savetxt('fnn_std_8000_prob_'+ character + '_test.csv',np.array(std_list),delimiter=',')
	elif is_train == 'Val':
		np.savetxt('fnn_predict_8000_prob_'+ character + '_val.csv',np.array(predict_list),delimiter=',')
		np.savetxt('fnn_MD_8000_prob_'+ character + '_val.csv',np.array(target_list),delimiter=',')
		np.savetxt('fnn_std_8000_prob_'+ character + '_val.csv',np.array(std_list),delimiter=',')
	elif is_train == 'Train':
		np.savetxt('fnn_predict_8000_prob_'+ character + '_train.csv',np.array(predict_list),delimiter=',')
		np.savetxt('fnn_MD_8000_prob_'+ character + '_train.csv',np.array(target_list),delimiter=',')
		np.savetxt('fnn_std_8000_prob_'+ character + '_train.csv',np.array(std_list),delimiter=',')
	# print (count)
	return test_loss,count,np.mean(np.array(std_list)), prob_list

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	#batch size = 64
	parser.add_argument('--batch-size', type=int, default=50, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=300, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', #ORIG 0.01
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--predict_character', type = int, default=1,
						help='1 = yeild stress, 2 = yeild strain, 3 = ultimate stress, 4 = ultimate strain') 
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	
	parser.add_argument('--save-model', action='store_true', default=True,
						help='For Saving the current Model')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu") 

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	train_loader = DataLoader( MechMNISTDataset(args = args, train=True,  transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((33.3523,), (85.9794,))])), batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader  = DataLoader( MechMNISTDataset(args = args, train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((33.3523,), (85.9794,))])), batch_size=args.test_batch_size, shuffle=False, **kwargs)
	val_loader  = DataLoader( MechMNISTDataset(args = args, train='Val', transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((33.3523,), (85.9794,))])), batch_size=args.test_batch_size, shuffle=False, **kwargs)


	model = Net().to(device)
	# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
	# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',threshold = 0.01)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,gamma=0.2)

	train_loss_list = np.zeros(args.epochs)
	val_loss_list = np.zeros(args.epochs)
	train_count_list = np.zeros(args.epochs)
	val_count_list = np.zeros(args.epochs)
	train_std_list = np.zeros(args.epochs)
	val_std_list = np.zeros(args.epochs)
	train_prob_list = np.zeros(args.epochs)
	val_prob_list = np.zeros(args.epochs)


	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		test_loss, count,std,prob_list = test(args, model, device, val_loader,'Val') # test error
		val_loss_list[epoch-1] = test_loss
		val_count_list[epoch-1] = count
		val_std_list[epoch-1] = std
		val_prob_list[epoch-1] = prob_list
		test_loss, count,std,prob_list = test(args, model, device, train_loader,'Train') # training error 
		train_loss_list[epoch-1] = test_loss
		train_count_list[epoch-1] = count
		train_std_list[epoch-1] = std
		train_prob_list[epoch-1] = prob_list
		scheduler.step()
	test_loss, count,std,prob_list = test(args, model, device, test_loader,'Test')
	test_loss, count,std,prob_list = test(args, model, device, val_loader,'Val')
	test_loss, count,std,prob_list = test(args, model, device, train_loader,'Train')
	character = num_to_string(args.predict_character)
	x = np.arange(1,args.epochs+1)
	plt.rcParams.update({'font.size': 14,'font.weight': 'bold'})
	plt.plot(x,(train_loss_list))
	plt.plot(x,(val_loss_list))
	# plt.title('Loss vs. Epochs')
	plt.xlabel('Epochs',fontsize=14,fontweight='bold')
	plt.ylabel('Loss (MSE)',fontsize=14,fontweight='bold')
	# plt.ylim(0,1)
	plt.legend(['train','val'])
	plt.tight_layout()
	plt.savefig(character + '_8000_porb_mse_loss_epoch_fnn.jpg',dpi=600)
	plt.show()
	plt.close() 

	plt.rcParams.update({'font.size': 14,'font.weight': 'bold'})
	plt.plot(x,(train_prob_list))
	plt.plot(x,(val_prob_list))
	# plt.title('Loss vs. Epochs')
	plt.xlabel('Epochs',fontsize=14,fontweight='bold')
	plt.ylabel('Loss (Probability)',fontsize=14,fontweight='bold')
	# plt.ylim(0,1)
	plt.legend(['train','val'])
	plt.tight_layout()
	plt.savefig(character + '_8000_prob_loss_epoch_fnn.jpg',dpi=600)
	plt.show()
	plt.close() 


	x = np.arange(1,args.epochs+1)
	plt.rcParams.update({'font.size': 14,'font.weight': 'bold'})
	plt.plot(x,(train_count_list))
	plt.plot(x,(val_count_list))
	# plt.title('Loss vs. Epochs')
	plt.xlabel('Epochs',fontsize=14,fontweight='bold')
	plt.ylabel('Accurate',fontsize=14,fontweight='bold')
	plt.legend(['train','val'])
	plt.tight_layout()
	plt.savefig(character + '_8000_prob_count_epoch_fnn.jpg',dpi=600)
	plt.show()
	plt.close() 

	plt.rcParams.update({'font.size': 14,'font.weight': 'bold'})
	plt.plot(x,np.exp(train_std_list)**2)
	plt.plot(x,np.exp(val_std_list)**2)
	plt.plot(x,train_loss_list)
	plt.plot(x,val_loss_list)
	# plt.title('Loss vs. Epochs')
	plt.xlabel('Epochs',fontsize=14,fontweight='bold')
	plt.ylabel('mean std',fontsize=14,fontweight='bold')
	plt.legend(['train','val','train-mse','val-mse'])
	plt.tight_layout()
	plt.savefig(character + '_8000_prob_std_epoch_fnn.jpg',dpi=600)
	plt.show()
	plt.close()

	plt.rcParams.update({'font.size': 14,'font.weight': 'bold'})
	# plt.plot(x,(np.exp(train_std_list))**2)
	plt.plot(x,(np.exp(val_std_list))**2)
	# plt.plot(x,train_loss_list)
	plt.plot(x,val_loss_list)
	# plt.title('Loss vs. Epochs')
	plt.xlabel('Epochs',fontsize=14,fontweight='bold')
	plt.ylabel('Uncertainty',fontsize=14,fontweight='bold')
	plt.legend(['val Uncertainty','val MSE'])
	plt.tight_layout()
	plt.savefig(character + '_8000_prob_uncertainty_epoch_fnn.jpg',dpi=600)
	plt.show()
	plt.close()
	np.savetxt('fnn_prob_std_list_'+ character + '_val.csv',np.array(test_std_list),delimiter=',')
	np.savetxt('fnn_prob_mse_list_'+ character + '_val.csv',np.array(test_loss_list),delimiter=',') 

	if (args.save_model):
		torch.save(model.state_dict(),"MECHmnist_fnn_prob.pt")

if __name__ == '__main__':
	main()