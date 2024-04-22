#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define transformations to apply to the data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

# Download and load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)
testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False)


# Define a simple neural network model
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork,self).__init__()
		self.fc1 = nn.Linear(28*28,128)
		self.fc2 = nn.Linear(128,64)
		self.fc3 = nn.Linear(64,10)
	def forward(self,x):
		x = x.view(-1,28*28) #Flatten the input tensor
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# Instantiate the neural netweok
model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model
for epoch in range(10):	# Train for 5 epochs
	running_loss = 0.0
	for i, data in enumerate(trainloader,0):
		inputs, labels = data
		optimizer.zero_grad()	# Zero the parameter gradients
		outputs = model(inputs)	# Forward pass
		loss = criterion(outputs,labels)	# Compute the loss
		loss.backward()		# Backward pass
		optimizer.step()	# Update weights
		running_loss += loss.item()
		if i%100 == 99:
			print('[%d,%5d] loss: %.3f' % (epoch+1, i+1, running_loss/100))
			running_loss = 0.0
print('Finished Training')

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():	# Disable gradient calculation for evaluation
	for data in testloader:
		inputs,labels = data
		outputs = model(inputs)
		_,predicted = torch.max(outputs,1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy on the test set: %d %%' %(100*correct/total))
