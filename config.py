import torch
number_of_epochs = 10
batch_size = 32
resize_x = 64
resize_y = 64
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']





