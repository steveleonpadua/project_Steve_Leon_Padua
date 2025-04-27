from model import CNN_Model as TheModel
from train import fit as the_trainer
from predict import classify_image as the_predictor
from dataset import Image_Folder as TheDataset
from dataset import CustomDataLoader as the_dataloader

from config import batch_size as the_batch_size
from config import number_of_epochs as total_epochs