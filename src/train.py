import argparse
from train_model import train_model
from load_save_model import save_model

ap = argparse.ArgumentParser()
ap.add_argument('data_dir', action="store", default="/home/workspace/aipnd-project/flowers")
ap.add_argument('--save_dir', dest='save_dir', action="store", default="./checkpoint_final_v1.pth", type=str)
ap.add_argument('--arch', dest='arch', action="store", default="vgg19")
ap.add_argument('--learning_rate', dest='learning_rate', action="store", default=0.001, type=float)
ap.add_argument('--hidden_units', dest="hidden_units", action="store", default=100, type=int)
ap.add_argument('--epochs',  dest="epochs", action="store", default=3, type=int)
ap.add_argument('--gpu', dest="gpu", action="store", default='gpu')
ap.add_argument('--dropout', dest="dropout", action="store", default=0.5, type=float)

arg_data = ap.parse_args()
data_dir = arg_data.data_dir
save_dir = arg_data.save_dir
arch = arg_data.arch
learning_rate = arg_data.learning_rate
hidden_units = arg_data.hidden_units
epochs = arg_data.epochs
device = arg_data.gpu
dropout = arg_data.dropout


train_model = train_model(data_dir,save_dir,arch,learning_rate,hidden_units,epochs,device,dropout)
train_model.train_execution()



