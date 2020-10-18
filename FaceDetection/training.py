from src import dataset, loss, network
from src.utils import custom_collate_fn
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch
from src.copy_loss import YoloLoss

class arguments():
    def __init__(self):

        self.dataset_path = 'data/'
        self.mode = 'train'
        self.image_size = 416
        self.batch_size = 10
        self.coord_scale = 5
        self.noobj_scale = 0.5
        self.epochs = 100
        self.save_model_path = "models\\almost_check_point_{}.pth"

def train(args):
    _gpu_flag = torch.cuda.is_available()

    train_dataset = dataset.FDDBDataset(args.dataset_path, args.mode, args.image_size, True)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = custom_collate_fn, shuffle = True) ## Shuffle is False

    val_dataset = dataset.FDDBDataset(args.dataset_path, 'val', args.image_size, False)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, collate_fn = custom_collate_fn, shuffle = True)     ## Shuffle is False

    my_model = network.Yolo()
    if _gpu_flag : my_model.cuda()
    #criterion = YoloLoss(1, my_model.anchors)
    criterion = loss.Custom_loss(batch_size = args.batch_size, anchors = my_model.anchors, coord_scale = args.coord_scale, noobj_scale= args.noobj_scale)
    optimizer = SGD(my_model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 0.0005)

    for epochs in range(args.epochs):

        my_model.train()
        for i, train_data in enumerate(train_dataloader):
            train_images, train_labels = train_data
            if _gpu_flag : 
                train_images = torch.tensor(train_images.cuda()).clone().detach().requires_grad_(True)
            else : 
                train_images = torch.tensor(train_images).clone().detach().requires_grad_(True)
            optimizer.zero_grad()

            output = my_model(train_images)
            total_loss, coord_loss, conf_loss, cls_loss = criterion(output, train_labels)
            
            
            total_loss.backward()
            optimizer.step()

            print("Epoch : {}/{} Iteration {}/{} Loss {:.2f} ( Coord {:.6f} Conf {:.6f} Cls {:.6f}".format(epochs, args.epochs, i+1, len(train_dataloader), total_loss, coord_loss, conf_loss, cls_loss))
        
        my_model.eval()

        total_val_loss = []
        coord_val_loss = []
        conf_val_loss = []
        cls_val_loss = []

        for i, val_data in enumerate(val_dataloader):            
            val_images, val_labels = val_data
            with torch.no_grad():
                if _gpu_flag :
                    #val_images = torch.tensor(train_images.cuda()).clone().detach().requires_grad_(True)
                    val_images = val_images.clone().detach().requires_grad_(False).cuda()
                else :
                    val_images = torch.tensor(val_images).clone().detach().requires_grad_(False)
                val_output = my_model(val_images)
                total_loss, coord_loss, conf_loss, cls_loss = criterion(val_output, val_labels)
                total_val_loss.append(total_loss)
                coord_val_loss.append(coord_loss)
                conf_val_loss.append(conf_loss)
                cls_val_loss.append(cls_loss)
        total_val_loss = torch.tensor(total_val_loss).sum(0)/val_dataloader.__len__()
        coord_val_loss = torch.tensor(coord_val_loss).sum(0)/val_dataloader.__len__()
        conf_val_loss = torch.tensor(conf_val_loss).sum(0)/val_dataloader.__len__()
        cls_val_loss = torch.tensor(cls_val_loss).sum(0)/val_dataloader.__len__()
        print("VAL ::: Epoch : {}/{} Iteration {}/{} Loss {:.2f} ( Coord {:.6f} Conf {:.6f} Cls {:.6f}".format\
            (epochs, args.epochs, i+1, len(val_dataloader), total_val_loss, coord_val_loss, conf_val_loss, cls_val_loss))
        
        if epochs % 30 == 0 :
            torch.save(my_model.state_dict(), args.save_model_path.format(epochs))

if __name__ == "__main__":
    args = arguments()
    train(args)


