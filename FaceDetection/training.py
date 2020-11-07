from src import dataset, loss, network
from src.utils import custom_collate_fn
from torch.utils.data import DataLoader
from torch.optim import SGD
import os
import torch

dataset_path = 'data'
MODEL_PATH = 'data\\models'
image_size = 416
BATCH_SIZE = 1
coord_scale = 5
noobj_scale = 0.5
EPOCHS = 100
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SAVE_MODEL_NAME = '{}_check_point.pth'

_CUDA_FLAG = torch.cuda.is_available()

def train():
    # Dataset/Dataloader
    train_dataset = dataset.FDDBDataset(mode = "train", image_size = image_size, root_path = dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, collate_fn = custom_collate_fn, shuffle = True)
    test_dataset = dataset.FDDBDataset(mode = "test", image_size = image_size, root_path = dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, collate_fn = custom_collate_fn, shuffle = True)

    model = network.Yolo()
    if _CUDA_FLAG : model.cuda()

    criterion = loss.Custom_loss(anchors = model.anchors, batch_size = BATCH_SIZE, coord_scale = coord_scale, noobj_scale= noobj_scale)
    optimizer = SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)

    for cur_epoch in range(EPOCHS):
        # Training
        model.train()
        optimizer.zero_grad()
        train_total_loss = []
        train_coord_loss = []
        train_conf_loss = []
        train_cls_loss = []
        for cur_iter, train_data in enumerate(train_dataloader):
            train_images, train_labels = train_data
            if _CUDA_FLAG : train_images = train_images.cuda()

            train_output = model(train_images)
            total_loss, coord_loss, conf_loss, cls_loss = criterion(train_output, train_labels)
            total_loss.backward()
            optimizer.step()

            # Append all losses for tensorboard
            train_total_loss.append(total_loss.detach())
            train_coord_loss.append(coord_loss.detach())
            train_conf_loss.append(conf_loss.detach())
            train_cls_loss.append(cls_loss.detach())
            print("TRAIN::: Epoch : {}/{} Iteration {}/{} Loss {:.2f} ( Coord {:.6f} Conf {:.6f} Cls {:.6f}".format\
                (cur_epoch, EPOCHS, cur_iter, len(train_dataloader), total_loss, coord_loss, conf_loss, cls_loss))
        
        # Get train losses for one epoch            
        train_total_loss = train_total_loss.sum(0)/len(train_dataloader)
        train_coord_loss = train_coord_loss.sum(0)/len(train_dataloader)
        train_conf_loss = train_conf_loss.sum(0)/len(train_dataloader)
        train_cols_loss = train_cols_loss.sum(0)/len(train_dataloader)
        print("TRAIN::: Epoch : {}/{} Loss {:.6f} (Coord {:.6f} Conf {:.6f} Cls {:.6f})".format\
            (cur_epoch, EPOCHS, train_total_loss, train_coord_loss, train_conf_loss, train_cols_loss))

        # Testing
        model.eval()
        test_total_loss = []
        test_coord_loss = []
        test_conf_loss = []
        test_cls_loss = []
        with torch.no_grad() :
            for cur_iter, test_data in enumerate(test_dataloader):            
                test_images, test_labels = test_data
                if _CUDA_FLAG : test_images = test_images.cuda()

                test_output = model(test_images)
                total_loss, coord_loss, conf_loss, cls_loss = criterion(test_output, test_labels)

                # Append all losses for tensorboard
                test_total_loss.append(total_loss)
                test_coord_loss.append(coord_loss)
                test_conf_loss.append(conf_loss)
                test_cls_loss.append(cls_loss)
                print("TEST::: Epoch : {}/{} Iteration {}/{} Loss {:.6f} (Coord {:.6f} Conf {:.6f} Cls {:.6f})".format\
                    (cur_epoch, EPOCHS, cur_iter, len(test_dataloader), total_loss, coord_loss, conf_loss, cls_loss))

            # Get test losses for one epoch            
            test_total_loss = test_total_loss.sum(0)/len(test_dataloader)
            test_coord_loss = test_coord_loss.sum(0)/len(test_dataloader)
            test_conf_loss = test_conf_loss.sum(0)/len(test_dataloader)
            tEst_cols_loss = test_cls_loss.sum(0)/len(test_dataloader)
            print("TEST::: Epoch : {}/{} Loss {:.6f} (Coord {:.6f} Conf {:.6f} Cls {:.6f})".format\
                (cur_epoch, EPOCHS, test_total_loss, test_coord_loss, test_conf_loss, test_cls_loss))
        
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, SAVE_MODEL_NAME.format(cur_epoch)))

if __name__ == "__main__":
    train()


