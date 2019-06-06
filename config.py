import torch


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataloader.py
    batch_size = 8
    num_workers = 2  # ...?
    # TODO
    photo_image_dir = "data/photo/"
    animation_image_dir = "data/animation/"
    edge_smoothed_image_dir = "data/edge_smoothed/"
    test_photo_image_dir = "data/test/"

    # CartoonGAN_train.py
    adam_beta1 = 0.5  # following dcgan
    lr = 0.0002
    num_epochs = 20
    initialization_epochs = 10
    content_loss_weight = 10
    print_every = 100
