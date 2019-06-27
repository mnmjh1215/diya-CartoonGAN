# train and test CycleGAN

from CycleGAN_model import Generator, Discriminator
from CartoonGAN_model import Generator as CartoonGAN_Generator, Discriminator as CartoonGAN_Discriminator
from CycleGAN_train import CycleGANTrainer
from config import CycleGANConfig as Config
from dataloader import load_image_dataloader

import torch
import matplotlib.pyplot as plt
import argparse
import torchvision.utils as tvutils
import os
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test',
                        action='store_true',
                        help='Use this argument to test generator and compute FID score')

    parser.add_argument('--model_path',
                        help='Path to saved model')

    parser.add_argument('--test_image_path',
                        default=Config.test_photo_image_dir,
                        help='Path to test photo images')

    parser.add_argument('--initialization_epochs',
                        type=int,
                        default=Config.initialization_epochs,
                        help='Number of epochs for initialization phase')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=Config.num_epochs,
                        help='Number of training epochs')

    parser.add_argument('--use_edge_smoothed_images',
                        action='store_true',
                        help='Use this argument to use edge smoothed images in training')

    parser.add_argument('--use_cartoongan_architecture',
                        action='store_true',
                        help="Use this argument to use CartoonGAN's generator and discriminator architecture")

    parser.add_argument('--test_animation_to_photo',
                        action='store_true',
                        help='Use this argument to test animation to photo transfer')

    args = parser.parse_args()

    return args


def load_model(G, F, D_x, D_y, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state_dict'])
    F.load_state_dict(checkpoint['F_state_dict'])
    D_y.load_state_dict(checkpoint['D_y_state_dict'])
    D_x.load_state_dict(checkpoint['D_x_state_dict'])


def load_generators(G, F, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state_dict'])
    F.load_state_dict(checkpoint['F_state_dict'])


def generate_and_save_images(generator, test_image_loader, save_path):
    # for each image in test_image_loader, generate image and save
    generator.eval()
    torch_to_image = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # [-1, 1] to [0, 1]
        transforms.ToPILImage()
    ])

    image_ix = 0
    for test_images, _ in test_image_loader:
        test_images = test_images.to(Config.device)
        generated_images = generator(test_images).detach().cpu()

        for i in range(len(generated_images)):
            image = generated_images[i]
            image = torch_to_image(image)
            image.save(os.path.join(save_path, '{0}.jpg'.format(image_ix)))
            image_ix += 1


def main():

    args = get_args()

    device = Config.device
    print("PyTorch running with device {0}".format(device))

    if args.test:
        assert args.model_path, 'model_path must be provided for testing'
        print('Testing...')

        print("Creating models...")
        if not args.use_cartoongan_architecture:
            G = Generator().to(device)
            F = Generator().to(device)
            G.eval()
            F.eval()

        else:
            G = CartoonGAN_Generator().to(device)
            F = CartoonGAN_Generator().to(device)

        print('Loading models...')
        load_generators(G, F, args.model_path)
        # Do testing stuff
        # ex. generate image, compute fid score

        if not args.test_animation_to_photo:
            generator = G
        else:
            generator = F

        test_images = load_image_dataloader(root_dir=args.test_image_path, batch_size=Config.batch_size * 2, shuffle=False)

        image_batch, _ = next(iter(test_images))
        image_batch = image_batch.to(Config.device)

        new_images = G(image_batch).detach().cpu()

        tvutils.save_image(image_batch, 'test_images.jpg', nrow=4, padding=2, normalize=True, range=(-1, 1))
        tvutils.save_image(new_images, 'generated_images.jpg', nrow=4, padding=2, normalize=True, range=(-1, 1))

        if not os.path.isdir('generated_images'):
            os.mkdir('generated_images')
        if not os.path.isdir('generated_images/CartoonGAN'):
            os.mkdir('generated_images/CartoonGAN/')
        generate_and_save_images(generator, test_images, 'generated_images/CartoonGAN/')

    else:
        print("Training...")

        print("Loading 2 generators and 2 discriminators")
        if not args.use_cartoongan_architecture:
            G = Generator().to(device)
            F = Generator().to(device)
            D_x = Discriminator().to(device)
            D_y = Discriminator().to(device)
        else:
            G = CartoonGAN_Generator().to(device)
            F = CartoonGAN_Generator().to(device)
            D_x = CartoonGAN_Discriminator().to(device)
            D_y = CartoonGAN_Discriminator().to(device)

        # load dataloaders
        photo_images = load_image_dataloader(root_dir=Config.photo_image_dir, batch_size=Config.batch_size)
        animation_images = load_image_dataloader(root_dir=Config.animation_image_dir, batch_size=Config.batch_size)
        edge_smoothed_images = load_image_dataloader(root_dir=Config.edge_smoothed_image_dir, batch_size=Config.batch_size)

        print("Loading Trainer...")
        trainer = CycleGANTrainer(G, F, D_x, D_y, photo_images, animation_images,
                                  edge_smoothed_images, use_edge_smoothed=args.use_edge_smoothed_images,
                                  use_initialization=(args.initialization_epochs > 0))
        if args.model_path:
            trainer.load_checkpoint(args.model_path)

        print('Start Training...')
        loss_D_x_hist, loss_D_y_hist, loss_G_GAN_hist, loss_F_GAN_hist, \
        loss_cycle_hist, loss_identity_hist = trainer.train(num_epochs=args.num_epochs,
                                                            initialization_epochs=args.initialization_epochs)

        plt.plot(loss_D_x_hist, label='D_x loss')
        plt.plot(loss_D_y_hist, label='D_y loss')
        plt.plot(loss_G_GAN_hist, label='G loss')
        plt.plot(loss_F_GAN_hist, label='F loss')
        plt.plot(loss_cycle_hist, label='cycle loss')
        plt.legend()
        plt.savefig('CycleGAN_train_history.jpg')
        plt.show()


if __name__ == '__main__':
    main()



