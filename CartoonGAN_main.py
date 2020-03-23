# train and test CartoonGAN

import CartoonGAN_model as models
import CartoonGAN_model_modified as modified_models
from CartoonGAN_train import CartoonGANTrainer
from config import CartoonGANConfig as Config
from dataloader import load_image_dataloader

import torch
import argparse
import torchvision.utils as tvutils
import os
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test',
                        action='store_true',
                        help='Use this argument to test generator')

    parser.add_argument('--model_path',
                        help='Path to saved model')

    parser.add_argument('--model_save_path',
                        default='checkpoints/CartoonGAN/',
                        help='Path to save trained model')

    parser.add_argument("--photo_image_dir",
                        default=Config.photo_image_dir,
                        help="Path to photo images")
    
    parser.add_argument("--animation_image_dir",
                        default=Config.animation_image_dir,
                        help="Path to animation images")
    
    parser.add_argument("--edge_smoothed_image_dir",
                        default=Config.edge_smoothed_image_dir,
                        help="Path to edge smoothed animation images")
    
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
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=Config.batch_size)

    parser.add_argument('--use_modified_model',
                        action='store_true',
                        help="Use this argument to use modified model")

    args = parser.parse_args()

    return args


def load_model(generator, discriminator, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])


def load_generator(generator, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=Config.device)
    generator.load_state_dict(checkpoint['generator_state_dict'])


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

    print("Creating models...")
    if args.use_modified_model:
        Generator = modified_models.Generator
        Discriminator = modified_models.Discriminator
        FeatureExtractor = modified_models.FeatureExtractor
    else:
        Generator = models.Generator
        Discriminator = models.Discriminator
        FeatureExtractor = models.FeatureExtractor

    generator = Generator().to(device)

    if args.test:
        assert args.model_path, 'model_path must be provided for testing'
        print('Testing...')
        generator.eval()

        print('Loading models...')
        load_generator(generator, args.model_path)
        
        test_images = load_image_dataloader(root_dir=args.test_image_path, batch_size=args.batch_size * 2, shuffle=False)

        image_batch, _ = next(iter(test_images))
        image_batch = image_batch.to(Config.device)

        new_images = generator(image_batch).detach().cpu()

        tvutils.save_image(image_batch, 'test_images.jpg', nrow=4, padding=2, normalize=True, range=(-1, 1))
        tvutils.save_image(new_images, 'generated_images.jpg', nrow=4, padding=2, normalize=True, range=(-1, 1))

        if not os.path.isdir('generated_images/CartoonGAN'):
            os.makedirs('generated_images/CartoonGAN/')
            
        print("Generating Images")
        # generate new images for all images in args.test_image_path, and save them to generated_images/CartoonGAN/ directory
        generate_and_save_images(generator, test_images, 'generated_images/CartoonGAN/')

    else:
        print("Training...")

        print("Loading Discriminator and Feature Extractor...")
        discriminator = Discriminator().to(device)
        feature_extractor = FeatureExtractor().to(device)

        # load dataloaders
        photo_images = load_image_dataloader(root_dir=args.photo_image_dir, batch_size=args.batch_size)
        animation_images = load_image_dataloader(root_dir=args.animation_image_dir, batch_size=args.batch_size)
        edge_smoothed_images = load_image_dataloader(root_dir=args.edge_smoothed_image_dir, batch_size=args.batch_size)

        print("Loading Trainer...")
        trainer = CartoonGANTrainer(generator, discriminator, feature_extractor, photo_images, animation_images,
                                    edge_smoothed_images, lsgan=args.use_modified_model)
        if args.model_path:
            trainer.load_checkpoint(args.model_path)

        print('Start Training...')
        loss_D_hist, loss_G_hist, loss_content_hist = trainer.train(num_epochs=args.num_epochs,
                                                                    initialization_epochs=args.initialization_epochs,
                                                                    save_path=args.model_save_path)


if __name__ == '__main__':
    main()



