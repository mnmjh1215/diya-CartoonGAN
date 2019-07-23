import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import CartoonGANConfig as Config


class CartoonGANTrainer:
    def __init__(self, generator, discriminator, feature_extractor,
                 photo_image_loader, animation_image_loader, edge_smoothed_image_loader,
                 content_loss_weight=Config.content_loss_weight, lsgan=False):
        """
        
        :param generator: CartoonGAN generator
        :param discriminator: CartoonGAN discriminator
        :param feature_extractor: feature extractor, VGG in CartoonGAN
        :param photo_image_loader:
        :param animation_image_loader:
        :param edge_smoothed_image_loader:
        """

        # just in case our generator and discriminator are not using Config.device
        self.generator = generator.to(Config.device)
        self.discriminator = discriminator.to(Config.device)
        self.feature_extractor = feature_extractor.to(Config.device)

        self.photo_image_loader = photo_image_loader
        self.animation_image_loader = animation_image_loader
        self.edge_smoothed_image_loader = edge_smoothed_image_loader

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=Config.lr, betas=(Config.adam_beta1, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=Config.lr,
                                         betas=(Config.adam_beta1, 0.999))
        if not lsgan:
            self.disc_criterion = nn.BCEWithLogitsLoss().to(Config.device)  # for discriminator GAN loss
            self.gen_criterion_gan = nn.BCEWithLogitsLoss().to(Config.device)  # for generator GAN loss
        else:
            # use Least Square GAN
            self.disc_criterion = nn.MSELoss().to(Config.device)
            self.gen_criterion_gan = nn.MSELoss().to(Config.device)
        self.gen_criterion_content = nn.L1Loss().to(Config.device)  # for generator content loss
        self.content_loss_weight = content_loss_weight

        self.curr_initialization_epoch = 0
        self.curr_epoch = 0

        self.init_loss_hist = []
        self.loss_D_hist = []
        self.loss_G_hist = []
        self.loss_content_hist = []
        self.print_every = Config.print_every

    def train(self, num_epochs=Config.num_epochs, initialization_epochs=Config.initialization_epochs,
              save_path='checkpoints/CartoonGAN/'):
        # if not initialized, do it!
        if self.curr_initialization_epoch < initialization_epochs:
            for init_epoch in range(self.curr_initialization_epoch, initialization_epochs):
                start = time.time()
                epoch_loss = 0

                for ix, (photo_images, _) in enumerate(self.photo_image_loader, 0):
                    photo_images = photo_images.to(Config.device)

                    loss = self.initialize_step(photo_images)
                    self.init_loss_hist.append(loss)
                    epoch_loss += loss

                    # print progress
                    if (ix + 1) % self.print_every == 0:
                        print("Initialization Phase Epoch {0} Iteration {1}: Content Loss: {2:.4f}".format(init_epoch+1,
                                                                                                           ix + 1,
                                                                                                           epoch_loss / (ix + 1)))

                print("Initialization Phase [{0}/{1}], {2:.4f} seconds".format(init_epoch + 1, initialization_epochs,
                                                                             time.time() - start))
                self.curr_initialization_epoch += 1

        for epoch in range(self.curr_epoch, num_epochs):
            start = time.time()
            epoch_loss_D = 0
            epoch_loss_G = 0
            epoch_loss_content = 0

            for ix, ((animation_images, _), (edge_smoothed_images, _), (photo_images, _)) in enumerate(
                    zip(self.animation_image_loader,
                        self.edge_smoothed_image_loader,
                        self.photo_image_loader), 0):
                # do train_step...!
                animation_images = animation_images.to(Config.device)
                edge_smoothed_images = edge_smoothed_images.to(Config.device)
                photo_images = photo_images.to(Config.device)
                
                loss_D, loss_G, loss_content = self.train_step(animation_images, edge_smoothed_images, photo_images)
                epoch_loss_D += loss_D
                epoch_loss_G += loss_G
                epoch_loss_content += loss_content

                self.loss_D_hist.append(loss_D)
                self.loss_G_hist.append(loss_G)
                self.loss_content_hist.append(loss_content)

                if (ix + 1) % self.print_every == 0:
                    print("Training Phase Epoch {0} Iteration {1}, loss_D: {2:.4f}, "
                          "loss_G: {3:.4f}, loss_content: {4:.4f}".format(epoch + 1, ix + 1, epoch_loss_D / (ix + 1),
                                                                          epoch_loss_G / (ix + 1),
                                                                          epoch_loss_content / (ix + 1)))

            # end of epoch
            print("Training Phase [{0}/{1}], {2:.4f} seconds".format(epoch + 1, num_epochs, time.time() - start))
            self.curr_epoch += 1

        # Training finished, save checkpoint
        if not os.path.isdir('checkpoints/'):
            os.mkdir('checkpoints/')
        if not os.path.isdir('checkpoints/CartoonGAN/'):
            os.mkdir('checkpoints/CartoonGAN/')

        self.save_checkpoint(os.path.join(save_path, 'checkpoint-epoch-{0}.ckpt'.format(num_epochs)))

        return self.loss_D_hist, self.loss_G_hist, self.loss_content_hist

    def train_step(self, animation_images, edge_smoothed_images, photo_images):
        self.discriminator.zero_grad()
        self.generator.zero_grad()

        loss_D = 0
        loss_G = 0
        loss_content = 0

        # 1. Train Discriminator
        # 1-1. Train Discriminator using animation images
        animation_disc_output = self.discriminator(animation_images)
        animation_target = torch.ones_like(animation_disc_output)
        loss_real = self.disc_criterion(animation_disc_output, animation_target)

        # 1-2. Train Discriminator using edge smoothed images
        edge_smoothed_disc_output = self.discriminator(edge_smoothed_images)
        edge_smoothed_target = torch.zeros_like(edge_smoothed_disc_output)
        loss_edge = self.disc_criterion(edge_smoothed_disc_output, edge_smoothed_target)

        # 1-3. Train Discriminator using generated images
        generated_images = self.generator(photo_images).detach()

        generated_output = self.discriminator(generated_images)
        generated_target = torch.zeros_like(generated_output)
        loss_generated = self.disc_criterion(generated_output, generated_target)

        loss_disc = loss_real + loss_edge + loss_generated

        loss_disc.backward()
        loss_D = loss_disc.item()

        self.disc_optimizer.step()

        # 2. Train Generator
        self.generator.zero_grad()

        # 2-1. Train Generator using adversarial loss, using generated images
        generated_images = self.generator(photo_images)

        generated_output = self.discriminator(generated_images)
        generated_target = torch.ones_like(generated_output)
        loss_adv = self.gen_criterion_gan(generated_output, generated_target)

        # 2-2. Train Generator using content loss
        x_features = self.feature_extractor((photo_images + 1) / 2).detach()
        Gx_features = self.feature_extractor((generated_images + 1) / 2)

        loss_content = self.content_loss_weight * self.gen_criterion_content(Gx_features, x_features)

        loss_gen = loss_adv + loss_content
        loss_gen.backward()

        loss_G = loss_adv.item()
        loss_content = loss_content.item()

        self.gen_optimizer.step()

        return loss_D, loss_G, loss_content

    def initialize_step(self, photo_images):
        self.generator.zero_grad()
        x_features = self.feature_extractor((photo_images + 1) / 2).detach()  # move [-1, 1] to [0, 1]
        Gx = self.generator(photo_images)
        Gx_features = self.feature_extractor((Gx + 1) / 2)  # move [-1, 1] to [0, 1]

        content_loss = self.content_loss_weight * self.gen_criterion_content(Gx_features, x_features)
        content_loss.backward()
        self.gen_optimizer.step()

        return content_loss.item()

    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'curr_epoch': self.curr_epoch,
            'curr_initialization_epoch': self.curr_initialization_epoch,
            'loss_G_hist': self.loss_G_hist,
            'loss_D_hist': self.loss_D_hist,
            'loss_content_hist': self.loss_content_hist
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.loss_G_hist = checkpoint['loss_G_hist']
        self.loss_D_hist = checkpoint['loss_D_hist']
        self.loss_content_hist = checkpoint['loss_content_hist']
        self.curr_epoch = checkpoint['curr_epoch']
        self.curr_initialization_epoch = checkpoint['curr_initialization_epoch']
