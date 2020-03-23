import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import CycleGANConfig as Config
from CartoonGAN_model import FeatureExtractor


class CycleGANTrainer:
    def __init__(self, G, F, D_x, D_y,
                 photo_image_loader, animation_image_loader, edge_smoothed_image_loader,
                 lambda_cycle=Config.lambda_cycle, lambda_identity=Config.lambda_identity,
                 use_edge_smoothed=False, use_initialization=False):

        self.G = G.to(Config.device)  # Generator X -> Y
        self.F = F.to(Config.device)  # Generator Y -> X
        self.D_x = D_x.to(Config.device)
        self.D_y = D_y.to(Config.device)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=Config.lr, betas=(Config.adam_beta1, 0.999))
        self.F_optimizer = optim.Adam(self.F.parameters(), lr=Config.lr, betas=(Config.adam_beta1, 0.999))
        self.D_x_optimizer = optim.Adam(self.D_x.parameters(), lr=Config.lr, betas=(Config.adam_beta1, 0.999))
        self.D_y_optimizer = optim.Adam(self.D_y.parameters(), lr=Config.lr, betas=(Config.adam_beta1, 0.999))

        self.photo_image_loader = photo_image_loader
        self.animation_image_loader = animation_image_loader
        if use_edge_smoothed:
            assert edge_smoothed_image_loader is not None
        self.edge_smoothed_image_loader = edge_smoothed_image_loader

        self.generated_x_images = ImagePool(50)
        self.generated_y_images = ImagePool(50)

        self.GAN_criterion = nn.MSELoss().to(Config.device)
        self.Cycle_criterion = nn.L1Loss().to(Config.device)
        self.Identity_criterion = nn.L1Loss().to(Config.device)

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        if use_initialization:
            self.Initialization_criterion = nn.L1Loss().to(Config.device)
            self.lambda_initialization = Config.content_loss_weight
            self.feature_extractor = FeatureExtractor().to(Config.device)

        self.use_edge_smoothed = use_edge_smoothed
        self.use_initialization = use_initialization

        self.curr_initialization_epoch = 0
        self.curr_epoch = 0

        self.init_loss_hist = []
        self.loss_D_x_hist = []
        self.loss_D_y_hist = []
        self.loss_G_GAN_hist = []
        self.loss_F_GAN_hist = []
        self.loss_cycle_hist = []
        self.loss_identity_hist = []
        self.print_every = Config.print_every

    def train(self, num_epochs=Config.num_epochs, initialization_epochs=0, save_path="checkpoints/CycleGAN/"):

        for init_epoch in range(self.curr_initialization_epoch, initialization_epochs):
            start = time.time()
            epoch_loss = 0

            for ix, ((photo_images, _), (animation_images, _)) in enumerate(
                    zip(self.photo_image_loader, self.animation_image_loader), 0
            ):
                photo_images = photo_images.to(Config.device)
                animation_images = animation_images.to(Config.device)

                loss = self.initialize_step(photo_images, animation_images)
                self.init_loss_hist.append(loss)
                epoch_loss += loss

                # print progress
                if (ix + 1) % self.print_every == 0:
                    print("Initialization Phase Epoch {0} Iteration {1}: Content Loss: {2:.4f}".format(init_epoch + 1,
                                                                                                       ix + 1,
                                                                                                       epoch_loss / (
                                                                                                                   ix + 1)))

            print("Initialization Phase [{0}/{1}], {2:.4f} seconds".format(init_epoch + 1, initialization_epochs,
                                                                           time.time() - start))
            self.curr_initialization_epoch += 1

        for epoch in range(self.curr_epoch, num_epochs):
            start = time.time()
            epoch_loss_D_x = 0
            epoch_loss_D_y = 0
            epoch_loss_G_GAN = 0
            epoch_loss_F_GAN = 0
            epoch_loss_cycle = 0
            epoch_loss_identity = 0

            for ix, ((photo_images, _), (animation_images, _), (edge_smoothed_images, _)) in enumerate(
                zip(self.photo_image_loader, self.animation_image_loader, self.edge_smoothed_image_loader), 0
            ):

                photo_images = photo_images.to(Config.device)
                animation_images = animation_images.to(Config.device)
                edge_smoothed_images = edge_smoothed_images.to(Config.device)

                loss_D_x, loss_D_y, loss_G_GAN, loss_F_GAN, loss_cycle, loss_identity = self.train_step(photo_images,
                                                                                                        animation_images,
                                                                                                        edge_smoothed_images)

                self.loss_D_x_hist.append(loss_D_x)
                self.loss_D_y_hist.append(loss_D_y)
                self.loss_G_GAN_hist.append(loss_G_GAN)
                self.loss_F_GAN_hist.append(loss_F_GAN)
                self.loss_cycle_hist.append(loss_cycle)
                self.loss_identity_hist.append(loss_identity)

                epoch_loss_D_x += loss_D_x
                epoch_loss_D_y += loss_D_y
                epoch_loss_G_GAN += loss_G_GAN
                epoch_loss_F_GAN += loss_F_GAN
                epoch_loss_cycle += loss_cycle
                epoch_loss_identity += loss_identity

                # print progress
                if (ix + 1) % self.print_every == 0:
                    print("Training Phase Epoch {0} Iteration {1}: loss_D_x: {2:.4f} loss_D_y: {3:.4f} loss_G: {4:.4f} loss_F: {5:.4f} "
                          "loss_cycle: {6:.4f} loss_identity: {7:.4f}".format(epoch + 1, ix+1, epoch_loss_D_x / (ix + 1), epoch_loss_D_y / (ix + 1),
                                                                              epoch_loss_G_GAN / (ix + 1), epoch_loss_F_GAN / (ix + 1),
                                                                              epoch_loss_cycle / (ix + 1), epoch_loss_identity / (ix + 1)))  # print progress

            self.curr_epoch += 1
            print("Training Phase [{0}/{1}], {2:.4f} seconds".format(self.curr_epoch, num_epochs, time.time() - start))

        # Training finished, save checkpoint
        self.save_checkpoint(os.path.join(save_path, 'checkpoint-epoch-{0}.ckpt'.format(num_epochs)))

        return self.loss_D_x_hist, self.loss_D_y_hist, self.loss_G_GAN_hist, self.loss_F_GAN_hist, \
               self.loss_cycle_hist, self.loss_identity_hist

    def train_step(self, photo_images, animation_images, edge_smoothed_images):
        # photo images are X, animation images are Y

        self.D_x.zero_grad()
        self.D_y.zero_grad()
        self.G.zero_grad()
        self.F.zero_grad()

        # Generate images and save them to image buffers
        generated_y = self.G(photo_images)
        self.generated_y_images.save(generated_y.detach())

        generated_x = self.F(animation_images)
        self.generated_x_images.save(generated_x.detach())

        # train D_y with animation_images and generated_y, (and optionally, edge_smoothed_images)
        animation_output = self.D_y(animation_images)
        animation_target = torch.ones_like(animation_output)
        loss_animation = self.GAN_criterion(animation_output, animation_target)
        loss_D_y = loss_animation

        generated_y_sample = self.generated_y_images.sample()
        generated_output = self.D_y(generated_y_sample)
        generated_target = torch.zeros_like(generated_output)
        loss_generated = self.GAN_criterion(generated_output, generated_target)
        loss_D_y += loss_generated

        if self.use_edge_smoothed:
            # discriminator is trained to classify edge smoothed animation images as 0 (not animation iamges)
            # this loss force generator to generate images with sharper edges
            edge_smoothed_output = self.D_y(edge_smoothed_images)
            edge_smoothed_target = torch.zeros_like(edge_smoothed_output)
            loss_edge_smoothed = self.GAN_criterion(edge_smoothed_output, edge_smoothed_target)
            loss_D_y += loss_edge_smoothed

        # according to cyclegan paper section 7.1, gradient of discriminators were divided by 2
        # to slow down the rate at which D learns relative G
        (loss_D_y / 2).backward()
        self.D_y_optimizer.step()

        # train D_x with photo_images and generated_x
        photo_output = self.D_x(photo_images)
        photo_target = torch.ones_like(photo_output)
        loss_photo = self.GAN_criterion(photo_output, photo_target)
        loss_D_x = loss_photo

        generated_x_sample = self.generated_x_images.sample()
        generated_output = self.D_x(generated_x_sample)
        generated_target = torch.zeros_like(generated_output)
        loss_generated = self.GAN_criterion(generated_output, generated_target)
        loss_D_x += loss_generated

        # according to cyclegan paper section 7.1, gradient of discriminators were divided by 2
        # to slow down the rate at which D learns relative G
        (loss_D_x / 2).backward()
        self.D_x_optimizer.step()

        # time to train G and F
        self.G.zero_grad()
        self.F.zero_grad()

        # 1. GAN loss
        generated_y_output = self.D_y(generated_y)
        generated_y_target = torch.ones_like(generated_y_output)
        loss_G_GAN = self.GAN_criterion(generated_y_output, generated_y_target)

        generated_x_output = self.D_x(generated_x)
        generated_x_target = torch.ones_like(generated_x_output)
        loss_F_GAN = self.GAN_criterion(generated_x_output, generated_x_target)

        # 2. Cycle-Consistency loss
        cycle_x = self.F(generated_y)  # X -> Y -> X
        cycle_y = self.G(generated_x)  # Y -> X -> Y

        loss_cycle = self.lambda_cycle * self.Cycle_criterion(cycle_x, photo_images)
        loss_cycle += self.lambda_cycle * self.Cycle_criterion(cycle_y, animation_images)

        # 3. identity loss
        G_y = self.G(animation_images)
        F_x = self.F(photo_images)
        loss_identity = self.lambda_identity * self.Identity_criterion(G_y, animation_images)
        loss_identity += self.lambda_identity * self.Identity_criterion(F_x, photo_images)

        generator_losses = loss_G_GAN + loss_F_GAN + loss_cycle + loss_identity
        generator_losses.backward()
        self.G_optimizer.step()
        self.F_optimizer.step()

        return loss_D_x.item(), loss_D_y.item(), loss_G_GAN.item(), loss_F_GAN.item(), \
               loss_cycle.item(), loss_identity.item()

    def initialize_step(self, photo_images, animation_images):
        # TODO
        # Train only using cycle-consistency and identity loss
        self.G.zero_grad()
        self.F.zero_grad()

        generated_y = self.G(photo_images)
        generated_x = self.F(animation_images)

        cycle_x = self.F(generated_y)  # X -> Y -> X
        cycle_y = self.G(generated_x)  # Y -> X -> Y

        loss_cycle = self.lambda_cycle * self.Cycle_criterion(cycle_x, photo_images)
        loss_cycle += self.lambda_cycle * self.Cycle_criterion(cycle_y, animation_images)

        G_y = self.G(animation_images)
        F_x = self.F(photo_images)
        loss_identity = self.lambda_identity * self.Identity_criterion(G_y, animation_images)
        loss_identity += self.lambda_identity * self.Identity_criterion(F_x, photo_images)

        initialization_loss = loss_cycle + loss_identity
        initialization_loss.backward()
        self.G_optimizer.step()
        self.F_optimizer.step()

        return initialization_loss.item()

    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {
                'G_state_dict': self.G.state_dict(),
                'F_state_dict': self.F.state_dict(),
                'D_x_state_dict': self.D_x.state_dict(),
                'D_y_state_dict': self.D_y.state_dict(),
                'G_optimizer_state_dict': self.G_optimizer.state_dict(),
                'F_optimizer_state_dict': self.F_optimizer.state_dict(),
                'D_x_optimizer_state_dict': self.D_x_optimizer.state_dict(),
                'D_y_optimizer_state_dict': self.D_y_optimizer.state_dict(),
                'loss_D_x_hist': self.loss_D_x_hist,
                'loss_D_y_hist': self.loss_D_y_hist,
                'loss_G_GAN_hist': self.loss_G_GAN_hist,
                'loss_F_GAN_hist': self.loss_F_GAN_hist,
                'loss_cycle_hist': self.loss_cycle_hist,
                'loss_identity_hist': self.loss_identity_hist,
                'init_loss_hist': self.init_loss_hist,
                'curr_epoch': self.curr_epoch

            }, checkpoint_path
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.F.load_state_dict(checkpoint['F_state_dict'])
        self.D_x.load_state_dict(checkpoint['D_x_state_dict'])
        self.D_y.load_state_dict(checkpoint['D_y_state_dict'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.F_optimizer.load_state_dict(checkpoint['F_optimizer_state_dict'])
        self.D_x_optimizer.load_state_dict(checkpoint['D_x_optimizer_state_dict'])
        self.D_y_optimizer.load_state_dict(checkpoint['D_y_optimizer_state_dict'])
        self.loss_D_x_hist = checkpoint['loss_D_x_hist']
        self.loss_D_y_hist = checkpoint['loss_D_y_hist']
        self.loss_G_GAN_hist = checkpoint['loss_G_GAN_hist']
        self.loss_F_GAN_hist = checkpoint['loss_F_GAN_hist']
        self.loss_cycle_hist = checkpoint['loss_cycle_hist']
        self.loss_identity_hist = checkpoint['loss_identity_hist']
        self.init_loss_hist = checkpoint['init_loss_hist']
        try:
            self.curr_epoch = checkpoint['curr_epoch']
        except:
            self.curr_epoch = int(checkpoint_path.split('-')[-1].split('.')[0])


class ImagePool:

    def __init__(self, maxlen=50):
        self.buffer = []
        self.maxlen = maxlen

    def save(self, images):
        for image in images:
            if len(self.buffer) >= self.maxlen:
                idx = np.random.randint(0, self.maxlen)
                self.buffer[idx] = image
            else:
                self.buffer.append(image)

    def sample(self, sample_size=Config.batch_size):
        idxs = np.random.choice(len(self.buffer), sample_size)
        return torch.stack([self.buffer[idx] for idx in idxs])

