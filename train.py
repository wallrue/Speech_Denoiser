# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:38:05 2023

@author: turner_le
"""
import os
import torch.nn.functional as F
import torch
import logging
import argparse
import torch.multiprocessing as mp
from tools.utils import power_compress, power_uncompress
from torchinfo import summary
from models.generator import TSCNet
from models import discriminator
from data import dataloader 
from tqdm import tqdm

MULTI_DEVICES=False
USE_GPU = False
if MULTI_DEVICES:
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group
else:
    from torch.nn.parallel import DataParallel as DDP   

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='dir to VCTK-DEMAND dataset',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

class Trainer:
    def __init__(self, train_ds, test_ds, gpu_id: int):
        # Hamming window: window length (intervals) 400 points; hop length (non-intersecting) 100 points
        self.gpu_id = torch.device('cuda:{}'.format(gpu_id)) if gpu_id != None else torch.device('cpu')
        self.n_fft, self.hop = 400, 100 
        self.train_ds, self.test_ds = train_ds, test_ds
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(self.gpu_id)
        summary(
            self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)], device=self.gpu_id
        )
        self.discriminator = discriminator.Discriminator(ndf=16).to(self.gpu_id)
        summary(
            self.discriminator,
            [
                (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
                (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
            ], 
            device=self.gpu_id,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * args.init_lr
        )

        if gpu_id != None: #Setup parallel processing
            self.model = DDP(self.model, device_ids=[gpu_id]) 
            self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])

    def forward(self, clean, noisy):
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }

    def G_loss(self, generator_outputs):
        predict_fake_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["est_mag"]
        )
        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss = (
            args.loss_weights[0] * loss_ri
            + args.loss_weights[1] * loss_mag
            + args.loss_weights[2] * time_loss
            + args.loss_weights[3] * gen_loss_GAN
        )
        return loss

    def D_loss(self, generator_outputs):
        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)
        pesq_score = torch.FloatTensor(pesq_score).to(self.gpu_id) if pesq_score != None else None
        
        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
            )
            predict_max_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["clean_mag"]
            )
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]
            ) + F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = None
        return discrim_loss_metric
    
    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward(clean, noisy)
        
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.G_loss(generator_outputs)

        discrim_loss_metric = self.D_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()
    
    def test(self):    
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))
        return gen_loss_avg

    def train_step(self, batch):
        # Trainer generator
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.G_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric = self.D_loss(generator_outputs)
        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])
        return loss.item(), discrim_loss_metric.item()
    
    def train(self):          
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )
        for epoch in tqdm(range(args.epochs)):
            self.model.train()
            self.discriminator.train()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                if (step % args.log_interval) == 0:
                    logging.info(
                        template.format(self.gpu_id, epoch, step, loss, disc_loss)
                    )
            gen_loss = self.test()
            path = os.path.join(
                args.save_model_dir,
                "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
            )
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            if self.gpu_id == torch.device('cuda:0') or self.gpu_id == torch.device('cpu'):
                torch.save(self.model.module.state_dict(), path)
                
            scheduler_G.step()
            scheduler_D.step()

def main(rank: int, world_size: int, args):
    # Init process
    if MULTI_DEVICES:
        init_process_group(backend="gloo", rank=rank, world_size=world_size) #Windows: "gloo", Linux: "nccl"
    
    # Process
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)
    train_ds, test_ds = dataloader.load_data(
        args.data_dir, args.batch_size, 1, args.cut_len, MULTI_DEVICES
    )
    trainer = Trainer(train_ds, test_ds, rank)
    trainer.train()
    
    # Deinit process
    if MULTI_DEVICES:
        destroy_process_group()
    
def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if MULTI_DEVICES:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = find_free_port() #"12355"
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    else:
        main(0 if USE_GPU else None, world_size, args)


