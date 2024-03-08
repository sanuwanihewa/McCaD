import argparse
import torch
import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import torchvision

from dataset_autoencoder import CreateDatasetSynthesis

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)







def train_syndiff(rank, gpu, args):

    from backbones.autoencoder import AutoEncoder
    from utils.EMA import EMA


    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    dataset = CreateDatasetSynthesis(phase="train", input_path=args.input_path)
    dataset_val = CreateDatasetSynthesis(phase="val", input_path=args.input_path)


    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=None,
                                              drop_last=True)

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  sampler=None,
                                                  drop_last=True)

    val_l1_loss = np.zeros([2, args.num_epoch, len(data_loader_val)])
    val_psnr_values = np.zeros([2, args.num_epoch, len(data_loader_val)])
    print('train data size:' + str(len(data_loader)))
    print('val data size:' + str(len(data_loader_val)))
    to_range_0_1 = lambda x: (x + 1.) / 2.


    autoencoder = AutoEncoder(args).to(device)
    args.num_channels = 1

    optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))


    if args.use_ema:
        optimizer_autoencoder = EMA(optimizer_autoencoder, ema_decay=args.ema_decay)

    scheduler_autoencoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_autoencoder, args.num_epoch,
                                                                           eta_min=1e-5)


    exp = args.exp
    output_path = args.output_path

    exp_path = os.path.join(output_path, exp)


    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']

        autoencoder.load_state_dict(checkpoint['autoencoder_dict'])

        optimizer_autoencoder.load_state_dict(checkpoint['optimizer_autoencoder'])
        scheduler_autoencoder.load_state_dict(checkpoint['scheduler_autoencoder'])


        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    for epoch in range(init_epoch, args.num_epoch + 1):

        for iteration, (x1) in enumerate(data_loader):

            real_data1 = x1[0].to(device, non_blocking=True)


            autoencoder.zero_grad()


            x2_0_predict_diff,_ = autoencoder(real_data1)

            errG2_L1 = F.l1_loss(x2_0_predict_diff, real_data1)




            torch.autograd.set_detect_anomaly(True)

            errG = errG2_L1
            errG.backward()

            optimizer_autoencoder.step()

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G-Sum: {}'.format(epoch, iteration,
                                                                                         errG.item(),
                                                                                          ))

        if not args.no_lr_decay:
            scheduler_autoencoder.step()

            if epoch % 10 == 0:


                fake_sample1,_  = autoencoder(real_data1)
                fake_sample1=fake_sample1.detach()
                fake_sample2 = torch.cat((real_data1, fake_sample1), axis=-1)
                torchvision.utils.save_image(fake_sample2,
                                             os.path.join(exp_path, 'sample2_discrete_epoch_{}.png'.format(epoch)),
                                             normalize=True)



            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'autoencoder_dict': autoencoder.state_dict(),
                               'optimizer_autoencoder': optimizer_autoencoder.state_dict(),
                               'scheduler_autoencoder': scheduler_autoencoder.state_dict(),

                               }

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer_autoencoder.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(autoencoder.state_dict(), os.path.join(exp_path, 'autoencoder_{}.pth'.format(epoch)))

                if args.use_ema:
                    optimizer_autoencoder.swap_parameters_with_ema(store_params_in_ema=True)


        for iteration, ( x_val) in enumerate(data_loader_val):
            real_data = x_val[0].to(device, non_blocking=True)


            fake_sample1,_  = autoencoder(real_data)
            fake_sample1=fake_sample1.detach()

            fake_sample1 = to_range_0_1(fake_sample1);
            fake_sample1 = fake_sample1 / fake_sample1.mean()
            real_data = to_range_0_1(real_data);
            real_data = real_data / real_data.mean()

            fake_sample1 = fake_sample1.cpu().numpy()
            real_data = real_data.cpu().numpy()
            val_l1_loss[1, epoch, iteration] = abs(fake_sample1 - real_data).mean()

            val_psnr_values[1, epoch, iteration] = psnr(real_data, fake_sample1, data_range=real_data.max())

        print(np.nanmean(val_psnr_values[1, epoch, :]))
        np.save('{}/val_l1_loss.npy'.format(exp_path), val_l1_loss)
        np.save('{}/val_psnr_values.npy'.format(exp_path), val_psnr_values)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', default='/data/BRATS/',
                        help='path to input data')
    parser.add_argument('--output_path', default='/results',
                        help='path to output saves')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5,
                        help='weightening of l1 loss part of diffusion ans cycle models')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--port_num', type=str, default='6021',
                        help='port selection for code')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train_syndiff, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        train_syndiff(rank=0, gpu=0, args=args)

