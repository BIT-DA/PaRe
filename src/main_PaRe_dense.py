# main2.py new method in ORCA's original framework
# main2_aug.py: add augmentation
import os
import argparse
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
from timeit import default_timer
from attrdict import AttrDict

from task_configs_PaRe import get_data, get_config, get_metric, get_optimizer_scheduler
from utils_PaRe import count_params, count_trainable_params, calculate_stats
from embedder_mix_PaRe_dense import get_tgt_model

torch.set_num_threads(2)

def main(use_determined, args, info=None, context=None):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = '/datasets' if use_determined else './datasets'
    print(args)

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)

    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    args.experiment_id = 65535

    # if load_embedder(use_determined, args):
    if args.align_method == 'srcinit' or args.align_method == 'randinit':
        args.embedder_epochs = 0

    model, embedder_stats = get_tgt_model(args, root, sample_shape, num_classes, loss, False, use_determined, context)

    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False)
        
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split)
    metric, compare_metrics = get_metric(root, args.dataset)
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    
    model, ep_start, id_best, train_score, train_losses, embedder_stats_saved = load_state(use_determined, args, context, model, None, None, n_train, freq=args.validation_freq, test=True)
    embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    
    offset = 0 if ep_start == 0 else 1
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    train_full = args.predictor_epochs == 0 or ep_start >= args.predictor_epochs
    
    if args.device == 'cuda':
        model.cuda()
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()

    print("\n------- Experiment Summary --------")
    print("id:", args.experiment_id)
    print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer.params.lr)
    print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
    print('align method:', args.align_method)
    print("finetune method:", args.finetune_method)
    print("param count:", count_params(model), count_trainable_params(model))
    # print(model)
    
    # 加载checkpoint中的optimizer和scheduler
    model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, model, optimizer, scheduler, n_train, freq=args.validation_freq)
    embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    train_time = []

    print("Turning off gradients in src_embedder")
    for name, param in model.named_parameters():
        if "embedder_src" in name:
            param.requires_grad_(False)
            print(name)

    print("\n------- Start Training --------" if ep_start == 0 else "\n------- Resume Training --------")

    for ep in range(ep_start, args.epochs + args.predictor_epochs):
        if not train_full and ep >= args.predictor_epochs:
            args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None, n_train=n_train)
            train_full = True

        time_start = default_timer()

        train_loss = train_one_epoch(context, args, model, ep, optimizer, scheduler, train_loader, src_train_loader, loss, n_train, decoder, transform)
        train_time_ep = default_timer() -  time_start 

        if ep % args.validation_freq == 0 or ep == args.epochs + args.predictor_epochs - 1: 
                
            val_loss, val_score = evaluate(context, args, model, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)

            train_losses.append(train_loss)
            train_score.append(val_score)
            train_time.append(train_time_ep)

            print("[train", "full" if ep >= args.predictor_epochs else "predictor", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))

            if use_determined:
                id_current = save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
                try:
                    context.train.report_training_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"train loss": train_loss, "epoch time": train_time_ep})
                    context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"val score": val_score})
                except:
                    pass
                
        if ep == args.epochs + args.predictor_epochs - 1:
            print("\n------- Start Test --------")
            test_scores = []
            test_model = model
            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            
            
            if use_determined:
                checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
                with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
                    np.save(os.path.join(path, 'test_score.npy'), test_scores)
            else:
                path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
                np.save(os.path.join(path, 'test_score.npy'), test_scores)

           
        if use_determined and context.preempt.should_preempt():
            print("paused")
            return

class SubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau
        self.EPSILON = np.finfo(np.float32).tiny

    def forward(self, scores):
        # print(scores.size())
        # scores = scores.view(1, -1)
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([self.EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

def mixup_embedding_gate(model, x, x_src, ep):
    x_flat = x.view(-1, 128)
    x_src_flat = x_src.view(-1, 128)
    flag=True

    result_x = []

    if x.size()[0] == x_src.size()[0]:
        coefficient = int(args.begin_num - args.end_num * (ep / args.epochs))
        sampler_src = SubsetOperator(k=coefficient, tau=0.9, hard=True)
        sampler_tar = SubsetOperator(k=3136-coefficient, tau=0.9, hard=True)

        gate_values_src = model.gate(x_src_flat)
        gate_values_src = gate_values_src.view(x_src.size()[0], x_src.size()[1])
        gate_values_tar = model.gate(x_flat)
        gate_values_tar = gate_values_tar.view(x.size()[0], x.size()[1])
        mask_src = sampler_src(gate_values_src)
        mask_tar = sampler_tar(gate_values_tar)
        for st in range(x_src.size()[0]):
            mask_src_st = mask_src[st]
            nonzero_indices_src = torch.nonzero(mask_src_st)
            mask_src_st = mask_src_st.view(-1, 1)
            mask_tar_st = mask_tar[st]
            zero_indices_tar = torch.nonzero(1 - mask_tar_st)
            mask_tar_st = mask_tar_st.view(-1, 1)
            x_src_mask = x_src[st] * mask_src_st
            x_tar_mask = x[st] * mask_tar_st
            result_tar = x_tar_mask.clone()
            result_tar[zero_indices_tar, :] = x_src_mask[nonzero_indices_src, :]
            result_x.append(result_tar)
        result_x = torch.stack(result_x)
        lam = coefficient / x.size()[1]
    else:
        result_x = x
        flag = False
        lam = 0
    return result_x, lam, flag

def mixup_embedding_gate_ori(model, x, x_src, ep):
    x_flat = x.view(-1, 128)
    x_src_flat = x_src.view(-1, 128)
    flag=True

    if x.size()[0] == x_src.size()[0]:
        coefficient = int(args.begin_num - args.end_num * (ep / args.epochs))
        gate_values_src = model.gate(x_src_flat)
        gate_values_src = gate_values_src.view(x_src.size()[0], x_src.size()[1])
        gate_values_tar = model.gate(x_flat)
        gate_values_tar = gate_values_tar.view(x.size()[0], x.size()[1])
        top_values_src, top_indices_src = torch.topk(gate_values_src, coefficient, dim=1)
        sorted_indices = torch.argsort(gate_values_tar, dim=1)
        top_indices_tar = sorted_indices[:, -coefficient:]
        for i in range(x.size()[0]):
            x[i, top_indices_tar[i], :] = x_src[i, top_indices_src[i], :]
        result_x = x
        lam = coefficient / x.size()[1]
    else:
        result_x = x
        lam = 1.0
        flag = False
    return result_x, lam, flag

def train_one_epoch(context, args, model, ep, optimizer, scheduler, loader, src_train_loader, loss, temp, decoder=None, transform=None):    

    model.train()

    # print(model)
    
    src_train_iter = iter(src_train_loader)
                    
    train_loss = 0
    total_dis_tar = 0
    total_dis_mix = 0

    optimizer.zero_grad()

    # loss_src = nn.CrossEntropyLoss()
    loss_src = nn.BCEWithLogitsLoss()

    for i, data in enumerate(loader):

        if transform is not None:
            x, y, z = data
            z = z.to(args.device)
        else:
            x, y = data 

        try:
            x_src, y_src = next(src_train_iter)
        except StopIteration:
            del src_train_iter
            src_train_iter = iter(src_train_loader)
            x_src, y_src = next(src_train_iter)
        
        x, y = x.to(args.device), y.to(args.device)
        x_src, y_src = x_src.to(args.device), y_src.to(args.device)

        y_src[y_src==255] = 0
        y_src[y_src>0] = 1

        out_raw, out = model(x)

        out_raw_source = model.embedder_source(x_src)[0]
        
        if ep < 10:
            out_raw_mix, lam, flag = mixup_embedding_gate(model, out_raw, out_raw_source, ep)
        else:
            out_raw_mix, lam, flag = mixup_embedding_gate_ori(model, out_raw, out_raw_source, ep)

        model.output_pre_with_raw = False
        model.forward_mid = True

        out_mixed = model(out_raw_mix)

        if isinstance(out, dict):
            out = out['out']

        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            out_mixed = decoder.decode(out_mixed).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)
            y_src = decoder.decode(y_src).view(x.shape[0], -1)

        if transform is not None:
            out = transform(out, z)
            out_mixed_tar = transform(out_mixed, z)
            y = transform(y, z)

        if args.dataset[:4] == "DRUG":
            out = out.squeeze(1)

        if flag:
            l = loss(out, y) + 0.3 * ((1-lam) * loss(out_mixed, y) + lam * loss_src(out_mixed, y_src.float()))
        else:
            l = loss(out, y)
        l.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if (i + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if args.lr_sched_iter:
            scheduler.step()

        train_loss += l.item()

        model.output_pre_with_raw = True
        model.forward_mid = False

        if i >= temp - 1:
            break

    if (not args.lr_sched_iter):
        scheduler.step()

    return train_loss / temp


def evaluate(context, args, model, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
    model.eval()
    
    eval_loss, eval_score = 0, 0

    model.output_pre_with_raw = False
    model.forward_mid = False
    
    if fsd_epoch is None:

        ys, outs, n_eval, n_data = [], [], 0, 0

        with torch.no_grad():
            for i, data in enumerate(loader):
                if transform is not None:
                    x, y, z = data
                    z = z.to(args.device)
                else:
                    x, y = data
                                    
                x, y = x.to(args.device), y.to(args.device)

                out = model(x)

                if isinstance(out, dict):
                    out = out['out']

                if decoder is not None:
                    out = decoder.decode(out).view(x.shape[0], -1)
                    y = decoder.decode(y).view(x.shape[0], -1)
                                    
                if transform is not None:
                    out = transform(out, z)
                    y = transform(y, z)

                if args.dataset[:4] == "DRUG":
                    out = out.squeeze(1)

                outs.append(out)
                ys.append(y)
                n_data += x.shape[0]

                if n_data >= args.eval_batch_size or i == len(loader) - 1:
                    outs = torch.cat(outs, 0)
                    ys = torch.cat(ys, 0)

                    eval_loss += loss(outs, ys).item()
                    eval_score += metric(outs, ys).item()
                    n_eval += 1

                    ys, outs, n_data = [], [], 0

            eval_loss /= n_eval
            eval_score /= n_eval

    else:
        outs, ys = [], []
        with torch.no_grad():
            for ix in range(loader.len):

                x, y = loader[ix]
                x, y = x.to(args.device), y.to(args.device)
                out = model(x).mean(0).unsqueeze(0)
                eval_loss += loss(out, y).item()
                outs.append(torch.sigmoid(out).detach().cpu().numpy()[0])
                ys.append(y.detach().cpu().numpy()[0])

        outs = np.asarray(outs).astype('float32')
        ys = np.asarray(ys).astype('int32')
        stats = calculate_stats(outs, ys)
        eval_score = 1-np.mean([stat['AP'] for stat in stats])
        eval_loss /= n_eval

    model.output_pre_with_raw = True
    model.forward_mid = False

    return eval_loss, eval_score


########################## Helper Funcs ##########################

def save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.align_method) + '_' + str(args.finetune_method) + '_exp' + str(args.experiment_id) + "/eph" + str(ep)
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
        return ep

    else:
        checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
        with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
            save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
            return uuid


def save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats):
    np.save(os.path.join(path, 'hparams.npy'), args)
    np.save(os.path.join(path, 'train_score.npy'), train_score)
    np.save(os.path.join(path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(path, 'embedder_stats.npy'), embedder_stats)

    model_state_dict = {
                'network_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
    torch.save(model_state_dict, os.path.join(path, 'state_dict.pt'))

    rng_state_dict = {
                'cpu_rng_state': torch.get_rng_state(),
                'gpu_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate()
            }
    torch.save(rng_state_dict, os.path.join(path, 'rng_state.ckpt'))


def load_embedder(use_determined, args):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        return os.path.isfile(os.path.join(path, 'state_dict.pt'))
    else:

        info = det.get_cluster_info()
        checkpoint_id = info.latest_checkpoint
        return checkpoint_id is not None


def load_state(use_determined, args, context, model, optimizer, scheduler, n_train, checkpoint_id=None, test=False, freq=1):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
            return model, 0, 0, [], [], None
    else:

        if checkpoint_id is None:
            info = det.get_cluster_info()
            checkpoint_id = info.latest_checkpoint
            if checkpoint_id is None:
                return model, 0, 0, [], [], None
        
        checkpoint = client.get_checkpoint(checkpoint_id)
        path = checkpoint.download()

    train_score = np.load(os.path.join(path, 'train_score.npy'))
    train_losses = np.load(os.path.join(path, 'train_losses.npy'))
    embedder_stats = np.load(os.path.join(path, 'embedder_stats.npy'))
    epochs = freq * (len(train_score) - 1) + 1
    checkpoint_id = checkpoint_id if use_determined else epochs - 1
    model_state_dict = torch.load(os.path.join(path, 'state_dict.pt'))
    model.load_state_dict(model_state_dict['network_state_dict'])
    
    if not test:
        optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(model_state_dict['scheduler_state_dict'])

        rng_state_dict = torch.load(os.path.join(path, 'rng_state.ckpt'), map_location='cpu')
        torch.set_rng_state(rng_state_dict['cpu_rng_state'])
        torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
        np.random.set_state(rng_state_dict['numpy_rng_state'])
        random.setstate(rng_state_dict['py_rng_state'])

        if use_determined: 
            try:
                for ep in range(epochs):
                    if ep % freq == 0:
                        context.train.report_training_metrics(steps_completed=(ep + 1) * n_train, metrics={"train loss": train_losses[ep // freq]})
                        context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train, metrics={"val score": train_score[ep // freq]})
            except:
                print("load error")

    return model, epochs, checkpoint_id, list(train_score), list(train_losses), embedder_stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')

    args = parser.parse_args()
    if args.config is not None:     
        import yaml

        with open(args.config, 'r') as stream:
            args = AttrDict(yaml.safe_load(stream)['hyperparameters'])
            main(False, args)

    else:
        import determined as det
        from determined.experimental import client
        from determined.pytorch import DataLoader

        info = det.get_cluster_info()
        args = AttrDict(info.trial.hparams)
        
        with det.core.init() as context:
            main(True, args, info, context)
