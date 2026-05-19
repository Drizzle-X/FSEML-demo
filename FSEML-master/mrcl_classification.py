import argparse
import logging
import bisect

import numpy as np
import torch
from tensorboardX import SummaryWriter

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner_factory import MetaLearnerFactory
logger = logging.getLogger('experiment')


def apply_meta_lr_schedule(maml, step, base_meta_lr, schedule_steps=None, schedule_values=None):
    if not schedule_steps or not schedule_values:
        target_lr = base_meta_lr
    else:
        idx = bisect.bisect_right(schedule_steps, step)
        target_lr = schedule_values[min(idx, len(schedule_values) - 1)]

    for param_group in maml.optimizer.param_groups:
        param_group["lr"] = target_lr

    return target_lr


def load_checkpoint_into_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    model.load_state_dict(state_dict)


def main(args):
    utils.set_seed(args.seed)
    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)                                                             
                               
                                                                                                                          
                                                                          
                                                                                                                       
                                                                               
                                                                                                                  
                   


    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

                                                                      
    args.classes = list(range(963))                           

                                           
    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)                    
    dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True)                     

                                   
                  
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5,                     
                                                shuffle=True, num_workers=args.num_workers)
                   
    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,                    
                                                 shuffle=True, num_workers=args.num_workers)
                              
    sampler = ts.SamplerFactory.get_sampler(
        args.dataset,
        args.classes,
        dataset,
        dataset_test,
        num_workers=args.num_workers,
    )                                                                                     
                                                                                                             
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    maml = MetaLearnerFactory.build(args)
    parallel_maml = maml
    if torch.cuda.device_count() > 1:
        print("We are using", torch.cuda.device_count(), "GPUs~")
        parallel_maml = torch.nn.DataParallel(maml)
    maml.to(device)
    parallel_maml.to(device)
                                                                                                     
    if args.checkpoint:                 
        load_checkpoint_into_model(maml.net, args.saved_model)

    maml = maml.to(device)              
                
    if args.treatment != "FSEML":
        utils.freeze_layers(args.rln, maml)                                 
                                                                           
    for step in range(args.steps):
        current_meta_lr = apply_meta_lr_schedule(
            maml,
            step,
            args.meta_lr,
            args.meta_lr_schedule_steps,
            args.meta_lr_schedule_values,
        )

        t1 = np.random.choice(args.classes, args.tasks, replace=False)                                                                              

        d_traj_iterators = []
        for t in t1:                              
            d_traj_iterators.append(sampler.sample_task([t]))                                     
        d_rand_iterator = sampler.get_complete_iterator()                                                          
                                 
                                             
        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args.update_step, reset=not args.no_reset)                                                        
        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

        accs, loss, loss_auto, loss_pre = parallel_maml(x_spt, y_spt, x_qry, y_qry)              
                                                      
        if step % 40 == 0:
                                                                       
            logger.info('step: %d \t meta_lr %.6f \t training acc %s and training loss %s, loss_auto: %s, loss_pre: %s', step, current_meta_lr, str(accs), str(loss), str(loss_auto), str(loss_pre))
        if step % 100 == 0 or step == 19999:
            torch.save(maml.net, args.model_name)        
        if step % 500 == 0 and step != 0:                                           
            utils.log_accuracy_train(maml, my_experiment, iterator_train, device, writer, step)                                     
            utils.log_accuracy(maml, my_experiment, iterator_test, device, writer, step)                                    


if __name__ == '__main__':                                              
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=70000)
    argparser.add_argument('--model_name', help='Name of model to be saved', default='FSEML_Model_main.net')
    argparser.add_argument('--treatment', help='Neuromodulation or OML or AE', default='FSEML')
    argparser.add_argument('--checkpoint', help='Use a checkpoint model', action='store_true')
    argparser.add_argument('--saved_model', help='Saved model to load', default='FSEML_Model_main.net')
    argparser.add_argument('--seed', type=int, help='Seed for random', default=9)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--meta-lr-schedule-steps', type=int, nargs='*', default=[])
    argparser.add_argument('--meta-lr-schedule-values', type=float, nargs='*', default=[])
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.05)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_omniglot_fseml_main")
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--num-workers', type=int, default=4)
    argparser.add_argument('--encoder-channels', type=int, default=32)
    argparser.add_argument('--latent-dim', type=int, default=128)
    argparser.add_argument('--adapter-dim', type=int, default=256)
    argparser.add_argument('--sparsity-target', type=float, default=0.05)
    argparser.add_argument('--sparsity-weight', type=float, default=5e-3)
    argparser.add_argument('--reconstruction-weight', type=float, default=0.1)
    argparser.add_argument('--weight-decay-weight', type=float, default=0.0)
    argparser.add_argument('--fda-weight', type=float, default=0.1)
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    argparser.add_argument("--replay", action="store_true", default=True)
    argparser.add_argument("--replay-mode", default="simple")
    argparser.add_argument("--replay-buffer-size", type=int, default=1000)
    argparser.add_argument("--replay-gap", type=int, default=480)
    argparser.add_argument("--replay-rate", type=float, default=0.1)
    argparser.add_argument("--replay-top-p", type=int, default=32)
    argparser.add_argument("--replay-partitions", type=int, default=10)
    argparser.add_argument("--visualize-replay", action="store_true")
    argparser.add_argument("--replay-viz-dir", default="replay_visualizations")
    argparser.add_argument("--replay-viz-max", type=int, default=10)
    argparser.add_argument('--model', type=str, help='epoch number', default="none")
    args = argparser.parse_args()

    if args.meta_lr_schedule_values:
        if len(args.meta_lr_schedule_values) != len(args.meta_lr_schedule_steps) + 1:
            raise ValueError("--meta-lr-schedule-values must contain exactly len(--meta-lr-schedule-steps) + 1 values")

    args.name = "/".join([args.dataset, str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
