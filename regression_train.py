# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from dgllife.utils import Meter, EarlyStopping
from hyperopt import fmin, tpe
from shutil import copyfile
from torch.optim import Adam
from torch.utils.data import DataLoader

from hyper import init_hyper_space
from utils import get_configure, mkdir_p, init_trial_path, \
    split_dataset, collate_molgraphs, load_model, predict, init_featurizer, load_dataset


from logger import Logger
import csv

from dgl.nn.pytorch import WeightAndSum


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(args['device']), masks.to(args['device'])
        prediction = predict(args, model, bg)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)

    logger.scalar_summary("Loss/Training", loss.item(), epoch+1)

    return train_meter.compute_metric(args['metric'])

#dump data
def dump_data(args, model, data_loader, flag):
    if flag == 0:
        filename = args['trial_path'] + '/train_'
    if flag == 1:
        filename = args['trial_path'] + '/valid_'
    if flag == 2:
        filename = args['trial_path'] + '/test_'

    smiles = []
    labels = []
    predictions = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            batch_smiles, bg, batch_labels, masks = batch_data
            batch_labels, masks = batch_labels.to(args['device']), masks.to(args['device'])
            batch_pred = predict(args, model, bg)

            smiles = smiles+batch_smiles

            labels.append(batch_labels.detach().cpu())
            predictions.append(batch_pred.detach().cpu())

    labels = torch.cat(labels, dim=0)
    predictions = torch.cat(predictions, dim=0)

    output_gt = np.concatenate([np.array(smiles).reshape(-1,1),labels],axis=1)
    output_pd = np.concatenate([np.array(smiles).reshape(-1,1),predictions],axis=1)

    with open(filename+'gt.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['SMILES','HOMO','LUMO','S1','SI'])
        for row in output_gt:
            w.writerow(row)
    with open(filename+'pd.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['SMILES','HOMO','LUMO','S1','SI'])
        for row in output_pd:
            w.writerow(row)

def run_an_eval_epoch(args, epoch,  model, data_loader, loss_criterion):
    model.eval()
    eval_meter = Meter()

    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            #labels = labels.to(args['device'])
            labels, masks = labels.to(args['device']), masks.to(args['device'])
            prediction = predict(args, model, bg)
            loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
            eval_meter.update(prediction, labels, masks)

    logger.scalar_summary("Loss/Validation", loss.item(), epoch+1)

    return eval_meter.compute_metric(args['metric'])

def main(args, exp_config, train_set, val_set, test_set):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })
    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()

    # Set up directory for saving results
    args = init_trial_path(args, exp_config)


    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    model = load_model(exp_config).to(args['device'])

    loss_criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                     weight_decay=exp_config['weight_decay'])

    stopper = EarlyStopping(patience=exp_config['patience'],
                            filename=args['trial_path'] + '/model.pth',
                            metric=args['metric'])




    for epoch in range(args['num_epochs']):
        train_score = run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        val_score   = run_an_eval_epoch(args, epoch, model, val_loader, loss_criterion)
        early_stop = stopper.step(np.mean(val_score), model)

        print('Epoch {:d}/{:d}, training {} {:.4f}, validation {} {:.4f}, best validation {} {:.4f}'.format(epoch+1, args['num_epochs'],
                                args['metric'], np.mean(train_score),
                                args['metric'], np.mean(val_score),
                                args['metric'], stopper.best_score))

        logger.scalar_summary("Training r2/HOMO", train_score[0], epoch+1)
        logger.scalar_summary("Training r2/LUMO", train_score[1], epoch+1)
        logger.scalar_summary("Training r2/S1", train_score[2], epoch+1)
        logger.scalar_summary("Training r2/SI", train_score[3], epoch+1)

        logger.scalar_summary("Validation r2/HOMO", val_score[0], epoch+1)
        logger.scalar_summary("Validation r2/LUMO", val_score[1], epoch+1)
        logger.scalar_summary("Validation r2/S1", val_score[2], epoch+1)
        logger.scalar_summary("Validation r2/SI", val_score[3], epoch+1)


        if early_stop:
            break

    stopper.load_checkpoint(model)

    test_score = run_an_eval_epoch(args, epoch, model, test_loader, loss_criterion)
    print('test {} {:.4f}'.format(args['metric'], np.mean(test_score)))


    dump_data(args, model, train_loader, 0)
    dump_data(args, model, val_loader, 1)
    dump_data(args, model, test_loader, 2)


    with open(args['trial_path'] + '/eval.txt', 'w') as f:
        f.write('Best val {}: {}\n'.format(args['metric'], stopper.best_score))
        f.write('Test {}: {}\n'.format(args['metric'], np.mean(test_score)))

    with open(args['trial_path'] + '/configure.json', 'w') as f:
        json.dump(exp_config, f, indent=2)

    return args['trial_path'], stopper.best_score

def bayesian_optimization(args, train_set, val_set, test_set):
    # Run grid search
    results = []

    candidate_hypers = init_hyper_space(args['model'])

    def objective(hyperparams):
        configure = deepcopy(args)
        trial_path, val_metric = main(configure, hyperparams, train_set, val_set, test_set)

        if args['metric'] in ['r2']:
            # Maximize R2 is equivalent to minimize the negative of it
            val_metric_to_minimize = -1 * val_metric
        else:
            val_metric_to_minimize = val_metric

        results.append((trial_path, val_metric_to_minimize))

        return val_metric_to_minimize

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=args['num_evals'])
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric = results[0]

    return best_trial_path

if __name__ == '__main__':
    from argparse import ArgumentParser


    parser = ArgumentParser('(Multitask) Regression')
    parser.add_argument('-c', '--csv-path', type=str, required=True,
                        help='Path to a csv file for loading a dataset')
    parser.add_argument('-sc', '--smiles-column', type=str, required=True,
                        help='Header for the SMILES column in the CSV file')
    parser.add_argument('-lv', '--log-values', action='store_true', default=False,
                        help='Whether to take logarithm of the labels for modeling')
    parser.add_argument('-t', '--task-names', default=None, type=str,
                        help='Header for the tasks to model. If None, we will model '
                             'all the columns except for the smiles_column in the CSV file. '
                             '(default: None)')
    parser.add_argument('-s', '--split',
                        choices=['scaffold_decompose', 'scaffold_smiles', 'random', 'consecutive'],
                        default='scaffold_smiles',
                        help='Dataset splitting method (default: scaffold_smiles). For scaffold '
                             'split based on rdkit.Chem.AllChem.MurckoDecompose, '
                             'use scaffold_decompose. For scaffold split based on '
                             'rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles, '
                             'use scaffold_smiles.')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-me', '--metric', choices=['r2', 'mae', 'rmse'], default='r2',
                        help='Metric for evaluation (default: r2)')
    parser.add_argument('-mo', '--model', choices=['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP',
                                                   'gin_supervised_contextpred',
                                                   'gin_supervised_infomax',
                                                   'gin_supervised_edgepred',
                                                   'gin_supervised_masking',
                                                   'NF'],
                        default='GCN', help='Model to use (default: GCN)')
    parser.add_argument('-a', '--atom-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for atoms (default: canonical)')
    parser.add_argument('-b', '--bond-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for bonds (default: canonical)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs allowed for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-nw', '--num-workers', type=int, default=1,
                        help='Number of processes for data loading (default: 1)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-p', '--result-path', type=str, default='regression_results',
                        help='Path to save training results (default: regression_results)')
    parser.add_argument('-ne', '--num-evals', type=int, default=None,
                        help='Number of trials for hyperparameter search (default: None)')

    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    if args['task_names'] is not None:
        args['task_names'] = args['task_names'].split(',')

    args = init_featurizer(args)
    df = pd.read_csv(args['csv_path'])
    mkdir_p(args['result_path'])
    dataset = load_dataset(args, df)

    # Whether to take the logarithm of labels for narrowing the range of values
    if args['log_values']:
        dataset.labels = dataset.labels.log()
    args['n_tasks'] = dataset.n_tasks


    train_set, val_set, test_set = split_dataset(args, dataset)


    if args['num_evals'] is not None:
        assert args['num_evals'] > 0, 'Expect the number of hyperparameter search trials to ' \
                                      'be greater than 0, got {:d}'.format(args['num_evals'])
        print('Start hyperparameter search with Bayesian '
              'optimization for {:d} trials'.format(args['num_evals']))
        trial_path = bayesian_optimization(args, train_set, val_set, test_set)
    else:
        print('Use the manually specified hyperparameters')
        exp_config = get_configure(args['model'])
        logger = Logger(args, exp_config, './')
        main(args, exp_config, train_set, val_set, test_set)

