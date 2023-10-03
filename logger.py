import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision
import random


class Logger(object):
    def __init__(self, args, exp_config, log_dir, log_hist=True):
        self.config = exp_config
        """Create a summary writer logging to log_dir."""
        if log_hist:    # Check a new folder for each log should be dreated
            if args['model'] == 'GCN': 
                log_dir = os.path.join(log_dir,
                                   args['model']+'-'\
                              +str(self.config['lr'])+'-'\
                              +str(self.config['batch_size'])+'-' \
                              +str(self.config['dropout'])+'-' \
                              +'(' \
                              +str(self.config['gnn_hidden_feats'])+',' \
                              +str(self.config['num_gnn_layers'])+')-' \
                              +'(' \
                              +str(self.config['gnn_hidden_feats2'])+',' \
                              +str(self.config['num_gnn_layers2'])+')' \
                              +'-' \
                              +str(self.config['predictor_hidden_feats'])+'_(' \
                              +datetime.datetime.now().strftime("%M")+'-' \
                              +datetime.datetime.now().strftime("%S")+')')
            if args['model'] == 'MPNN': 
                log_dir = os.path.join(log_dir,
                                  args['model']+'-'\
                             +str(self.config['lr'])+'-' \
                             +str(self.config['batch_size'])+'-' \
                             +str(self.config['node_out_feats'])+'-' \
                             +str(self.config['edge_hidden_feats'])+'-' \
                             +str(self.config['num_step_message_passing'])+'-' \
                             +str(self.config['num_step_set2set'])+'-' \
                             +str(self.config['num_layer_set2set'])+'_' \
                             +datetime.datetime.now().strftime("%M"))

        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def list_of_image(self, tag, images, step):
        """Log scalar variables."""
        grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, grid, step)
