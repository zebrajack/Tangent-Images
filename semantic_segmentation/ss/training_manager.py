import torch.nn.functional as F

from mapped_convolution.util import *


def visualize_rgb(rgb, stats):
    # Scale back to [0,255]
    mean, std = stats
    device = rgb.get_device()
    mean = torch.tensor(mean).view(-1, 1, 1).to(device)
    std = torch.tensor(std).view(-1, 1, 1).to(device)

    rgb *= std[:3]
    rgb += mean[:3]
    return (255 * rgb).byte()


def iou_score(pred_cls, true_cls, nclass=14, drop=[0]):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []
    for i in range(nclass):
        if i not in drop:
            intersect = ((pred_cls == i).byte() +
                         (true_cls == i).byte()).eq(2).sum().item()
            union = ((pred_cls == i).byte() +
                     (true_cls == i).byte()).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)
    return torch.tensor(intersect_), torch.tensor(union_)


def accuracy(pred_cls, true_cls, nclass=14, drop=[0]):

    positive = torch.histc(true_cls.cpu().float(),
                           bins=nclass,
                           min=0,
                           max=nclass,
                           out=None)
    per_cls_counts = []
    tpos = []
    for i in range(nclass):
        if i not in drop:
            true_positive = ((pred_cls == i).byte() +
                             (true_cls == i).byte()).eq(2).sum().item()
            tpos.append(true_positive)
            per_cls_counts.append(positive[i])

    return torch.tensor(tpos), torch.tensor(per_cls_counts)


class PerspectiveManagerPanoSemSeg(NetworkManager):

    def __init__(self,
                 network,
                 checkpoint_dir,
                 batch_size,
                 path_to_color_map=None,
                 name='',
                 train_dataloader=None,
                 val_dataloader=None,
                 test_dataloader=None,
                 criterion=None,
                 optimizer=None,
                 visdom=None,
                 scheduler=None,
                 num_epochs=20,
                 image_shape=None,
                 base_order=0,
                 max_sample_order=7,
                 data_format='pano',
                 random_sample_size=0,
                 validation_freq=1,
                 visualization_freq=5,
                 evaluation_sample_freq=-1,
                 device=None,
                 drop_unknown=False,
                 stats=None,
                 distributed=False,
                 local_rank=0,
                 train_mode=True):

        super().__init__(network, name, train_dataloader, val_dataloader,
                         test_dataloader, criterion, optimizer, scheduler,
                         num_epochs, validation_freq, visualization_freq,
                         evaluation_sample_freq, device)

        # Directory to store checkpoints
        self.checkpoint_dir = checkpoint_dir

        self.bs = batch_size

        # Mapping of labels to color
        # 14 x 3
        self.color_map = torch.from_numpy(
            np.loadtxt(path_to_color_map, dtype=np.uint8)[:, 1:])

        # 'pano' or perspective ('data') images
        self.data_format = data_format

        if self.data_format == 'pano':
            self.random_sample_size = random_sample_size
            self.num_patches = compute_num_faces(base_order)
            self.base_order = base_order
            self.max_sample_order = max_sample_order
            self.nearest_resample_to_texture = ResampleToUVTexture(
                image_shape, base_order, max_sample_order, 1,
                'nearest').to(device)
            self.bispherical_resample_to_texture = ResampleToUVTexture(
                image_shape, base_order, max_sample_order, 1,
                'bispherical').to(device)
            self.nearest_resample_from_uv_layer = ResampleFromUVTexture(
                image_shape, base_order, max_sample_order, 'nearest').to(device)
            self.bilinear_resample_from_uv_layer = ResampleFromUVTexture(
                image_shape, base_order, max_sample_order,
                'bilinear').to(device)

        # Containers to hold validation results
        self.iou = 0
        self.accuracy = 0
        self.intersections = 0
        self.unions = 0
        self.true_positives = 0
        self.per_cls_counts = 0
        self.count = 0
        self.mean_val_loss = 0

        # Track the best inlier ratio recorded so far
        self.best_pixel_accuracy = 0.0
        self.is_best = False

        # Track the best inlier ratio recorded so far
        self.best_d1_inlier = 0.0
        self.is_best = False

        # List of length 2 [Visdom instance, env]
        self.vis = visdom

        # Loss trackers
        self.loss = AverageMeter()

        self.drop_unknown = drop_unknown
        self.stats = stats
        self.distributed = distributed
        self.local_rank = local_rank
        self.train_mode = train_mode

    def print(self, *args, **kwargs):
        if self.local_rank == 0:
            print(*args, **kwargs)

    def parse_data(self, data, train=True):
        '''
        Returns a list of the inputs as first output, a list of the GT as a second output, and a list of the remaining info as a third output. Must be implemented.
        '''

        # Parse the relevant data
        rgb = data[0].to(self.device)
        labels = data[1].to(self.device)
        basename = data[-1]

        if self.data_format == 'pano':
            self.bs = rgb.shape[0]  # Store number of batches
            rgb, labels = self.pano_data_to_patches(rgb, labels)

            if train:
                # Randomly sample patches
                k = torch.randperm(self.num_patches)[:self.random_sample_size]
                rgb = rgb[k, ...].contiguous()
                labels = labels[k, ...].contiguous()

            # Collapse patches and batches into one dimension
            C, H, W = rgb.shape[-3:]
            rgb = rgb.view(-1, C, H, W)
            labels = labels.view(-1, 1, H, W)

        inputs = rgb
        gt = labels
        other = basename

        return inputs, gt, other

    def pano_data_to_patches(self, rgb=None, gt=None):
        """
        Returns a set of patches

        rgb : B x 3 x H x W
        gt : B x 1 x H x W

        returns
            RGB: N x B x 3 x H x W
            gt: N x B x 1 x H x W
        """
        if rgb is not None:
            rgb = self.bispherical_resample_to_texture(rgb)
            rgb = rgb.permute(2, 0, 1, 3, 4).contiguous()
            rgb = torch.flip(rgb, (-2, )).contiguous()

        if gt is not None:
            gt = self.nearest_resample_to_texture(gt)
            gt = gt.permute(2, 0, 1, 3, 4).contiguous()
            gt = torch.flip(gt, (-2, )).contiguous()

        return rgb, gt

    def compute_labels(self, output, gt_labels):
        """Converts raw outputs to numerical labels"""

        pred_labels = F.softmax(output, dim=1)
        pred_labels = pred_labels.argmax(dim=1, keepdim=True)

        # Mask out invalid areas in predictions
        pred_labels[gt_labels == 0] = 0
        pred_labels[gt_labels == 14] = 14

        return pred_labels

    def train(self, checkpoint_path=None, weights_only=False):
        self.print('Starting training')

        # Load pretrained parameters if desired
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, weights_only)
            if weights_only:
                self.initialize_visualizations()
        else:
            # Initialize any training visualizations
            self.initialize_visualizations()

        # Train for specified number of epochs
        for self.epoch in range(self.epoch, self.num_epochs):

            # Run an epoch of training
            self.train_one_epoch()

            if self.epoch % self.validation_freq == 0:
                self.save_checkpoint()
                self.validate()
                self.visualize_metrics()

            # Increment the LR scheduler
            if self.scheduler is not None:
                self.scheduler.step(self.mean_val_loss)

    def train_one_epoch(self):

        # Put the model in train mode
        self.network.train(mode=self.train_mode)

        # Load data
        end = time.time()
        iteration = 0
        for batch_num, data in enumerate(self.train_dataloader):
            iteration += 1

            # Parse the data into inputs, ground truth, and other
            inputs, gt, other = self.parse_data(data)

            if self.data_format == 'pano':
                # Run a forward pass on each effective batch of the random samples
                N, C, H, W = inputs.shape
                output = self.forward_pass(inputs)

                # Compute the loss
                loss = self.compute_loss(output, gt)

                # Backpropagation of the loss
                self.backward_pass(loss)
                self.loss.update(loss.item(), N)

                # Every few batches
                if batch_num % self.visualization_freq == 0:
                    # Visualize the loss
                    self.visualize_loss(batch_num)
                    self.visualize_training_metrics(batch_num)
                    self.visualize_samples(inputs, gt, other, output)
                    self.print_batch_report(batch_num)

                # Update batch times
                self.batch_time_meter.update(time.time() - end)
                end = time.time()

            else:
                output = self.forward_pass(inputs)
                total_loss = self.compute_loss(output, gt)
                self.backward_pass(total_loss)
                self.loss.update(total_loss.item(), inputs.shape[0])

                # Update batch times
                self.batch_time_meter.update(time.time() - end)
                end = time.time()

                # Every few batches
                if batch_num % self.visualization_freq == 0:

                    # Visualize the loss
                    self.visualize_loss(batch_num)
                    self.visualize_training_metrics(batch_num)
                    self.visualize_samples(inputs, gt, other, output)
                    self.print_batch_report(batch_num)



    def forward_pass(self, inputs):
        '''
        Returns the network output
        '''
        output = self.network(inputs)
        if isinstance(output, dict):
            output = output['out']
        if self.drop_unknown:
            output[:, 0] = -1e8
        return output

    def validate(self):
        self.print('Validating model....')

        # Put the model in eval mode
        self.network.eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            cnt = 0
            for batch_num, data in enumerate(self.test_dataloader):
                self.print('Validating batch {}/{}'.format(
                    batch_num, len(self.test_dataloader)),
                      end='\r')

                # Parse the data
                inputs, gt, other = self.parse_data(data, False)

                if self.data_format == 'pano':
                    # Run a forward pass on each pass separately
                    # (necessary due to BatchNorm)
                    output = self.forward_pass(inputs)
    
                    if self.distributed:
                        K = torch.distributed.get_world_size()
                        output_all = [torch.empty_like(output)
                                      for _ in range(K)]
                        gt_all = [torch.empty_like(gt)
                                      for _ in range(K)]
                        torch.distributed.all_gather(output_all, output)
                        torch.distributed.all_gather(gt_all, gt)
                    if self.local_rank == 0:
                        if self.distributed:
                            output = torch.cat(output_all, 0)
                            gt = torch.cat(gt_all, 0)

                        # Compute validation loss
                        val_loss = self.compute_loss(output, gt, True)
                        self.mean_val_loss += val_loss
                        cnt += 1

                        # Compute the evaluation metrics
                        self.compute_eval_metrics(output, gt)

                elif self.data_format == 'data':
                    output = self.forward_pass(inputs)

                    if self.distributed:
                            K = torch.distributed.get_world_size()
                            output_all = [torch.empty_like(output)
                                          for _ in range(K)]
                            gt_all = [torch.empty_like(gt)
                                          for _ in range(K)]
                            torch.distributed.all_gather(output_all, output)
                            torch.distributed.all_gather(gt_all, gt)


                    if self.local_rank == 0:
                        if self.distributed:
                            output = torch.cat(output_all, 0)
                            gt = torch.cat(gt_all, 0)
                        # Compute validation loss
                        val_loss = self.compute_loss(output, gt, True)
                        self.mean_val_loss += val_loss
                        cnt += 1

                        # Compute the evaluation metrics
                        self.compute_eval_metrics(output, gt)

        # Print a report on the validation results
        self.print('Validation finished in {} seconds'.format(time.time() - s))
        self.print_evaluation_report()
        self.mean_val_loss /= max(cnt, 1)

    def compute_loss(self, output, gt, disable_running_mean=False):
        '''
        Returns a loss as a list where the first element is the total loss
        '''
        if self.drop_unknown:
            # NB: criterion must ignore -1 in this case, and not 0
            output = output[:, 1:]
            gt = gt - 1

        loss = self.criterion(output, gt.squeeze(1).long())

        if not disable_running_mean:
            self.loss.update(loss.item(), output.shape[0])
        return loss

    def reset_eval_metrics(self):
        '''
        Resets metrics used to evaluate the model
        '''
        self.iou = 0
        self.accuracy = 0
        self.intersections = 0
        self.unions = 0
        self.true_positives = 0
        self.per_cls_counts = 0
        self.count = 0
        self.is_best = False
        self.mean_val_loss = 0

    def compute_eval_metrics(self, output, gt):
        '''
        Computes metrics used to evaluate the model
        '''
        gt_labels = gt.long()
        pred_labels = self.compute_labels(output, gt_labels)

        # Compute scores
        int_, uni_ = iou_score(pred_labels, gt_labels)
        true_positive, per_class_count = accuracy(pred_labels, gt_labels)
        self.intersections += int_.float()
        self.unions += uni_.float()
        self.true_positives += true_positive.float()
        self.per_cls_counts += per_class_count.float()
        self.count += output.shape[0]

    def load_checkpoint(self, checkpoint_path=None, weights_only=False):
        '''
        Initializes network and/or optimizer with pretrained parameters
        '''
        if checkpoint_path is not None:
            print('Loading checkpoint \'{}\''.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            # If we want to continue training where we left off, load entire training state
            if not weights_only:
                self.epoch = checkpoint['epoch']
                experiment_name = checkpoint['experiment']
                self.best_d1_inlier = checkpoint['best_vertex_d1_inlier']
                self.loss.from_dict(checkpoint['loss_meter'])
            else:
                print('NOTE: Loading weights only')

            # Load the optimizer and model state
            load_optimizer(self.optimizer, checkpoint['optimizer'], self.device)
            load_partial_model(self.network, checkpoint['state_dict'])

            print('Loaded checkpoint \'{}\' (epoch {})'.format(
                checkpoint_path, checkpoint['epoch']))
        else:
            print('WARNING: No checkpoint found')

    def initialize_visualizations(self):
        '''
        Initializes visualizations
        '''

        if self.local_rank != 0:
            return

        self.vis.line(X=torch.zeros(1, 1).long(),
                      Y=torch.zeros(1, 1).float(),
                      win='losses',
                      opts=dict(title='Loss Plot',
                                markers=False,
                                xlabel='Iteration',
                                ylabel='Loss',
                                legend=['Total Loss']))

        self.vis.line(X=torch.zeros(1, 2).long(),
                      Y=torch.zeros(1, 2).float(),
                      win='error_metrics',
                      opts=dict(title='Semantic Segmentation Error Metrics',
                                markers=True,
                                xlabel='Epoch',
                                ylabel='Error',
                                legend=['Mean IOU', 'Mean Class Accuracy']))

        self.vis.line(X=torch.zeros(1, 1).long(),
                      Y=torch.zeros(1, 1).float(),
                      win='val_loss',
                      opts=dict(title='Mean Validation Loss Plot',
                                markers=False,
                                xlabel='Epoch',
                                ylabel='Loss',
                                legend=['Mean Validation Loss']))

    def visualize_loss(self, batch_num):
        '''
        Updates the visdom display with the loss
        '''
        if self.local_rank != 0:
            return

        total_num_batches = self.epoch * len(self.train_dataloader) + batch_num
        self.vis.line(X=torch.tensor([total_num_batches]),
                      Y=torch.tensor([self.loss.avg]),
                      win='losses',
                      update='append',
                      opts=dict(legend=['Total Loss']))

    def visualize_samples(self, inputs, gt, other, output):
        '''
        Updates the visdom display with samples
        '''
        if self.local_rank != 0:
            return

        # Parse loaded data
        rgb = inputs[:, :3, :, :]
        gt_labels = gt.long()
        pred_labels = self.compute_labels(output, gt_labels)
        basename = other

        # Data is N x C x H x W
        # Outputs are N x C x H x W
        if self.data_format == 'pano':
            k = torch.randperm(rgb.shape[0])[:16]
            rgb_img = visualize_rgb(rgb[k, ...], self.stats).cpu()
            pred_img = self.color_map[
                pred_labels[k, ...].squeeze(1)].byte().permute(
                    0, 3, 1, 2).cpu()
            gt_img = self.color_map[gt_labels[k, ...].squeeze(1)].byte().permute(
                0, 3, 1, 2).cpu()
        else:
            rgb_img = visualize_rgb(rgb, self.stats).cpu()
            gt_img = self.color_map[gt_labels.squeeze(1)].byte().cpu().permute(
                0, 3, 1, 2)
            pred_img = self.color_map[
                pred_labels.squeeze(1)].byte().cpu().permute(0, 3, 1, 2)

        self.vis.images(rgb_img,
                        win='rgb',
                        opts=dict(title='Input RGB Image',
                                  caption='Input RGB Image'))

        self.vis.images(gt_img,
                        win='gt',
                        opts=dict(title='Ground Truth Segmentation',
                                  caption='Ground Truth Segmentation'))

        self.vis.images(pred_img,
                        win='output',
                        opts=dict(title='Output Segmentation',
                                  caption='Output Segmentation'))

    def visualize_metrics(self):
        '''
        Updates the visdom display with the metrics
        '''
        if self.local_rank != 0:
            return

        miou = self.iou.mean()
        mca = self.accuracy.mean()

        errors = torch.tensor([miou, mca])
        errors = errors.view(1, -1)
        epoch_expanded = torch.ones(errors.shape) * (self.epoch + 1)
        self.vis.line(X=epoch_expanded,
                      Y=errors,
                      win='error_metrics',
                      update='append',
                      opts=dict(legend=['Mean IOU', 'Mean Class Accuracy']))

        val_loss = torch.tensor([self.mean_val_loss])
        val_loss = val_loss.view(1, -1)
        epoch_expanded = torch.ones(val_loss.shape) * (self.epoch + 1)
        self.vis.line(X=epoch_expanded,
                      Y=val_loss,
                      win='val_loss',
                      update='append',
                      opts=dict(legend=['Mean Validation Loss']))

        self.vis.bar(X=self.iou,
                     win='class_iou',
                     opts=dict(title='Per-Class IOU'))

        self.accuracy[np.isnan(self.accuracy)] = 0
        self.vis.bar(X=self.accuracy,
                     win='class_accuracy',
                     opts=dict(title='Per-Class Accuracy'))

    def print_batch_report(self, batch_num):
        '''
        Prints a report of the current batch
        '''
        if self.local_rank != 0:
            return

        print('Epoch: [{0}][{1}/{2}]\t'
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\n\n'.format(
                  self.epoch + 1,
                  batch_num + 1,
                  len(self.train_dataloader),
                  batch_time=self.batch_time_meter,
                  loss=self.loss))

    def print_evaluation_report(self):
        '''
        Prints a report of the evaluation results
        '''
        if self.local_rank != 0:
            return

        self.iou = self.intersections / torch.clamp_min(self.unions, 1e-6)
        self.accuracy = self.true_positives / torch.clamp_min(self.per_cls_counts, 1e-6)
        # self.iou = self.intersections / self.unions
        # self.accuracy = self.true_positives / self.per_cls_counts
        mean_acc = self.accuracy.mean()

        print('Epoch: {}\n'
              '  Avg. IOU: {:.4f}\n'
              '  Avg. Pixel Accuracy: {:.4f}\n\n'.format(
                  self.epoch + 1, self.iou.mean(), mean_acc))

        print('Per-Class IOU')
        print(self.iou)

        print('Per-Class Accuracy')
        print(self.accuracy)

        # Also update the best state tracker
        if self.best_pixel_accuracy < mean_acc:
            self.best_pixel_accuracy = mean_acc
            self.is_best = True

    def save_checkpoint(self):
        '''
        Saves the model state
        '''
        if self.local_rank != 0:
            return

        # Save latest checkpoint (constantly overwriting itself)
        checkpoint_path = osp.join(self.checkpoint_dir, 'checkpoint_latest.pth')

        # Actually saves the latest checkpoint and also updating the file holding the best one
        save_checkpoint(
            {
                'epoch': self.epoch + 1,
                'experiment': self.name,
                'state_dict': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss_meter': self.loss.to_dict(),
                'best_vertex_d1_inlier': self.best_d1_inlier
            },
            self.is_best,
            filename=checkpoint_path)

        # Copies the latest checkpoint to another file stored for each epoch
        history_path = osp.join(self.checkpoint_dir,
                                'checkpoint_{:03d}.pth'.format(self.epoch + 1))
        shutil.copyfile(checkpoint_path, history_path)
        print('Checkpoint saved')

    def save_samples(self, inputs, gt, other, output):

        if self.sample_dir is None:
            print('No sample directory specified')
            return