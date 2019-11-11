
import torch

from torch.utils.tensorboard import SummaryWriter


def variable_summaries(name:str, var):
    writer.add_histogram(name, var, iteration)
    mean = var.mean()
    writer.add_scalar('mean/' + name, var.mean(), iteration) 
    stddev = torch.sqrt(torch.mean(torch.sqrt(var - mean)))
    writer.add_scalar('stddev/' + name, stddev, iteration)

class MySummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        super(MySummaryWriter, self).__init__(log_dir, comment, purge_step, 
                                    max_queue,flush_secs, filename_suffix)
        self.global_step = 0
    
    def addStep(self):
        self.global_step += 1

    def variable_summaries(self, tag:str, name:str, var):
        if self.global_step % 500 != 0:
            return
        self.add_histogram(name, var)
        mean = var.mean()
        self.add_scalar(tag+'/mean/' + name, var.mean()) 
        stddev = torch.sqrt(torch.mean(torch.sqrt(var - mean)))
        self.add_scalar(tag+'/stddev/' + name, stddev)

    def add_scalar(self, tag, scalar_value, iteration=None, walltime=None):
        super(MySummaryWriter, self).add_scalar(tag, 
                scalar_value, iteration or self.global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, walltime=None):
        super(MySummaryWriter, self).add_scalars(main_tag, 
                tag_scalar_dict, self.global_step, walltime)
    
    def add_histogram(self, tag, values, bins='tensorflow', walltime=None, max_bins=None):
        super(MySummaryWriter, self).add_histogram(tag, values, self.global_step, bins, walltime, max_bins)
    
    def add_histogram_raw(self, tag, min, max, num, sum, sum_squares,
                          bucket_limits, bucket_counts, walltime=None):
        super(MySummaryWriter, self).add_histogram_raw(tag, min, max,
                num, sum, sum_squares, bucket_limits, bucket_counts, walltime)
    
    def add_image(self, tag, img_tensor, walltime=None, dataformats='CHW'):
        super(MySummaryWriter, self).add_image(tag,  img_tensor, 
                                self.global_step, walltime, dataformats)
    
    def add_images(self, tag, img_tensor, walltime=None, dataformats='NCHW'):
        super(MySummaryWriter, self).add_images(tag, img_tensor, 
                                self.global_step, walltime, dataformats)
    
    def add_image_with_boxes(self, tag, img_tensor, box_tensor,
                             walltime=None, rescale=1, dataformats='CHW'):
        super(MySummaryWriter, self).add_image_with_boxes(tag, img_tensor, box_tensor, 
                                self.global_step, walltime, rescale, dataformats)
    
    def add_figure(self, tag, figure, close=True, walltime=None):
        super(MySummaryWriter, self).add_figure(tag, figure,self.global_step,
                                         close, walltime)

    def add_video(self, tag, vid_tensor, fps=4, walltime=None):
        super(MySummaryWriter, self).add_video(tag, vid_tensor,self.global_step,
                                         fps, walltime)

    def add_audio(self, tag, snd_tensor,  sample_rate=44100, walltime=None):
        super(MySummaryWriter, self).add_audio(tag, snd_tensor,self.global_step,
                                         sample_rate, walltime)

    def add_text(self, tag, text_string, walltime=None):
        super(MySummaryWriter, self).add_text(tag, text_string, self.global_step, walltime)

    def add_embedding(self, mat, metadata=None, label_img=None,  tag='default', metadata_header=None):
        super(MySummaryWriter, self).add_embedding(mat, metadata, label_img, 
                                            self.global_step, tag, metadata_header)

    def add_pr_curve(self, tag, labels, predictions, num_thresholds=127, weights=None, walltime=None):
        super(MySummaryWriter, self).add_pr_curve(tag, labels, predictions, 
                                            self.global_step, num_thresholds, weights, walltime)

    def add_pr_curve_raw(self, tag, true_positive_counts,false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         num_thresholds=127,
                         weights=None,
                         walltime=None):
        super(MySummaryWriter, self).add_pr_curve_raw(tag, true_positive_counts,false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         self.global_step,
                         num_thresholds,
                         weights,
                         walltime)