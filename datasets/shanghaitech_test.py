from glob import glob
from os.path import basename
from os.path import isdir
from os.path import join
from typing import List
from typing import Tuple




import cv2
import torch
import numpy as np
import skimage.io as io
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from skimage.transform import resize
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from .base import VideoAnomalyDetectionDataset, ToFloatTensor3D, ToFloatTensor3DMask 

class ShanghaiTechTestHandler(VideoAnomalyDetectionDataset):

    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.
        :param path: The folder in which ShanghaiTech is stored.
        """
        super(ShanghaiTechTestHandler, self).__init__()
        self.path = path
        # Test directory
        self.test_dir = join(path, 'testing')
        # Transform
        self.transform = transforms.Compose([ToFloatTensor3D(normalize=True)])
        # Load all test ids
        self.test_ids = self.load_test_ids()
        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        self.cur_video_gt = None

    def load_test_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.
        :return: The list of test ids.
        """
        return sorted([basename(d) for d in glob(join(self.test_dir, 'nobackground_frames_resized', '**')) if isdir(d)])

    def load_test_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.
        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.shape
        sequence_dir = join(self.test_dir,  'nobackground_frames_resized', video_id)
        img_list = sorted(glob(join(sequence_dir, '*.jpg')))
        #print(f"Creating clips for {sequence_dir} dataset with length {t}...")
        return np.stack([np.uint8(io.imread(img_path)) for img_path in img_list])

    def load_test_sequence_gt(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads the groundtruth of a test video in memory.
        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        clip_gt = np.load(join(self.test_dir,  'test_frame_mask', f'{video_id}.npy'))
        return clip_gt

    def test(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.
        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.shape
        self.cur_video_id = video_id
        self.cur_video_frames = self.load_test_sequence_frames(video_id)
        self.cur_video_gt = self.load_test_sequence_gt(video_id)
        self.cur_len = len(self.cur_video_frames) - t + 1

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples being fed to the model.
        """
        return 3, 16, 256, 512

    @property
    def test_videos(self):
        # type: () -> List[str]
        """
        Returns all available test videos.
        """
        return self.test_ids

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return self.cur_len

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Provides the i-th example.
        """
        c, t, h, w = self.shape
        clip = self.cur_video_frames[i:i+t]
        sample = clip
        # Apply transform
        if self.transform:
            sample = self.transform(sample)
        return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return default_collate

    def __repr__(self):
        return f'ShanghaiTech (video id = {self.cur_video_id})'

def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x


class ResultsAccumulator:
    """
    Accumulates results in a buffer for a sliding window
    results computation. Employed to get frame-level scores
    from clip-level scores.
    ` In order to recover the anomaly score of each
    frame, we compute the mean score of all clips in which it
    appears`
    """
    def __init__(self, nb_frames_per_clip):
        # type: (int) -> None
        """
        Class constructor.
        :param nb_frames_per_clip: the number of frames each clip holds.
        """

        # This buffers rotate.
        self._buffer = np.zeros(shape=(nb_frames_per_clip,), dtype=np.float32)
        self._counts = np.zeros(shape=(nb_frames_per_clip,))

    def push(self, score):
        # type: (float) -> None
        """
        Pushes the score of a clip into the buffer.
        :param score: the score of a clip
        """

        # Update buffer and counts
        self._buffer += score
        self._counts += 1

    def get_next(self):
        # type: () -> float
        """
        Gets the next frame (the first in the buffer) score,
        computed as the mean of the clips in which it appeared,
        and rolls the buffers.
        :return: the averaged score of the frame exiting the buffer.
        """

        # Return first in buffer
        ret = self._buffer[0] / self._counts[0]
        
        # Roll time backwards
        self._buffer = np.roll(self._buffer, shift=-1)
        self._counts = np.roll(self._counts, shift=-1)

        # Zero out final frame (next to be filled)
        self._buffer[-1] = 0
        self._counts[-1] = 0
        
        return ret

    @property
    def results_left(self):
        # type: () -> np.int32
        """
        Returns the number of frames still in the buffer.
        """
        return np.sum(self._counts != 0).astype(np.int32)


class VideoAnomalyDetectionResultHelper(object):
    """
    Performs tests for video anomaly detection datasets (UCSD Ped2 or Shanghaitech).
    """

    def __init__(self, dataset, model, c, R, boundary, device, end_to_end_training, debug, output_file):
        # type: (VideoAnomalyDetectionDataset, BaseModule, str, str) -> None
        """
        Class constructor.
        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param output_file: text file where to save results.
        """
        self.dataset = dataset
        self.model = model
        self.hc = c
        self.keys = list(c.keys())
        self.R = R
        self.boundary = boundary
        self.device = device
        self.end_to_end_training = end_to_end_training
        self.debug = debug
        self.output_file = output_file

    def _get_scores(self, d_lstm):
        # Eval novelty scores
        dist = {k: torch.sum((d_lstm[k] - self.hc[k].unsqueeze(0)) ** 2, dim=1) for k in self.keys}
        scores = {k: torch.zeros((dist[k].shape[0],), device=self.device) for k in self.keys}
        overall_score = torch.zeros((dist[self.keys[0]].shape[0],), device=self.device)
        for k in self.keys:
            if self.boundary == 'soft':
                scores[k] += dist[k] - self.R[k] ** 2
                overall_score += dist[k] - self.R[k] ** 2
            else:
                scores[k] += dist[k]
                overall_score += dist[k]
        scores = {k: scores[k]/len(self.keys) for k in self.keys}
        return scores, overall_score/len(self.keys)

    @torch.no_grad()
    def test_video_anomaly_detection(self):
        # type: () -> None
        """
        Actually performs tests.
        """
        self.model.eval().to(self.device)

        c, t, h, w = self.dataset.raw_shape

        # Prepare a table to show results
        vad_table = self.empty_table

        # Set up container for anomaly scores from all test videos
        ## oc: one class
        ## rc: reconstruction
        ## as: overall anomaly score
        global_oc = []
        global_rc = []
        global_as = []
        global_as_by_layer = {k: [] for k in self.keys}
        global_y = []
        global_y_by_layer = {k: [] for k in self.keys}

        # Get accumulators
        results_accumulator_rc = ResultsAccumulator(nb_frames_per_clip=t)
        results_accumulator_oc = ResultsAccumulator(nb_frames_per_clip=t)
        results_accumulator_oc_by_layer = {k: ResultsAccumulator(nb_frames_per_clip=t) for k in self.keys}
        print(self.dataset.test_videos)
        # Start iteration over test videos
        for cl_idx, video_id in tqdm(enumerate(self.dataset.test_videos), total=len(self.dataset.test_videos), desc="Test on Video"):
            # Run the test
            self.dataset.test(video_id)
            loader = DataLoader(self.dataset, collate_fn=self.dataset.collate_fn)

            # Build score containers
            sample_rc = np.zeros(shape=(len(loader) + t - 1,))
            sample_oc = np.zeros(shape=(len(loader) + t - 1,))
            sample_oc_by_layer = {k: np.zeros(shape=(len(loader) + t - 1,)) for k in self.keys}
            sample_y = self.dataset.load_test_sequence_gt(video_id)
        
            for i, x in tqdm(enumerate(loader), total=len(loader), desc=f'Computing scores for {self.dataset}', leave=False):
                # x.shape = [1, 3, 16, 256, 512]
                x = x.to(self.device)

                if self.end_to_end_training:
                    x_r, _, d_lstm = self.model(x)
                    recon_loss = torch.mean(torch.sum((x_r - x) ** 2, dim=tuple(range(1, x_r.dim()))))
                else:
                    _, d_lstm = self.model(x)
                    recon_loss = torch.tensor([0.0])
                
                # Eval one class score for current clip
                oc_loss_by_layer, oc_overall_loss = self._get_scores(d_lstm)

                # Feed results accumulators
                results_accumulator_rc.push(recon_loss.item())
                sample_rc[i] = results_accumulator_rc.get_next()
                results_accumulator_oc.push(oc_overall_loss.item())
                sample_oc[i] = results_accumulator_oc.get_next()

                for k in self.keys:
                    if k != "tdl_lstm_o_0" and k != "tdl_lstm_o_1":
                        results_accumulator_oc_by_layer[k].push(oc_loss_by_layer[k].item())
                        sample_oc_by_layer[k][i] = results_accumulator_oc_by_layer[k].get_next()

            # Get last results layer by layer
            for k in self.keys:
                if k != "tdl_lstm_o_0" and k != "tdl_lstm_o_1":
                    while results_accumulator_oc_by_layer[k].results_left != 0:
                        index = (- results_accumulator_oc_by_layer[k].results_left)
                        sample_oc_by_layer[k][index] = results_accumulator_oc_by_layer[k].get_next()
                    
                    min_, max_ = sample_oc_by_layer[k].min(), sample_oc_by_layer[k].max()

                    # Computes the normalized novelty score given likelihood scores, reconstruction scores
                    # and normalization coefficients (Eq. 9-10).
                    sample_ns = (sample_oc_by_layer[k] - min_) / (max_ - min_)
                    
                    # Update global scores (used for global metrics)
                    global_as_by_layer[k].append(sample_ns)
                    global_y_by_layer[k].append(sample_y)

                    try:
                        # Compute AUROC for this video
                        this_video_metrics = [
                            roc_auc_score(sample_y, sample_ns),    # anomaly score == one class metric
                            0.,
                            0.
                        ]
                        #vad_table.add_row([k] + [video_id] + this_video_metrics)
                    except ValueError:
                        # This happens for sequences in which all frames are abnormal
                        # Skipping this row in the table (the sequence will still count for global metrics)
                        continue
            
            # Get last results
            while results_accumulator_oc.results_left != 0:
                index = (- results_accumulator_oc.results_left)
                sample_oc[index] = results_accumulator_oc.get_next()
                sample_rc[index] = results_accumulator_rc.get_next()
                
            min_oc, max_oc, min_rc, max_rc = sample_oc.min(), sample_oc.max(), sample_rc.min(), sample_rc.max()

            # Computes the normalized novelty score given likelihood scores, reconstruction scores
            # and normalization coefficients (Eq. 9-10).
            sample_oc = (sample_oc - min_oc) / (max_oc - min_oc)
            sample_rc = (sample_rc - min_rc) / (max_rc - min_rc) if (max_rc - min_rc) > 0 else np.zeros_like(sample_rc)
            sample_as = sample_oc + sample_rc
            
            # Update global scores (used for global metrics)
            global_oc.append(sample_oc)
            global_rc.append(sample_rc)
            global_as.append(sample_as)
            global_y.append(sample_y)

            try:
                # Compute AUROC for this video
                this_video_metrics = [
                    roc_auc_score(sample_y, sample_oc),   # one class metric
                    roc_auc_score(sample_y, sample_rc),   # reconstruction metric
                    roc_auc_score(sample_y, sample_as)    # anomaly score
                ]
                #vad_table.add_row(['Overall'] + [video_id] + this_video_metrics)
            except ValueError:
                # This happens for sequences in which all frames are abnormal
                # Skipping this row in the table (the sequence will still count for global metrics)
                continue

            if self.debug: break

        # Compute global AUROC and print table
        for k in self.keys:
            if k != "tdl_lstm_o_0" and k != "tdl_lstm_o_1":
                global_as_by_layer[k] = np.concatenate(global_as_by_layer[k])
                global_y_by_layer[k] = np.concatenate(global_y_by_layer[k])
                global_metrics = [
                    roc_auc_score(global_y_by_layer[k], global_as_by_layer[k]),    # anomaly score == one class metric
                    0.,
                    0.
                ]
                vad_table.add_row([k] + ['avg'] + global_metrics)

        # Compute global AUROC and print table
        global_oc = np.concatenate(global_oc)
        global_rc = np.concatenate(global_rc)
        global_as = np.concatenate(global_as)
        global_y = np.concatenate(global_y)
        global_metrics = [
            roc_auc_score(global_y, global_oc),   # one class metric
            roc_auc_score(global_y, global_rc),   # reconstruction metric
            roc_auc_score(global_y, global_as)    # anomaly score
        ]

        vad_table.add_row(['Overall'] + ['avg'] + global_metrics)
        print(vad_table)

        # Save table
        with open(self.output_file, mode='w') as f:
            f.write(str(vad_table))

    @property
    def empty_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the video anomaly detection setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        table.field_names = ['Layer key', 'VIDEO-ID', 'OC metric', 'Recon metric', 'AUROC-AS']
        table.float_format = '0.3'
        return table
