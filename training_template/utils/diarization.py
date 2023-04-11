from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np
import librosa
import torch
import torch.nn as nn

from ..metrics.der import calculate_metrics

def diarization_collate_fn(batch):
    return {
        'xs': [x for x, _ in batch],
        'ts': [t for _, t in batch]
    }

def get_labeledSTFT(
    kaldi_obj,
    start: int,
    end: int,
    frame_size: int,
    frame_shift: int,
    n_speakers: int = None,
    use_speaker_id: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts STFT and corresponding diarization labels for
    given recording id and start/end times
    Args:
        kaldi_obj (KaldiData)
        rec (str): recording id
        start (int): start frame index
        end (int): end frame index
        frame_size (int): number of samples in a frame
        frame_shift (int): number of shift samples
        n_speakers (int): number of speakers
            if None, the value is given from data
    Returns:
        Y: STFT
            (n_frames, n_bins)-shape np.complex64 array,
        T: label
            (n_frames, n_speakers)-shape np.int32 array.
    """
    data=kaldi_obj[0]
    rate=kaldi_obj[1]
    Y = stft(data, frame_size, frame_shift)
    speakers = np.unique(kaldi_obj[4]).tolist()
    if n_speakers is None:
        n_speakers = len(speakers)
    T = np.zeros((Y.shape[0], n_speakers), dtype=np.int32)
    spk_index=[]
    for (start_id,end_id,speaker_id) in zip(kaldi_obj[2], kaldi_obj[3], kaldi_obj[4]):

        start_frame = np.rint(
            start_id * rate / frame_shift).astype(int)
        end_frame = np.rint(
            end_id * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            if rel_end is None:
                rel_end = T.shape[0]
            if rel_start is None:
                rel_start=0
            if rel_end-rel_start<=0:continue
            if speaker_id not in spk_index:
                spk_index.append(speaker_id)
            speaker_index=spk_index.index(speaker_id)
            T[rel_start:rel_end, speaker_index] = 1


    return Y, T[:,:len(spk_index)]


def splice(Y: np.ndarray, context_size: int = 0) -> np.ndarray:
    """ Frame splicing
    Args:
        Y: feature
            (n_frames, n_featdim)-shape numpy array
        context_size:
            number of frames concatenated on left-side
            if context_size = 5, 11 frames are concatenated.
    Returns:
        Y_spliced: spliced feature
            (n_frames, n_featdim * (2 * context_size + 1))-shape
    """
    Y_pad = np.pad(
        Y,
        [(context_size, context_size), (0, 0)],
        'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(Y_pad),
        (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
        (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced


def stft(
    data: np.ndarray,
    frame_size: int,
    frame_shift: int
) -> np.ndarray:
    """ Compute STFT features
    Args:
        data: audio signal
            (n_samples,)-shape np.float32 array
        frame_size: number of samples in a frame (must be a power of two)
        frame_shift: number of samples between frames
    Returns:
        stft: STFT frames
            (n_frames, n_bins)-shape np.complex64 array
    """
    # round up to nearest power of 2
    fft_size = 1 << (frame_size - 1).bit_length()
    # HACK: The last frame is omitted
    #       as librosa.stft produces such an excessive frame
    if len(data) % frame_shift == 0:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T[:-1]
    else:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T


def subsample(
    Y: np.ndarray,
    T: np.ndarray,
    subsampling: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """ Frame subsampling
    """
    Y_ss = Y[::subsampling]
    T_ss = T[::subsampling]
    return Y_ss, T_ss


def transform(
    Y: np.ndarray,
    sampling_rate: int,
    feature_dim: int,
    transform_type: str,
    dtype: type = np.float32,
) -> np.ndarray:
    """ Transform STFT feature
    Args:
        Y: STFT
            (n_frames, n_bins)-shape array
        transform_type:
            None, "log"
        dtype: output data type
            np.float32 is expected
    Returns:
        Y (numpy.array): transformed feature
    """
    Y = np.abs(Y)
    if transform_type.startswith('logmel'):
        n_fft = 2 * (Y.shape[1] - 1)
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft,n_mels= feature_dim)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        if transform_type == 'logmel_meannorm':
            mean = np.mean(Y, axis=0)
            Y = Y - mean
        elif transform_type == 'logmel_meanvarnorm':
            mean = np.mean(Y, axis=0)
            Y = Y - mean
            std = np.maximum(np.std(Y, axis=0), 1e-10)
            Y = Y / std
    else:
        raise ValueError('Unknown transform_type: %s' % transform_type)
    return Y.astype(dtype)


def pad_labels(ts: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    ts_padded = []
    for _, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts_padded.append(torch.cat((
                t, torch.zeros((t.shape[0], out_size - t.shape[1])).to(t.device)), dim=1))
        elif t.shape[1] > out_size:
            # truncate
            logging.warn(f"Skipping {t.shape} with expected maximum {out_size}")
            # raise ValueError
        else:
            ts_padded.append(t)
    return torch.stack(ts_padded)


def save_checkpoint(
    args,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor
) -> None:
    Path(f"{args.output_path}/models").mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        f"{args.output_path}/models/checkpoint_{epoch}.tar"
    )


def load_checkpoint(args: SimpleNamespace, filename: str):
    model = get_model(args)
    optimizer = setup_optimizer(args, model)

    assert isfile(filename), \
        f"File {filename} does not exist."
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def load_initmodel(args: SimpleNamespace):
    return load_checkpoint(args, args.initmodel)


def get_model(args: SimpleNamespace) -> nn.Module:
    if args.model_type == 'TransformerEDA':
        model = TransformerEDADiarization(
            
            in_size=args.feature_dim * (1 + 2 * args.context_size),
            n_units=args.hidden_size,
            e_units=args.encoder_units,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            dropout=args.transformer_encoder_dropout,
            attractor_loss_ratio=args.attractor_loss_ratio,
            attractor_encoder_dropout=args.attractor_encoder_dropout,
            attractor_decoder_dropout=args.attractor_decoder_dropout,
            detach_attractor_loss=args.detach_attractor_loss,
        )
    else:
        raise ValueError('Possible model_type are "TransformerEDA"')
    return model


def average_checkpoints(
    device: torch.device,
    model: nn.Module,
    models_path: str,
    epochs: str
) -> nn.Module:
    epochs = parse_epochs(epochs)
    states_dict_list = []
    for e in epochs:
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(join(
            models_path,
            f"checkpoint_{e}.tar"), map_location=device)
        copy_model.load_state_dict(checkpoint['model_state_dict'])
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list, device)
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model


def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key].to(device)

    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state


def parse_epochs(string: str) -> List[int]:
    parts = string.split(',')
    res = []
    for p in parts:
        if '-' in p:
            interval = p.split('-')
            res.extend(range(int(interval[0])+1, int(interval[1])+1))
        else:
            res.append(int(p))
    return res

def pad_sequence(features, labels, seq_len=None):

    assert len(features) == len(labels), (
        f"Features and labels in batch were expected to match but got "
        "{len(features)} features and {len(labels)} labels.")

    features_padded = []
    labels_padded = []
    if seq_len is None:
        seq_len = max([i.shape[0] for i in features])

    for i, _ in enumerate(features):

        assert features[i].shape[0] == labels[i].shape[0], (
            f"Length of features and labels were expected to match but got {features[i].shape[0]} and {labels[i].shape[0]}"
        )

        length = features[i].shape[0]

        if length < seq_len:
            extend = seq_len - length

            features_padded.append(
                torch.cat(
                    (
                        features[i], 
                        -torch.ones((extend, features[i].shape[1])).to(features[i].device)
                    ), dim=0))

            labels_padded.append(
                torch.cat(
                    (
                        labels[i], 
                        -torch.ones((extend, labels[i].shape[1])).to(labels[i].device)
                    ), dim=0))

        elif length > seq_len:
            raise (f"Sequence of length {length} was received but only {seq_len} was expected.")

        else:
            features_padded.append(features[i])
            labels_padded.append(labels[i])

    return features_padded, labels_padded

def get_n_speaker(label):
    label2 = label.detach().cpu().numpy()
    label2 = np.where(label2>=0,label2,0.)
    labels_counter = label2.sum(0)
    labels_counter = np.where(labels_counter>0,1,0).sum()
    return labels_counter

def compute_loss_and_metrics(
    model: torch.nn.Module,
    labels: torch.Tensor,
    input: torch.Tensor,
    return_metrics: bool
) :
    n_speakers = np.asarray([get_n_speaker(t) for t in labels])
    y_pred, attractor_loss, acc_spk, spk_predict = model(input, labels, n_speakers=n_speakers)
    loss, standard_loss, labels = model.get_loss(
        y_pred, labels, n_speakers, attractor_loss)
    y_pred = y_pred.detach()
    mask = torch.zeros_like(y_pred)
    for index,spk in enumerate(spk_predict):
        if spk == -1:
            continue
        else:
            mask[index,:,:spk] = 1.
    if return_metrics:
        metrics = calculate_metrics(labels, y_pred.detach().sigmoid() * mask.to(y_pred.device), threshold=0.5)
        metrics['acc_spk'] = acc_spk
        return loss, metrics
    else:
        return loss
