from typing import Dict
import torch


def calc_diarization_error(pred, label, max_speaker=None, label_delay=0, threshold=0.5):
    """
    Calculates diarization error stats for reporting.
    Args:
      pred (torch.FloatTensor): (T,C)-shaped pre-activation values
      label (torch.FloatTensor): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |
    Returns:
      res: dict of diarization error stats
    """
    label = label[:len(label) - label_delay, ...]
    label=label.clone().detach()
    mask = label < 0
    label[mask] = 0
    decisions = (pred[label_delay:, ...] > threshold).long().clone()
    decisions[mask] = 0
    # print(decisions)
    n_ref = label.sum(axis=-1).long()
    n_sys = decisions.sum(axis=-1).long()
    res = {}
    res['speech_scored'] = (n_ref > 0).sum()
    res['speech_miss'] = ((n_ref > 0) & (n_sys == 0)).sum()
    res['speech_falarm'] = ((n_ref == 0) & (n_sys > 0)).sum()
    res['speaker_scored'] = (n_ref).sum()+1e-4
    res['speaker_miss'] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
    res['speaker_falarm'] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
    n_map = ((label == 1) & (decisions == 1)).sum(axis=-1)
    res['speaker_error'] = (torch.min(n_ref, n_sys) - n_map).sum()
    res['correct'] = (label == decisions).sum() / label.shape[1]
    res['diarization_error'] = (res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    res['DER'] = res['diarization_error'] / res['speaker_scored'] * 100
    return res


def report_diarization_error(ys, labels, max_speakers=None, threshold=0.5):
    """
    Reports diarization errors
    Should be called with torch.no_grad
    Args:
      ys: B-length list of predictions (torch.FloatTensor)
      labels: B-length list of labels (torch.FloatTensor)
    """
    stats_avg = {}
    if max_speakers is None:
      max_speakers=[None,] * len(ys)
    for y, t, m in zip(ys, labels, max_speakers):
      stats = calc_diarization_error(y, t, max_speaker=m, threshold=threshold)
      for k, v in stats.items():
          if k not in stats_avg: stats_avg[k] = [] 
          stats_avg[k].append(v)
    return stats_avg


def calculate_metrics(
    target: torch.Tensor,
    decisions: torch.Tensor,
    threshold: float = 0.5,
    round_digits: int = 2,
) -> Dict[str, float]:
    return report_diarization_error(
        decisions, target, threshold=threshold
    )
