"""EEG data augmentation: time shift, channel dropout, Gaussian noise.

All augmentations operate on raw segment tensors of shape [C, T] or [B, C, T].
Only applied during training.
"""

import torch


def time_shift(x: torch.Tensor, max_shift_sec: float, sampling_rate: int = 200) -> torch.Tensor:
    """Randomly shift the signal in time by cropping/zero-padding.

    Args:
        x: [C, T] or [B, C, T] EEG segment tensor.
        max_shift_sec: maximum shift in seconds.
        sampling_rate: sampling rate in Hz (after resample, typically 200).

    Returns:
        Augmented tensor of same shape.
    """
    max_shift = int(max_shift_sec * sampling_rate)
    if max_shift == 0:
        return x

    if x.dim() == 2:
        # [C, T]
        C, T = x.shape
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return x
        result = torch.zeros_like(x)
        if shift > 0:
            result[:, shift:] = x[:, :T - shift]
        else:
            result[:, :T + shift] = x[:, -shift:]
        return result
    elif x.dim() == 3:
        # [B, C, T]
        B, C, T = x.shape
        result = torch.zeros_like(x)
        for b in range(B):
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift == 0:
                result[b] = x[b]
            elif shift > 0:
                result[b, :, shift:] = x[b, :, :T - shift]
            else:
                result[b, :, :T + shift] = x[b, :, -shift:]
        return result
    return x


def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """Randomly zero out entire channels.

    Args:
        x: [C, T] or [B, C, T] EEG segment tensor.
        p: probability of dropping each channel.
    """
    if p <= 0:
        return x

    if x.dim() == 2:
        C = x.shape[0]
        mask = torch.rand(C, device=x.device) > p
        # ensure at least one channel survives
        if not mask.any():
            mask[0] = True
        return x * mask.unsqueeze(1).float()
    elif x.dim() == 3:
        B, C, _ = x.shape
        mask = (torch.rand(B, C, device=x.device) > p).float()
        # ensure at least one channel per sample survives
        for b in range(B):
            if not mask[b].any():
                mask[b, 0] = 1.0
        return x * mask.unsqueeze(2)
    return x


def gaussian_noise(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to the signal.

    Args:
        x: EEG segment tensor of any shape.
        std: noise standard deviation.
    """
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise


def apply_augmentation(
    x: torch.Tensor,
    time_shift_sec: float = 1.0,
    channel_dropout_p: float = 0.1,
    noise_std: float = 0.01,
    sampling_rate: int = 200,
) -> torch.Tensor:
    """Apply all augmentations in sequence.

    Args:
        x: [C, T] segment tensor (before model reshape).
        time_shift_sec: max time shift in seconds.
        channel_dropout_p: channel dropout probability.
        noise_std: Gaussian noise std.
        sampling_rate: sampling rate in Hz (after resample).

    Returns:
        Augmented tensor.
    """
    if time_shift_sec > 0:
        x = time_shift(x, time_shift_sec, sampling_rate)
    if channel_dropout_p > 0:
        x = channel_dropout(x, channel_dropout_p)
    if noise_std > 0:
        x = gaussian_noise(x, noise_std)
    return x
