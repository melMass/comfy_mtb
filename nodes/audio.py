from typing import TypedDict

import torch
import torchaudio


class AudioDict(TypedDict):
    """Comfy's representation of AUDIO data."""

    sample_rate: int
    waveform: torch.Tensor


AudioData = AudioDict | list[AudioDict]


class MtbAudio:
    """Base class for audio processing."""

    @classmethod
    def is_stereo(
        cls,
        audios: AudioData,
    ) -> bool:
        if isinstance(audios, list):
            return any(cls.is_stereo(audio) for audio in audios)
        else:
            return audios["waveform"].shape[1] == 2

    @staticmethod
    def resample(audio: AudioDict, common_sample_rate: int) -> AudioDict:
        if audio["sample_rate"] != common_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio["sample_rate"], new_freq=common_sample_rate
            )
            return {
                "sample_rate": common_sample_rate,
                "waveform": resampler(audio["waveform"]),
            }
        else:
            return audio

    @staticmethod
    def to_stereo(audio: AudioDict) -> AudioDict:
        if audio["waveform"].shape[1] == 1:
            return {
                "sample_rate": audio["sample_rate"],
                "waveform": torch.cat(
                    [audio["waveform"], audio["waveform"]], dim=1
                ),
            }
        else:
            return audio

    @classmethod
    def preprocess_audios(
        cls, audios: list[AudioDict]
    ) -> tuple[list[AudioDict], bool, int]:
        max_sample_rate = max([audio["sample_rate"] for audio in audios])

        resampled_audios = [
            cls.resample(audio, max_sample_rate) for audio in audios
        ]

        is_stereo = cls.is_stereo(audios)
        if is_stereo:
            audios = [cls.to_stereo(audio) for audio in resampled_audios]

        return (audios, is_stereo, max_sample_rate)


class MTB_AudioStack(MtbAudio):
    """Stack/Overlay audio inputs (dynamic inputs).

    - pad audios to the longest inputs.
    - resample audios to the highest sample rate in the inputs.
    - convert them all to stereo if one of the inputs is.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("stacked_audio",)
    CATEGORY = "mtb/audio"
    FUNCTION = "stack"

    def stack(self, **kwargs: AudioDict) -> tuple[AudioDict]:
        audios, is_stereo, max_rate = self.preprocess_audios(
            list(kwargs.values())
        )

        max_length = max([audio["waveform"].shape[-1] for audio in audios])

        padded_audios: list[torch.Tensor] = []
        for audio in audios:
            padding = torch.zeros(
                (
                    1,
                    2 if is_stereo else 1,
                    max_length - audio["waveform"].shape[-1],
                )
            )
            padded_audio = torch.cat([audio["waveform"], padding], dim=-1)
            padded_audios.append(padded_audio)

        stacked_waveform = torch.stack(padded_audios, dim=0).sum(dim=0)

        return (
            {
                "sample_rate": max_rate,
                "waveform": stacked_waveform,
            },
        )


class MTB_AudioSequence(MtbAudio):
    """Sequence audio inputs (dynamic inputs).

    - adding silence_duration between each segment.
    - resample audios to the highest sample rate in the inputs.
    - convert them all to stereo if one of the inputs is.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "silence_duration": (("FLOAT"), {"default": 0.0, "step": 0.01})
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("sequenced_audio",)
    CATEGORY = "mtb/audio"
    FUNCTION = "sequence"

    def sequence(self, silence_duration: float, **kwargs: AudioDict):
        audios, is_stereo, max_rate = self.preprocess_audios(
            list(kwargs.values())
        )

        silence = torch.zeros(
            (
                1,
                2 if is_stereo else 1,
                int(silence_duration * max_rate),
            )
        )
        sequence: list[torch.Tensor] = []
        for i, audio in enumerate(audios):
            sequence.append(audio["waveform"])
            if i < len(audios) - 1:
                sequence.append(silence)

        sequenced_waveform = torch.cat(sequence, dim=-1)
        return (
            {
                "sample_rate": max_rate,
                "waveform": sequenced_waveform,
            },
        )


__nodes__ = [MTB_AudioSequence, MTB_AudioStack]
