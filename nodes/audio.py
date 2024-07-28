import torch
import torchaudio


class MTB_AudioSequence:
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

    def sequence(self, silence_duration: float, **kwargs):
        audios = kwargs.values()

        common_sample_rate = max([audio["sample_rate"] for audio in audios])

        is_stereo = any(
            waveform.shape[1] == 2
            for waveform in [audio["waveform"] for audio in audios]
        )

        resampled_audios = []
        for audio in audios:
            if audio["sample_rate"] != common_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=audio["sample_rate"], new_freq=common_sample_rate
                )
                audio["waveform"] = resampler(audio["waveform"])

            # convert to stereo
            if is_stereo and audio["waveform"].shape[1] == 1:
                audio["waveform"] = torch.cat(
                    [audio["waveform"], audio["waveform"]], dim=1
                )
            resampled_audios.append(audio)

        silence = torch.zeros(
            (
                1,
                2 if is_stereo else 1,
                int(silence_duration * common_sample_rate),
            )
        )
        sequence = []
        for i, audio in enumerate(resampled_audios):
            sequence.append(audio["waveform"])
            if i < len(resampled_audios) - 1:
                sequence.append(silence)

        sequenced_waveform = torch.cat(sequence, dim=-1)
        return (
            {
                "sample_rate": common_sample_rate,
                "waveform": sequenced_waveform,
            },
        )


__nodes__ = [
    MTB_AudioSequence,
]
