# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings

warnings.filterwarnings("ignore")


def write_f32le_to_stdout(audio):
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    stream_data = np.asarray(audio, dtype='<f4', order='C').tobytes()
    fd = sys.stdout.fileno()
    offset = 0
    while offset < len(stream_data):
        written = os.write(fd, stream_data[offset:])
        if written <= 0:
            raise RuntimeError('Failed to write stream output to stdout')
        offset += written
    sys.stdout.flush()


def write_flac_to_stdout(audio, sample_rate):
    import tempfile
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, sample_rate, format='FLAC', subtype='PCM_24')
        with open(tmp_path, 'rb') as f:
            data = f.read()
        fd = sys.stdout.fileno()
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise RuntimeError('Failed to write FLAC stream output to stdout')
            offset += written
        sys.stdout.flush()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write_mp3_to_stdout(audio, sample_rate):
    import tempfile
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, sample_rate, bitrate_mode='CONSTANT', compression_level=0.0)
        with open(tmp_path, 'rb') as f:
            data = f.read()
        fd = sys.stdout.fileno()
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise RuntimeError('Failed to write MP3 stream output to stdout')
            offset += written
        sys.stdout.flush()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def read_audio(path, sample_rate):
    try:
        audio, sr = sf.read(path, always_2d=True)
        if sr != sample_rate:
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=sample_rate).T
        return audio.T, sample_rate
    except Exception:
        audio, sr = librosa.load(path, sr=sample_rate, mono=False)
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=0)
        return audio, sample_rate


def run_inference(
    model: "torch.nn.Module",
    args: "argparse.Namespace",
    config: dict,
    device: "torch.device",
    verbose: bool = False
) -> None:
    """
    Process audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : argparse.Namespace
        Arguments containing input, input folder, output folder, and processing options.
    config : dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()
    model.eval()

    if args.input:
        if os.path.isdir(args.input):
            mixture_paths = sorted(
                glob.glob(os.path.join(args.input, "**/*.*"), recursive=True)
            )
            mixture_paths = [p for p in mixture_paths if os.path.isfile(p)]
            input_root = args.input
        else:
            mixture_paths = [args.input]
            input_root = os.path.dirname(args.input)
    elif args.input_folder:
        # Recursively collect all files from input directory
        mixture_paths = sorted(
            glob.glob(os.path.join(args.input_folder, "**/*.*"), recursive=True)
        )
        mixture_paths = [p for p in mixture_paths if os.path.isfile(p)]
        input_root = args.input_folder
    else:
        print("No input provided. Please use --input or --input_folder.")
        return

    sample_rate: int = getattr(config.audio, "sample_rate", 44100)
    streaming = getattr(args, "stream_f32le_instrumental", False) or getattr(args, "stream_f32le_vocal", False)

    if not streaming:
        print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")
    else:
        if len(mixture_paths) != 1:
            print("Streaming modes require exactly one input track. Use --input for streaming.", file=sys.stderr)
            return

    instruments: list[str] = prefer_target_instrument(config)[:]
    if not streaming:
        os.makedirs(args.store_dir, exist_ok=True)

    # Wrap paths with progress bar if not in verbose mode
    if not verbose and not streaming:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    # Determine whether to use detailed progress bar
    if args.disable_detailed_pbar or streaming:
        detailed_pbar = False
    else:
        detailed_pbar = True

    for path in mixture_paths:
        # Get relative path from input folder
        relative_path: str = os.path.relpath(path, input_root)
        # Extract directory and file name
        dir_name: str = os.path.dirname(relative_path)
        file_name: str = os.path.splitext(os.path.basename(path))[0]

        try:
            mix, sr = read_audio(path, sample_rate)
        except Exception as e:
            if not streaming:
                print(f"Cannot read track: {format(path)}")
                print(f"Error message: {str(e)}")
            continue

        # Convert mono audio to expected channel format if needed
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)

        original_mono = mix.shape[0] == 1
        if original_mono:
            if "num_channels" in config.audio:
                if config.audio["num_channels"] == 2:
                    if not streaming:
                        print("Convert mono track to stereo...")
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()

        # Normalize input audio if enabled
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                mix, norm_params = normalize_audio(mix)

        # Set extract_instrumental in config so demix callback can see it
        if isinstance(config, dict):
            config['extract_instrumental'] = args.extract_instrumental
        else:
            config.extract_instrumental = args.extract_instrumental

        stream_callback = None
        if streaming:
            def stream_callback(waveforms_chunk):
                if getattr(args, "stream_f32le_vocal", False):
                    instr_key = "vocals" if "vocals" in instruments else instruments[0]
                    chunk = waveforms_chunk[instr_key]
                else:
                    chunk = waveforms_chunk["instrumental"]

                if original_mono:
                    chunk = chunk[0]
                else:
                    chunk = chunk.T

                if getattr(args, "flac", False):
                    write_flac_to_stdout(chunk, sr)
                elif getattr(args, "mp3", False):
                    write_mp3_to_stdout(chunk, sr)
                else:
                    write_f32le_to_stdout(chunk)

        # Perform source separation
        waveforms_orig = demix(
            config,
            model,
            mix,
            device,
            model_type=args.model_type,
            pbar=detailed_pbar,
            callback=stream_callback
        )

        if streaming:
            continue

        # Apply test-time augmentation if enabled
        if args.use_tta:
            waveforms_orig = apply_tta(
                config,
                model,
                mix,
                waveforms_orig,
                device,
                args.model_type
            )

        # Extract instrumental track if requested
        if args.extract_instrumental:
            instr = "vocals" if "vocals" in instruments else instruments[0]
            waveforms_orig["instrumental"] = mix_orig - waveforms_orig[instr]
            if "instrumental" not in instruments:
                instruments.append("instrumental")

        for instr in instruments:
            estimates = waveforms_orig[instr]

            # Denormalize output audio if normalization was applied
            if "normalize" in config.inference:
                if config.inference["normalize"] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            if getattr(args, "flac", False):
                codec = "flac"
            elif getattr(args, "mp3", False):
                codec = "mp3"
            else:
                peak: float = float(np.abs(estimates).max())
                if peak <= 1.0 and args.pcm_type != 'FLOAT':
                    codec = "flac"
                else:
                    codec = "wav"

            subtype = args.pcm_type

            # Generate output directory structure using relative paths
            dirnames, fname = format_filename(
                args.filename_template,
                instr=instr,
                start_time=int(start_time),
                file_name=file_name,
                dir_name=dir_name,
                model_type=args.model_type,
                model=os.path.splitext(
                    os.path.basename(args.start_check_point)
                )[0],
            )

            # Create output directory
            output_dir: str = os.path.join(args.store_dir, *dirnames)
            os.makedirs(output_dir, exist_ok=True)

            output_path: str = os.path.join(output_dir, f"{fname}.{codec}")
            if codec == 'mp3':
                sf.write(output_path, estimates.T, sr, bitrate_mode='CONSTANT', compression_level=0.0)
            elif codec == 'flac':
                sf.write(output_path, estimates.T, sr, format='FLAC', subtype='PCM_24')
            else:
                sf.write(output_path, estimates.T, sr, subtype=subtype)

            # Draw and save spectrogram if enabled
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)

    if not streaming:
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

def format_filename(template, **kwargs):
    '''
    Formats a filename from a template. e.g "{file_name}/{instr}"
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    '''
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname

def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_inference(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
