from enum import IntEnum
import logging
import os
import re
import unicodedata

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from pydub import AudioSegment

import simfile
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine



class BiasKernel(IntEnum):
    LOUDEST = 0
    RISING = 1


def slugify(value, allow_unicode=False):
    """
    https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value)
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def guess_paradigm(sync_bias):
    if sync_bias > -0.0035 and sync_bias < 0.0035:
        return 'probably null'
    elif sync_bias > 0.0055 and sync_bias < 0.0125:
        return 'probably +9ms'
    else:
        return 'unclear paradigm'


def check_sync_bias(simfile_dir, plot_dir=None, show_intermediate_plots=False, kernel_type=BiasKernel.RISING):
    # Open simfile
    test_audio_path = None
    test_simfile_path = None
    for f in os.listdir(simfile_dir):
        if os.path.splitext(f)[1] in ['.ssc', '.sm']:
            if (test_simfile_path is None) or (os.path.splitext(test_simfile_path)[1] == '.sm'):
                test_simfile_path = os.path.join(simfile_dir, f)
    if test_simfile_path is None:
        # Not a simfile!
        return None

    test_simfile = simfile.open(test_simfile_path)

    simfile_artist = test_simfile.artisttranslit or test_simfile.artist
    simfile_title = test_simfile.titletranslit or test_simfile.title

    # Default to first chart
    chart = test_simfile.charts[0]

    # Account for split audio
    if not hasattr(chart, 'music') or chart.music is None:
        test_audio_path = os.path.join(simfile_dir, test_simfile.music)
    else:
        test_audio_path = os.path.join(simfile_dir, chart.music)

    engine = TimingEngine(TimingData(test_simfile, chart))

    ###################################################################
    # Load audio using pydub
    audio_ext = os.path.splitext(test_audio_path)[1]
    audio = AudioSegment.from_file(test_audio_path, format=audio_ext[1:])
    audio_data = np.array(audio.get_array_of_samples())

    # Account for stereo audio and normalize
    # https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3
    if audio.channels == 2:
        audio_data = audio_data.reshape((-1, 2))
    audio_data = audio_data / 2**15

    ###################################################################
    # Create spectrogram from audio
    # https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
    # https://stackoverflow.com/questions/47954034/plotting-spectrogram-in-audio-analysis
    window_ms = 5       # Window to calculate spectrogram over, ms
    step_ms = 1         # Overlap between windows (effectively step size), ms
    eps = 1e-9          # Epsilon for logarithms

    nperseg = int(audio.frame_rate * window_ms * 1e-3)              # number of samples per spectrogram segment
    noverlap = nperseg - int(audio.frame_rate * step_ms * 1e-3)     # number of overlap samples
    frequencies, times, spectrogram = signal.spectrogram(
        audio_data[:, 0],       # Mono left channel please
        fs=audio.frame_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False
    )
    splog = np.log2(spectrogram + eps)                              # Calculate in log domain

    if show_intermediate_plots:
        fig = plt.figure(figsize=(30, 6))
        plt.pcolormesh(times, frequencies, splog)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Full spectrogram for {simfile_artist} - "{simfile_title}"')
        plt.show()

    ###################################################################
    # Use beat timing information to construct a "fingerprint"
    # of audio spectra around the time each beat takes place
    fingerprint_sec = 0.10                  # Moving fingerprint window (100ms is quite reasonable)
    frequency_emphasis_factor = 3000        # filt(f) = f * e^(-f / emphasis); use None to bypass

    # Recalculate actual timestamps of spectrogram measurements
    actual_step = (nperseg - noverlap) / audio.frame_rate
    fingerprint_size = 2 * int(0.5 * round(fingerprint_sec / actual_step))
    fingerprint_times = np.arange(-fingerprint_size // 2, fingerprint_size // 2) * actual_step
    
    # Accumulator over beats, summed in the frequency domain
    acc = np.zeros((frequencies.size, fingerprint_size))

    # Time-scale digest (frequencies flattened to single value,
    # each beat gets a fingerprint width)
    digest = np.zeros((0, fingerprint_size))

    # For each beat in the song that has a full
    # fingerprint's width of surrounding audio data:
    b = 0
    while True:
        t = engine.time_at(b)
        b += 1
        if (t < 0):
            # Too early
            continue
        if (t > audio.duration_seconds):
            # Too late
            break

        t_s = max(0,              int(t / actual_step - fingerprint_size * 0.5))
        t_f = min(times.shape[0], int(t / actual_step + fingerprint_size * 0.5))
        if (t_f - t_s != fingerprint_size):
            # Not enough data at this beat tbh
            continue
        
        frequency_weights = 1
        if frequency_emphasis_factor is not None:
            # filt(f) = f * e^(-f / emphasis); use None to bypass
            frequency_weights = np.tile(frequencies * np.exp(-frequencies / frequency_emphasis_factor), [fingerprint_size, 1]).T
        spfilt = splog[:, t_s:t_f] * frequency_weights

        # Accumulate, and add to digest
        acc += spfilt
        digest = np.vstack([digest, np.sum(spfilt, axis=0)])
        

    ###################################################################
    # Apply a convolution to detect the downbeat attack

    if kernel_type == BiasKernel.LOUDEST:
        # Loudest point of attack
        time_edge_kernel = [
            [1, 3, 10, 3, 1],
            [1, 3, 10, 3, 1],
            [1, 3, 10, 3, 1],
            [1, 3, 10, 3, 1],
            [1, 3, 10, 3, 1]
        ]
        time_edge_offset = 0.000
    else:   # BiasKernel.RISING
        # Leading edge of attack
        time_edge_kernel = [
            [1, 3, 10, 0, -10, -3, -1],
            [1, 3, 10, 0, -10, -3, -1],
            [1, 3, 10, 0, -10, -3, -1],
            [1, 3, 10, 0, -10, -3, -1],
            [1, 3, 10, 0, -10, -3, -1]
        ]
        time_edge_offset = 0.003

    acc_edge = signal.convolve2d(acc, time_edge_kernel, mode='same', boundary='wrap')
    # acc_edge = signal.convolve2d(digest, time_edge_kernel, mode='same', boundary='wrap')

    # Flatten frequencies to a value that only depends on time
    acc_edge_sum = np.sum(acc_edge, axis=0)
    fingerprint_times = np.arange(-fingerprint_size // 2, fingerprint_size // 2) * actual_step

    # Choose the highest response to the convolution as the downbeat attack
    sync_bias = fingerprint_times[np.argmax(acc_edge_sum)] + time_edge_offset
    probable_bias = guess_paradigm(sync_bias)
    # print(f'Sync bias: {sync_bias:0.3f} ({probable_bias})')

    # Set up visuals to show the user what's going on.
    plot_title = f'Sync fingerprint for {simfile_artist} - "{simfile_title}"\nDerived sync bias: {sync_bias:0.3f} ({probable_bias})'
    sanitized_title = slugify(simfile_title, allow_unicode=False)
    time_ticks = np.arange(-fingerprint_sec * 1000, fingerprint_sec * 1000, 10)
    
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = os.getcwd()

    # Accumulator in frequency domain
    fig = plt.figure(figsize=(6, 6))
    plt.pcolormesh(fingerprint_times * 1e3, frequencies, acc)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [msec]')
    plt.xticks(time_ticks)
    plt.plot(np.ones(np.shape(frequencies)) * sync_bias * 1e3, frequencies, 'r-')
    plt.title(plot_title)
    if show_intermediate_plots:
        plt.show()
    fig.savefig(os.path.join(plot_dir, f'bias-freqdomain-{sanitized_title}.png'))
    plt.close(fig)

    # Accumulator in frequency domain after convolution
    fig = plt.figure(figsize=(6, 6))
    plt.pcolormesh(fingerprint_times * 1e3, frequencies, acc_edge)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [msec]')
    plt.xticks(time_ticks)
    plt.plot(np.ones(np.shape(frequencies)) * sync_bias * 1e3, frequencies, 'r-')
    plt.title(plot_title)
    if show_intermediate_plots:
        plt.show()
    fig.savefig(os.path.join(plot_dir, f'bias-freqkernel-{sanitized_title}.png'))
    plt.close(fig)

    # Digest in beat domain
    fig = plt.figure(figsize=(6, 6))
    plt.pcolormesh(fingerprint_times * 1e3, np.arange(digest.shape[0]), digest)
    plt.clim(np.percentile(digest[:], 10), np.percentile(digest[:], 90))
    plt.ylabel('Beat Index')
    plt.xlabel('Time [msec]')
    plt.xticks(time_ticks)
    plt.plot(np.ones(np.shape(digest)[0]) * sync_bias * 1e3, np.arange(digest.shape[0]), 'r-')
    plt.title(plot_title)
    if show_intermediate_plots:
        plt.show()
    fig.savefig(os.path.join(plot_dir, f'bias-beatdigest-{sanitized_title}.png'))
    plt.close(fig)

    if show_intermediate_plots:
        # Quick and dirty plot of convolution response
        plt.plot(fingerprint_times, acc_edge_sum)
        plt.show()

    # Done!
    return sync_bias


def process_pack(pack_dir):
    plot_dir = os.path.join(pack_dir, '__bias-check')

    sync_bias_map = {}
    for d in os.listdir(pack_dir):
        d_full = os.path.join(pack_dir, d)
        if os.path.isdir(d_full):
            try:
                sync_bias = check_sync_bias(d_full, plot_dir=plot_dir)
                if sync_bias is not None:
                    sync_bias_map[d] = sync_bias
                    logging.info(f'{d:>50s}: derived sync bias = {sync_bias:+0.3f} ({guess_paradigm(sync_bias)})')
            except Exception as e:
                logging.exception(e)

    return sync_bias_map

