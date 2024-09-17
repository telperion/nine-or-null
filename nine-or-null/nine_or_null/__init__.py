_VERSION = '0.8.4'

from collections.abc import Container
import csv
from datetime import datetime as dt
from enum import IntEnum
import logging
import os
import re
import sys
import unicodedata

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from scipy import signal
from pydub import AudioSegment

import simfile
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine

_NINEORNULL_NULL = 0
_NINEORNULL_P9MS = 9
_CSV_FIELDNAMES = [
    'path',
    'slot',
    'bias',
    'conf',
    'interquintile',
    'stdev',
    'paradigm',
    'timestamp',
    'fingerprint_ms',
    'window_ms',
    'step_ms',
    'kernel_type',
    'kernel_target',
    'sample_rate',
    'title',
    'titletranslit',
    'subtitle',
    'subtitletranslit',
    'artist',
    'artisttranslit',
]
_PARAMETERS = {
    # Default parameters.
    'root_path':        'Path to a simfile, pack, or collection of packs to analyze. If not provided, the GUI is invoked instead.',
    'report_path':      'The destination directory for the sync bias report and audio fingerprint plots. If not provided, defaults to "<root_path>/__bias-check".',
    'overwrite':        'If not set, skip files that already have bias check data in the directory.',
    'consider_null':    'Consider charts close enough to 0ms bias to be "correct" under the null (StepMania) sync paradigm.',
    'consider_p9ms':    'Consider charts close enough to +9ms bias to be "correct" under the In The Groove sync paradigm.',
    'tolerance':        'If a simfile\'s sync bias lands within a paradigm ± this tolerance, that counts as "close enough".',
    'confidence_limit': 'If the confidence in a simfile\'s sync bias is below this value, it will not be considered for unbiasing.',
    'fingerprint_ms':   '[ms] Time margin on either side of the beat to analyze.',
    'window_ms':        '[ms] The spectrogram algorithm\'s moving window parameter.',
    'step_ms':          '[ms] Controls the spectrogram algorithm\'s overlap parameter, but expressed as a step size.',
    'kernel_target':    'Choose whether to convolve with the beat digest ("digest") or the spectral accumulator ("accumulator").',
    'kernel_type':      'Choose a kernel that responds to a rising edge ("rising") or local loudness ("loudest").',
    'magic_offset_ms':  '[ms] Add a constant value to the time of maximum kernel response. I haven\'t tracked the cause of this down yet. Might be related to attack perception?',
    'full_spectrogram': 'Analyze the full spectrogram in one go - this will make the program run slower...',
    'to_paradigm':      'Choose a target paradigm for the pack unbiasing step. This will modify your simfiles!'
}
_THEORETICAL_UPPER = 0.83
_NEARNESS_SCALAR = 10    # milliseconds
_NEARNESS_OFFSET = 0.5   # milliseconds

class FloatRange(Container):
    # Endpoint inclusive.
    def __init__(self, lo=None, hi=None):
        self.lo = lo
        self.hi = hi

    def __iter__(self):
        return iter([f'>= {self.lo}', f'<= {self.hi}'])

    def __contains__(self, value):
        if (self.lo is not None) and (value < self.lo):
            return False
        if (self.hi is not None) and (value > self.hi):
            return False
        return True

class BiasKernel(IntEnum):
    RISING = 0
    LOUDEST = 1

class KernelTarget(IntEnum):
    DIGEST = 0
    ACCUMULATOR = 1


_PARAM_DEFAULTS = {
    # Default parameters.
    'root_path':        None,
    'report_path':      None,
    'overwrite':        True,
    'consider_null':    True,
    'consider_p9ms':    True,
    'tolerance':        3.0,
    'confidence_limit': 80,
    'fingerprint_ms':   50,
    'window_ms':        10,
    'step_ms':          0.2,
    'kernel_target':    KernelTarget.DIGEST,
    'kernel_type':      BiasKernel.RISING,
    'magic_offset_ms':  0.0,
    'full_spectrogram': False,
    'to_paradigm':      None
}

def timestamp():
    return dt.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]


def slot_abbreviation(steps_type, chart_slot, chart_index=0, paradigm='null'):
    logging.info(steps_type)
    logging.info(chart_slot)
    if paradigm == '+9ms':
        map_style = {
            'dance-single': 'S',
            'dance-double': 'D'
        }
        map_slot = {
            'Challenge': 'X',
            'Hard': 'H',
            'Medium': 'M',
            'Easy': 'E',
            'Beginner': 'N',
            'Edit': '.'
        }
        return map_style.get(steps_type, '?') \
               + map_slot.get(chart_slot, '?') \
               + (chart_slot == 'Edit' and f'{chart_index}' or '')
    else:   # Charts that don't fit a paradigm are probably DDR charts...no shade but
        map_style = {
            'dance-single': 'SP',
            'dance-double': 'DP'
        }
        map_slot = {
            'Challenge': 'C',
            'Hard': 'E',
            'Medium': 'D',
            'Easy': 'B',
            'Beginner': 'b',
            'Edit': 'X'
        }
        return map_slot.get(chart_slot, '?') \
               + (chart_slot == 'Edit' and f'{chart_index}' or '') \
               + map_style.get(steps_type, '?')


def slot_expansion(abbr):
    if abbr[-2:] in ['SP', 'DP']:
        map_style = {
            'SP': 'dance-single',
            'DP': 'dance-double'
        }
        map_slot = {
            'C': 'Challenge',
            'E': 'Hard',
            'D': 'Medium',
            'B': 'Easy',
            'b': 'Beginner',
            'X': 'Edit'
        }
        steps_type = map_style[abbr[-2:]]
        chart_slot = map_slot[abbr[0]]
        if len(abbr) > 3:
            chart_index = int(abbr[1:-2])
        else:
            chart_index = None
    elif abbr[0] in ['S', 'D']:
        map_style = {
            'S': 'dance-single',
            'D': 'dance-double'
        }
        map_slot = {
            'X': 'Challenge',
            'H': 'Hard',
            'M': 'Medium',
            'E': 'Easy',
            'N': 'Beginner',
            '.': 'Edit'
        }
        steps_type = map_style[abbr[0]]
        chart_slot = map_slot[abbr[1]]
        if len(abbr) > 2:
            chart_index = int(abbr[2:])
        else:
            chart_index = None
    else:
        raise Exception(f'Couldn\'t deduce meaning of slot abbreviation "{abbr}"')

    return steps_type, chart_slot, chart_index


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


def timedelta_as_hhmmss(delta):
    total_sec = int(delta.total_seconds())
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = (total_sec % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def guess_paradigm(sync_bias_ms, tolerance=3, consider_null=True, consider_p9ms=True, short_paradigm=True, **kwargs):
    if consider_null and (sync_bias_ms > _NINEORNULL_NULL - tolerance and sync_bias_ms < _NINEORNULL_NULL + tolerance):
        return short_paradigm and 'null' or 'probably null'
    elif consider_p9ms and (sync_bias_ms > _NINEORNULL_P9MS - tolerance and sync_bias_ms < _NINEORNULL_P9MS + tolerance):
        return short_paradigm and '+9ms' or 'probably +9ms'
    else:
        return short_paradigm and '????' or 'unclear paradigm'


class FlippableAxes:
    def __init__(self, axes: plt.Axes, flip: bool = False):
        self.ax = axes
        self.flip = flip

    def plot_flip(self, x, y, *args, **kwargs):
        return self.ax.plot(y, x, *args, **kwargs)

    def pcolormesh_flip(self, x, y, d, *args, **kwargs):
        return self.ax.pcolormesh(y, x, d.T, *args, **kwargs)

    def __getattr__(self, attr):
        if attr == 'set_xlabel':
            return self.ax.set_ylabel if self.flip else self.ax.set_xlabel
        if attr == 'set_ylabel':
            return self.ax.set_xlabel if self.flip else self.ax.set_ylabel
        
        if attr == 'set_xticks':
            return self.ax.set_yticks if self.flip else self.ax.set_xticks
        if attr == 'set_yticks':
            return self.ax.set_xticks if self.flip else self.ax.set_yticks
        
        if attr == 'set_xlim':
            return self.ax.set_ylim if self.flip else self.ax.set_xlim
        if attr == 'set_ylim':
            return self.ax.set_xlim if self.flip else self.ax.set_ylim
        
        if attr == 'xaxis':
            return self.ax.yaxis if self.flip else self.ax.xaxis
        if attr == 'yaxis':
            return self.ax.xaxis if self.flip else self.ax.yaxis
        
        if attr == 'plot':
            return self.plot_flip if self.flip else self.ax.plot
        if attr == 'pcolormesh':
            return self.pcolormesh_flip if self.flip else self.ax.pcolormesh
        
        return getattr(self.ax, attr)
    

def plot_fingerprint(fingerprint, target_axes, **kwargs):
    # Set up visuals to show the user what's going on.
    times_ms = fingerprint['time_values']
    frequencies_kHz = fingerprint['frequencies']
    acc = fingerprint['freq_domain']
    digest = fingerprint['beat_digest']
    beat_indices = fingerprint['beat_indices']
    sync_bias = fingerprint['bias_result']
    post_kernel_flat = fingerprint['convolution']
    post_kernel = fingerprint['post_kernel']
    plot_title = fingerprint['plots_title']
    fingerprint_ms = kwargs.get('fingerprint_ms', 50)
    magic_offset_ms = kwargs.get('magic_offset_ms', 0.0)
    kernel_target = kwargs.get('kernel_target', KernelTarget.DIGEST)
    hide_yticks = kwargs.get('hide_yticks', False)
    flip_axes = kwargs.get('flip_axes', False)

    edge_discard = 5        # TODO: pull in from calling function I guess
    if beat_indices is None:
        beat_indices = np.arange(digest.shape[0])

    time_ticks = np.hstack((
        np.arange(0, times_ms[ 0]-1, -10),
        np.arange(0, times_ms[-1]+1,  10)
    ))
    frequency_line = np.ones(np.shape(frequencies_kHz)) * sync_bias
    beatindex_line = np.ones(np.shape(digest)[0])   * sync_bias
    post_kernel_over_freq = np.interp(
        post_kernel_flat,
        (
            post_kernel_flat[edge_discard:-edge_discard].min(),
            post_kernel_flat[edge_discard:-edge_discard].max()
        ),
        (
            frequencies_kHz.min() * 0.9 + frequencies_kHz.max() * 0.1,
            frequencies_kHz.min() * 0.1 + frequencies_kHz.max() * 0.9
        ))
    post_kernel_over_beat = np.interp(
        post_kernel_flat,
        (
            post_kernel_flat[edge_discard:-edge_discard].min(),
            post_kernel_flat[edge_discard:-edge_discard].max()
        ),
        (
            beat_indices.min() * 0.9 + beat_indices.max() * 0.1,
            beat_indices.min() * 0.1 + beat_indices.max() * 0.9
        ))

    # Accumulator in frequency domain
    ax = FlippableAxes(target_axes[0], flip_axes)
    ax.clear()
    pcm = ax.pcolormesh(times_ms, frequencies_kHz, acc)
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlabel('Time [msec]', labelpad=-12)
    ax.plot(times_ms + magic_offset_ms, post_kernel_over_freq, 'w-')
    ax.plot(frequency_line, frequencies_kHz, 'r-')
    if hide_yticks:
        ax.set_yticks([])
    ax.set_xlim(-fingerprint_ms, fingerprint_ms)
    ax.xaxis.set_major_locator(mticker.FixedLocator(time_ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f'{v:0.0f}' for v in time_ticks]))
    ax.xaxis.set_minor_locator(mticker.FixedLocator((-fingerprint_ms * 0.7, fingerprint_ms * 0.7)))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter((r'$\longleftarrow\:feels\enspace later\:\longleftarrow$', r'$\longrightarrow\:feels\enspace earlier\:\longrightarrow$')))
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=90 if flip_axes else 0, size=10, va="center")
    ax.tick_params('y' if flip_axes else 'x', which='minor', pad=24, bottom=False)
    ax.get_figure().suptitle(plot_title)

    # Digest in beat domain
    ax = FlippableAxes(target_axes[1], flip_axes)
    ax.clear()
    pcm = ax.pcolormesh(times_ms, beat_indices, digest)
    pcm.set_clim(np.percentile(digest[:], 10), np.percentile(digest[:], 90))
    ax.set_ylabel('Beat Index')
    ax.set_xlabel('Time [msec]', labelpad=-12)
    ax.plot(times_ms + magic_offset_ms, post_kernel_over_beat, 'w-')
    ax.plot(beatindex_line, beat_indices, 'r-')
    if hide_yticks:
        ax.set_yticks([])
    ax.set_xlim(-fingerprint_ms, fingerprint_ms)
    ax.xaxis.set_major_locator(mticker.FixedLocator(time_ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f'{v:0.0f}' for v in time_ticks]))
    ax.xaxis.set_minor_locator(mticker.FixedLocator((-fingerprint_ms * 0.7, fingerprint_ms * 0.7)))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter((r'$\longleftarrow\:feels\enspace later\:\longleftarrow$', r'$\longrightarrow\:feels\enspace earlier\:\longrightarrow$')))
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=90 if flip_axes else 0, size=10, va="center")
    ax.tick_params('y' if flip_axes else 'x', which='minor', pad=24, bottom=False)
    ax.get_figure().suptitle(plot_title)

    # Post-convolution plot
    ax = FlippableAxes(target_axes[2], flip_axes)
    ax.clear()
    if kernel_target == KernelTarget.ACCUMULATOR:
        pcm = ax.pcolormesh(times_ms, frequencies_kHz, post_kernel)
        ax.set_ylabel('Frequency [kHz]')
        ax.plot(times_ms + magic_offset_ms, post_kernel_over_freq, 'w-')
        ax.plot(frequency_line, frequencies_kHz, 'r-')
    else: # kernel_target == KernelTarget.DIGEST
        pcm = ax.pcolormesh(times_ms, beat_indices, post_kernel)
        ax.set_ylabel('Beat Index')
        ax.plot(times_ms + magic_offset_ms, post_kernel_over_beat, 'w-')
        ax.plot(beatindex_line, beat_indices, 'r-')
    pcm.set_clim(np.percentile(post_kernel[:], 3), np.percentile(post_kernel[:], 97))
    ax.set_xlabel('Time [msec]', labelpad=-12)
    if hide_yticks:
        ax.set_yticks([])
    ax.set_xlim(-fingerprint_ms, fingerprint_ms)
    ax.xaxis.set_major_locator(mticker.FixedLocator(time_ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f'{v:0.0f}' for v in time_ticks]))
    ax.xaxis.set_minor_locator(mticker.FixedLocator((-fingerprint_ms * 0.7, fingerprint_ms * 0.7)))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter((r'$\longleftarrow\:feels\enspace later\:\longleftarrow$', r'$\longrightarrow\:feels\enspace earlier\:\longrightarrow$')))
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=90 if flip_axes else 0, size=10, va="center")
    ax.tick_params('y' if flip_axes else 'x', which='minor', pad=24, bottom=False)
    ax.get_figure().suptitle(plot_title)


def get_full_title(base_simfile):
    simfile_artist   = base_simfile.artisttranslit   or base_simfile.artist
    simfile_title    = base_simfile.titletranslit    or base_simfile.title
    simfile_subtitle = base_simfile.subtitletranslit or base_simfile.subtitle
    return f'{simfile_title}{simfile_subtitle and (" " + simfile_subtitle) or ""}'


def find_music(simfile_dir, music_filename):
    if (music_filename is None) or (len(music_filename) == 0):
        # Any info whatsoever about the music filename?
        music_stem = ''
    else:
        if os.path.isfile(os.path.join(simfile_dir, music_filename)):
            # Already know which music file is being used.
            return music_filename

        # Let's at least look for a matching filename stem.
        music_stem = os.path.splitext(os.path.split(music_filename)[1])[0]
    
    files = os.listdir(simfile_dir)
    # StepMania supports MP3, WAV, OGA, and OGG
    # Project OutFox community might use Opus or FLAC
    music_options = [f for f in files if os.path.splitext(f)[1] in ['.wav', '.mp3', '.oga', '.ogg', '.opus', '.flac']]
    if music_stem == '':
        # Any audio file will be accepted here.
        music_options_fn = music_options
    else:
        # Only audio files that match the presumed stem are accepted.
        music_options_fn = [f for f in music_options if os.path.splitext(f)[0].lower() == music_stem.lower()]
    
    if len(music_options_fn) != 1:
        # Last-ditch effort. Any music files, even if they don't match the filename??
        music_options_fn = music_options

    if len(music_options_fn) == 0:
        # No audio...
        raise FileNotFoundError(f'No audio file matching {music_filename}')
    elif len(music_options_fn) > 1:
        # Too many audio...
        raise FileNotFoundError(f'Too many audio files matching {music_filename}')

    music_found = music_options_fn[0]
    logging.info(f"Simfile/chart audio substitution: {(music_filename is None) and '' or music_filename} --> {music_found}")
    return music_found


def check_sync_bias(root_path, simfile_dir, base_simfile, chart_index=None, report_path=None, save_plots=True, show_intermediate_plots=False, **kwargs):
    fingerprint = {
        'beat_digest': None,    # Beat digest fingerprint (beat index vs. time)
        'beat_indices': None,   # Beat indices that contributed to the digest
        'freq_domain': None,    # Accumulation in frequency domain (frequency vs. time)
        'post_kernel': None,    # Post-kernel
        'bias_result': None,    # Scalar value result of the bias analysis
        'convolution': None,    # 1-D plot of the convolution response (where the time at max determines the bias)
        'time_values': None,    # x-axis
        'frequencies': None,    # y-axis (for frequencies)
        'plots_title': None,    # title used for the plot (contains simfile info, bias, etc.)
        'files_title': None     # title stem used for saving plots to files
    }

    kernel_type      = kwargs.get('kernel_type',      BiasKernel.RISING)
    kernel_target    = kwargs.get('kernel_target',    KernelTarget.DIGEST)
    magic_offset_ms  = kwargs.get('magic_offset_ms',  0.0)                    # Why though
    full_spectrogram = kwargs.get('full_spectrogram', False)
    overwrite        = kwargs.get('overwrite',        True)

    full_simfile_dir = os.path.join(root_path, simfile_dir)

    simfile_artist   = base_simfile.artisttranslit   or base_simfile.artist
    simfile_title    = base_simfile.titletranslit    or base_simfile.title
    simfile_subtitle = base_simfile.subtitletranslit or base_simfile.subtitle
    full_title = get_full_title(base_simfile)
    sanitized_title_existence_test = slugify(full_title, allow_unicode=False)

    # Early exit to avoid overwriting
    existing_outputs = os.listdir(report_path)

    # Account for split audio
    audio_path = os.path.join(full_simfile_dir, find_music(full_simfile_dir, base_simfile.music))
    chart = None
    if chart_index is not None:
        chart = base_simfile.charts[chart_index]
        if chart.get('MUSIC') is not None:
            audio_path = os.path.join(full_simfile_dir, find_music(full_simfile_dir, chart.music))

    engine = TimingEngine(TimingData(base_simfile, chart))

    ###################################################################
    # Load audio using pydub
    audio_ext = os.path.splitext(audio_path)[1]
    audio = AudioSegment.from_file(audio_path, format=audio_ext[1:])
    audio_data = np.array(audio.get_array_of_samples())

    # Account for stereo audio and normalize
    # https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3
    if audio.channels == 2:
        #audio_data = audio_data.reshape((-1, 2)).sum(1) * 0.5       # Reshape to stereo and average the two channels
        #audio_data = audio_data.reshape((-1, 2))[:, 0].flatten()    # Pull mono only
        audio_data = audio_data.reshape((-1, 2)).max(1)              # Reshape to stereo and average the two channels
    audio_data = audio_data / 2**15

    ###################################################################
    # Create spectrogram from audio
    # https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
    # https://stackoverflow.com/questions/47954034/plotting-spectrogram-in-audio-analysis
    fingerprint_ms  = kwargs.get('fingerprint_ms', 50)      # Moving fingerprint window (100ms is quite reasonable)
    window_ms       = kwargs.get('window_ms', 10)           # Window to calculate spectrogram over, ms
    step_ms         = kwargs.get('step_ms', 0.2)            # Overlap between windows (effectively step size), ms
    freq_emphasis   = kwargs.get('freq_emphasis', 3000)     # filt(f) = f * e^(-f / emphasis); use None to bypass
    eps = 1e-9      # Epsilon for logarithms

    nperseg = int(audio.frame_rate * window_ms * 1e-3)      # number of samples per spectrogram segment
    nstep = int(audio.frame_rate * step_ms * 1e-3)          # number of samples per spectrogram step
    noverlap = nperseg - nstep                              # number of overlap samples

    # Recalculate actual timestamps of spectrogram measurements
    actual_step = nstep / audio.frame_rate
    fingerprint_size = 2 * int(round(fingerprint_ms * 1e-3 / actual_step))

    frequencies = None
    times = None
    spectrogram = None
    window_size = nperseg / nstep
    spectrogram_offset = np.sqrt(0.5) * window_size                     # trying to figure out why this isn't half a window...smh
    # spectrogram_offset = 0.5 * window_size                              # maybe it should be??
    # print(spectrogram_offset * actual_step)
    n_time_taps = ((audio_data.shape[0] - nperseg) / nstep).__ceil__()  # ceil(samples / step size)
    n_freq_taps = 1 + nperseg // 2                                      # Nyquist of the spectrogram segment (nperseg)

    # print(fingerprint_size)

    if full_spectrogram:
        # print(f'audio: {audio_data.shape}, nperseg: {nperseg}, noverlap: {noverlap}, actual_step: {actual_step}, n_spectral_taps: {n_time_taps}, n_freq_taps: {n_freq_taps}')
        frequencies, times, spectrogram = signal.spectrogram(
            audio_data[:],
            fs=audio.frame_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False
        )
        # print(times[:10])
        # print(f'freqs, times, spec: {frequencies.shape}, {times.shape}, {spectrogram.shape}')
        splog = np.log2(spectrogram + eps)                              # Calculate in log domain

        if show_intermediate_plots:
            fig = plt.figure(figsize=(30, 6))
            plt.pcolormesh(times, frequencies * 1e-3, splog)
            plt.ylabel('Frequency [kHz]')
            plt.xlabel('Time [sec]')
            plt.title(f'Full spectrogram for {simfile_artist} - "{simfile_title}"')
            plt.show()

    ###################################################################
    # Use beat timing information to construct a "fingerprint"
    # of audio spectra around the time each beat takes place
    
    # Accumulator over beats, summed in the frequency domain
    acc = np.zeros((n_freq_taps, fingerprint_size))

    # Time-scale digest (frequencies flattened to single value,
    # each beat gets a fingerprint width)
    digest = np.zeros((0, fingerprint_size))

    # For each beat in the song that has a full
    # fingerprint's width of surrounding audio data:
    b = 0
    beat_indices = []
    t_last = -np.inf
    while True:
        t = engine.time_at(b)
        b += 1
        if (t < 0):
            # Too early
            continue
        if (t > audio.duration_seconds):
            # Too late
            break
        if (t - t_last < fingerprint_ms * 1e-3):
            # Too soon
            continue
        t_last = t

        # Because the spectrogram doesn't "start" until a full window is in view,
        # it has an inherent offset that amounts to half a window.
        # spectrogram_offset = window_ms * 0.5e-3
        # t_offset = (t - spectrogram_offset)

        t_s = int(round(t / actual_step - spectrogram_offset - fingerprint_size * 0.5))
        t_f = int(round(t / actual_step - spectrogram_offset + fingerprint_size * 0.5))
        if full_spectrogram:
            print(f'{t_s}~{t_f}: {times[t_s]:0.6f}~{times[t_f]:0.6f} -> {(times[t_f]+times[t_s])*0.5:0.6f} vs. {t:0.6f}')

        t_s = max(0,           t_s)
        t_f = min(n_time_taps, t_f)
        if (t_f - t_s != fingerprint_size):
            # Not enough data at this beat tbh
            continue
            
        if full_spectrogram:
            sp_snippet = splog[:, t_s:t_f]
        else:
            t_sample_s = t_s * nstep
            t_sample_f = t_f * nstep + nperseg - 1 
            # print(f't_sample: {t_sample_s}:{t_sample_f}')
            
            frequencies, times, spectrogram = signal.spectrogram(
                audio_data[t_sample_s:t_sample_f],
                fs=audio.frame_rate,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=False
            )
            # print(f'freqs, times, spec: {frequencies.shape}, {times.shape}, {spectrogram.shape}')
            sp_snippet = np.log2(spectrogram + eps)         # Calculate in log domain
        
        frequency_weights = 1
        if freq_emphasis is not None:
            # filt(f) = f * e^(-f / emphasis); use None to bypass
            frequency_weights = np.tile(frequencies * np.exp(-frequencies / freq_emphasis), [fingerprint_size, 1]).T
        spfilt = sp_snippet * frequency_weights

        # Accumulate, and add to digest
        acc += spfilt[:n_freq_taps, :]
        digest = np.vstack([digest, np.sum(spfilt, axis=0)])
        beat_indices.append(b-1)
        

    ###################################################################
    # Apply a convolution to detect the downbeat attack

    if kernel_type == BiasKernel.LOUDEST:
        # Loudest point of attack
        time_edge_kernel = np.array([
            [1, 3, 10, 3, 1],
            [1, 3, 10, 3, 1],
            [1, 3, 10, 3, 1],
            [1, 3, 10, 3, 1],
            [1, 3, 10, 3, 1]
        ])
    else:   # BiasKernel.RISING
        # Leading edge of attack
        time_edge_kernel = np.array([
            [1, 1, 0, -1, -1],
            [1, 1, 0, -1, -1],
            [1, 1, 0, -1, -1],
            [1, 1, 0, -1, -1],
            [1, 1, 0, -1, -1]
        ])
    edge_discard = time_edge_kernel.shape[1] // 2

    if kernel_target == KernelTarget.ACCUMULATOR:
        post_kernel = signal.convolve2d(acc,    time_edge_kernel, mode='same', boundary='wrap')
    else: # kernel_target == KernelTarget.DIGEST
        post_kernel = signal.convolve2d(digest, time_edge_kernel, mode='same', boundary='wrap')
    
    # Flatten convolved fingerprint to a value that only depends on time
    post_kernel_flat = np.sum(post_kernel, axis=0)
    fingerprint_times = np.arange(-fingerprint_size // 2, fingerprint_size // 2) * actual_step
    fingerprint_times_ms = fingerprint_times * 1e3

    # Choose the highest response to the convolution as the downbeat attack
    post_kernel_clip = post_kernel_flat[edge_discard:-edge_discard]
    i_max = np.argmax(post_kernel_clip)
    sync_bias_ms = fingerprint_times_ms[i_max + edge_discard] + magic_offset_ms
    probable_bias = guess_paradigm(sync_bias_ms, short_paradigm=False, **kwargs)
    # print(f'Sync bias: {sync_bias:0.3f} ({probable_bias})')

    # Calculate a confidence statistic based on the presence of conflicting
    # high-level response distant from the chosen peak
    v_clip = np.interp(post_kernel_clip, (min(post_kernel_clip), max(post_kernel_clip)), (0, 1))
    t_clip = fingerprint_times_ms[edge_discard:-edge_discard]
    v_std = np.std(v_clip)
    v_mean = np.mean(v_clip)
    v_median = np.median(v_clip)
    v_20 = np.percentile(v_clip, 20)
    v_80 = np.percentile(v_clip, 80)
    v_max = v_clip[i_max]
    v_max_check = np.vstack((np.zeros_like(v_clip), (v_clip - v_median) / (v_max - v_median)))
    v_max_rivaling = np.max(v_max_check, axis=0)
    t_close_check = np.vstack((np.zeros_like(t_clip), abs(t_clip - t_clip[i_max]) - _NEARNESS_OFFSET)) / _NEARNESS_SCALAR
    t_close_enough = np.max(t_close_check, axis=0)
    max_influence = np.power(v_max_rivaling, 4) * np.power(t_close_enough, 1.5)
    total_max_influence = np.sum(max_influence) / np.size(max_influence)
    sync_confidence = min(1, (1 - np.power(total_max_influence, 0.2)) / _THEORETICAL_UPPER)
    conv_interquintile = v_80 - v_20
    conv_stdev = v_std


    plot_tag_vars = kwargs.get('tag_vars', {}) 
    if len(plot_tag_vars) == 0:
        plot_tag = ''
        plot_tag_filename = ''
    else:
        plot_tag = ' (' + ', '.join(f'{k} = {v.format(kwargs.get(k))}' for k, v in plot_tag_vars.items()) + ')'
        plot_tag_filename = '-' + '-'.join(f'{k}_{v.format(kwargs.get(k))}' for k, v in plot_tag_vars.items())

    chart_tag = ''
    if chart is not None:
        fingerprint['steps_type'] = chart['STEPSTYPE']
        fingerprint['chart_slot'] = chart['DIFFICULTY']
        chart_tag = ' ' + slot_abbreviation(chart['STEPSTYPE'], chart['DIFFICULTY'], chart_index=chart_index, paradigm=guess_paradigm(sync_bias_ms, **kwargs))
    fingerprint['sample_rate']  = audio.frame_rate
    fingerprint['beat_digest']  = digest
    fingerprint['beat_indices'] = np.array(beat_indices)
    fingerprint['freq_domain']  = acc
    fingerprint['post_kernel']  = post_kernel
    fingerprint['convolution']  = post_kernel_flat
    fingerprint['frequencies']  = frequencies * 1e-3
    fingerprint['time_values']  = fingerprint_times_ms
    fingerprint['bias_result']  = sync_bias_ms
    fingerprint['confidence']   = sync_confidence
    fingerprint['conv_stdev']   = conv_stdev
    fingerprint['conv_quint']   = conv_interquintile
    # fingerprint['plots_title']  = \
    #     f'Sync fingerprint{plot_tag}\n{simfile_artist} - "{full_title}"{chart_tag}' + \
    #     f'\n{sync_bias_ms:+0.1f} ms bias ({probable_bias}), {round(sync_confidence*100):d}% conf'
    fingerprint['plots_title']  = \
        f'{full_title}{chart_tag}\n{simfile_artist}' + \
        f'\nSync fingerprint{plot_tag}: {sync_bias_ms:+0.1f} ms bias, {round(sync_confidence*100):d}% conf'
    
    sanitized_title = slugify(re.sub(r'[\\/]', '-', simfile_dir) + chart_tag, allow_unicode=False)
    target_axes = []
    target_figs = []
    for i in range(3):
        fig = plt.figure(figsize=(12, 6))
        target_figs.append(fig)
        target_axes.append(fig.add_subplot(1, 1, 1))
    
    plot_fingerprint(fingerprint, target_axes, flip_axes=True, **kwargs)

    # DEBUG: convolution output for confidence research
    with open(os.path.join(report_path, f'convolution-{sanitized_title}.csv'), 'w', newline='', encoding='ascii') as conv_fp:
        writer = csv.writer(conv_fp)
        for t, v in zip(fingerprint_times_ms, post_kernel_flat):
            writer.writerow([f'{t:0.6f}', f'{v:0.6f}'])

    for i, v in enumerate(['freqdomain', 'beatdigest', 'postkernel']):
        fig = target_figs[i]
        if show_intermediate_plots:
            fig.show()
        if save_plots:
            fig.savefig(os.path.join(report_path, f'bias-{v}-{sanitized_title}{plot_tag_filename}.png'))
        plt.close(fig)

    plot_hook_gui = kwargs.get('plot_hook_gui')
    if plot_hook_gui is not None:
        plot_fingerprint(fingerprint, plot_hook_gui.figure.get_axes(), hide_yticks=True, **kwargs)
        plot_hook_gui.canvas.draw()

    if show_intermediate_plots:
        # Quick and dirty plot of convolution response
        plt.plot(fingerprint_times_ms, post_kernel_flat)
        plt.show()

    # Done!
    return fingerprint


def check_paths(params):
    # Verify existence of root path
    root_path = params['root_path']
    if not os.path.isdir(root_path):
        raise Exception(f'Root directory doesn\'t exist: {root_path}')
    else:
        print(f"Root directory exists: {root_path}")

    # Verify existence of root path
    report_path = params['report_path']
    if report_path is None:
        report_path = os.path.join(root_path, '__bias-check')
        params['report_path'] = report_path
    if not os.path.isdir(report_path):
        try:
            os.makedirs(report_path)
            print(f"Report directory created: {report_path}")
        except Exception as e:
            raise Exception(f'Report directory can\'t be created: {report_path}')
    else:
        print(f"Report directory exists: {report_path}")


def setup_logging(report_path: str):
    # Set up logging
    log_stamp = timestamp()
    log_path = os.path.join(report_path, f'nine-or-null-{log_stamp}.log')
    log_fmt = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.basicConfig(
        filename=log_path,
        encoding='utf-8',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    for handler in logging.getLogger().handlers:
        handler.setFormatter(log_fmt)

    csv_path = os.path.join(report_path, f'nine-or-null-{log_stamp}.csv')
    return {
        'log_stamp': log_stamp,
        'log_path':  log_path,
        'csv_path':  csv_path
    }


def batch_process(root_path=None, **kwargs):
    gui_hook = kwargs.get('gui_hook')
    csv_hook = kwargs.get('csv_hook')

    if root_path is None:
        root_path = os.getcwd()
    simfile_dirs = []
    for r, d, f in os.walk(root_path):
        for fn in f:
            if os.path.splitext(fn)[1] in ['.ssc', '.sm']:
                simfile_dirs.append(os.path.relpath(r, root_path))
    
    simfile_dirs = sorted(list(set(simfile_dirs)))
    fingerprints = {}
    logging.info(f'Found {len(simfile_dirs)} simfiles in {root_path}')
    for d in simfile_dirs:
        logging.info(f'\t{d}')

    time_start = dt.utcnow()
    for i, p in enumerate(simfile_dirs):
        # Open simfile
        full_simfile_path = os.path.join(root_path, p)
        test_simfile_path = None
        for f in os.listdir(full_simfile_path):
            if os.path.splitext(f)[1] in ['.ssc', '.sm']:
                if (test_simfile_path is None) or (os.path.splitext(test_simfile_path)[1] == '.sm'):
                    test_simfile_path = os.path.join(full_simfile_path, f)
        if test_simfile_path is None:
            # How did this happen!
            continue

        try:
            time_elapsed = dt.utcnow() - time_start
            time_elapsed_str = timedelta_as_hhmmss(time_elapsed) + ' elapsed'
            if i > 0:
                time_expected = time_elapsed * (len(simfile_dirs) / i)
                time_elapsed_str += ', ' + timedelta_as_hhmmss(time_expected) + ' expected'
            logging.info(f'({i+1:d}/{len(simfile_dirs):d}: {time_elapsed_str})')
            if gui_hook is not None:
                gui_hook.SetStatusText(f'({i+1:d}/{len(simfile_dirs):d}: {time_elapsed_str}) Checking sync bias on {p}...')
                gui_hook.allow_to_update()

            base_simfile = simfile.open(test_simfile_path, strict=False)
            # Account for split timing.
            charts_within = [None]
            for chart_index, chart in enumerate(base_simfile.charts):
                if any(k in chart for k in ['OFFSET', 'BPMS', 'STOPS', 'DELAYS', 'WARPS']):
                    logging.info(f'{base_simfile.title}: {chart_index} ({chart.difficulty}) has split timing')
                    charts_within.append(chart_index)

            for split_chart in charts_within:
                fp = check_sync_bias(root_path, p, base_simfile, chart_index=split_chart, save_plots=True, show_intermediate_plots=False, **kwargs)
                sync_bias_ms = fp['bias_result']
                sync_confidence = fp['confidence']
                conv_quint = 'conv_quint' in fp and f"{fp['conv_quint']:0.6f}" or '----'
                conv_stdev = 'conv_stdev' in fp and f"{fp['conv_stdev']:0.6f}" or '----'
                
                chart_abbr = '*'
                if split_chart is not None:
                    chart = base_simfile.charts[split_chart]
                    chart_abbr = slot_abbreviation(chart['STEPSTYPE'], chart['DIFFICULTY'], chart_index=split_chart, paradigm=guess_paradigm(sync_bias_ms, **kwargs))
                    
                fp_lookup = os.path.join(p, chart_abbr)
                fingerprints[fp_lookup] = fp

                logging.info(f'\t{fp_lookup}')
                logging.info(f'\tderived sync bias = {sync_bias_ms:+0.1f} ms ({guess_paradigm(sync_bias_ms, short_paradigm=False, **kwargs)})')
                logging.info(f'\tbias confidence = {round(sync_confidence*100):3d}% (interquintile spread = {conv_quint}, stdev = {conv_stdev})')
                if gui_hook is not None:
                    row_index = len(fingerprints)-1
                    gui_hook.grid_results.InsertRows(row_index, 1)
                    gui_hook.grid_results.SetCellValue(row_index, 0, p)
                    gui_hook.grid_results.SetCellValue(row_index, 1, chart_abbr)
                    gui_hook.grid_results.SetCellValue(row_index, 2, f'{sync_bias_ms:+0.1f}')
                    gui_hook.grid_results.SetCellValue(row_index, 3, f'{round(sync_confidence*100):3d}%')
                    gui_hook.grid_results.SetCellValue(row_index, 4, guess_paradigm(sync_bias_ms, **kwargs))
                    gui_hook.grid_results.MakeCellVisible(row_index, 4)
                    for j in range(4):
                        gui_hook.grid_results.SetReadOnly(row_index, j)
                    gui_hook.grid_results.ForceRefresh()
                    gui_hook.allow_to_update()
                if csv_hook is not None:
                    row = {
                        'path': p,
                        'slot': chart_abbr,
                        'bias': f'{sync_bias_ms:0.3f}',
                        'conf': f'{sync_confidence:0.4f}',
                        'interquintile': f"{fp.get('conv_quint', None)}",
                        'stdev': f"{fp.get('conv_stdev', None)}",
                        'paradigm': guess_paradigm(sync_bias_ms, **kwargs),
                        'timestamp': timestamp(),
                        'sample_rate': fp.get('sample_rate', None)
                    }
                    for simfile_attr in ['title', 'titletranslit', 'subtitle', 'subtitletranslit', 'artist', 'artisttranslit', 'credit']:
                        row[simfile_attr] = base_simfile.get(simfile_attr.upper(), '')
                    for param in ['fingerprint_ms', 'window_ms', 'step_ms', 'kernel_type', 'kernel_target']:
                        row[param] = kwargs.get(param, None)
                    csv_hook.writerow(row)
        except Exception as e:
            logging.exception(e)

    return fingerprints
    

def batch_adjust(fingerprints, target_bias, **params):
    if target_bias == '+9ms':
        source_bias = 'null'
        bias_shift = +0.009
    elif target_bias == 'null':
        source_bias = '+9ms'
        bias_shift = -0.009
    else:
        raise Exception(f'What paradigm does "{target_bias}" represent?')

    logging.info(f'Converting charts with +9ms (In The Groove) bias to null (StepMania)...')
    affect_rows = params.get('affect_rows')
    for i, k in enumerate(fingerprints):
        if affect_rows is not None and i not in affect_rows:
            continue
        current_paradigm = fingerprints[k].get('bias_adjust', guess_paradigm(fingerprints[k]['bias_result'], **params))
        current_confidence = fingerprints[k].get('confidence', 100)
        if current_paradigm == source_bias and current_confidence >= params.get('confidence_limit', 0):
            logging.info(f'\t{k}')
            # Open simfile
            p, abbr = os.path.split(k)
            test_simfile_path = None
            for f in os.listdir(p):
                if os.path.splitext(f)[1] in ['.ssc', '.sm']:
                    if (test_simfile_path is None) or (os.path.splitext(test_simfile_path)[1] == '.sm'):
                        test_simfile_path = os.path.join(p, f)
            if test_simfile_path is None:
                # How did this happen!
                logging.info(f'What? Couldn\'t find a simfile at "{p}"')
                continue

            with simfile.mutate(
                test_simfile_path,
                backup_filename=test_simfile_path + ".oldsync",
                strict=False
            ) as sm:
                try:
                    if abbr == '*':
                        new_offset = float(sm.offset) + bias_shift
                        logging.info(f'\t{float(sm.offset):6.3f} -> {new_offset:6.3f}: {k}')
                        sm.offset = f'{new_offset:0.3f}'
                    else:
                        steps_type, chart_slot, chart_index = slot_expansion(abbr)
                        if chart_index is None:
                            chart_index = [i for i, c in enumerate(sm.charts) if c['STEPSTYPE'] == steps_type and c['DIFFICULTY'] == chart_slot][0]
                        prev_offset = float(sm.charts[chart_index].get('OFFSET', sm.offset))
                        new_offset = prev_offset + bias_shift
                        logging.info(f'\t{prev_offset:6.3f} -> {new_offset:6.3f}: {k}')
                        sm.charts[chart_index]['OFFSET'] = f'{new_offset:0.3f}'
                    fingerprints[k]['bias_result'] += bias_shift * 1e3
                    fingerprints[k]['bias_adjust'] = target_bias
                    
                    gui_hook = params.get('gui_hook')
                    if gui_hook is not None:
                        font_cell = gui_hook.grid_results.GetCellFont(i, 0)
                        gui_hook.grid_results.SetCellValue(i, 2, f"{fingerprints[k]['bias_result']:+0.1f}")
                        gui_hook.grid_results.SetCellValue(i, 4, target_bias)
                        for j in range(gui_hook.grid_results.GetNumberCols()):
                            gui_hook.grid_results.SetCellFont(i, j, font_cell.MakeBold())


                except Exception as e:
                    raise Exception(f'Something happened while adjusting bias for {test_simfile_path}') from e
    
    logging.info(f'Converting charts with +9ms (In The Groove) bias to null (StepMania)...Done!')
            
