_VERSION = '0.4.4'

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
    'consider_null':    'Consider charts close enough to 0ms bias to be "correct" under the null (StepMania) sync paradigm.',
    'consider_p9ms':    'Consider charts close enough to +9ms bias to be "correct" under the In The Groove sync paradigm.',
    'tolerance':        'If a simfile\'s sync bias lands within a paradigm ± this tolerance, that counts as "close enough".',
    'fingerprint_ms':   '[ms] Time margin on either side of the beat to analyze.',
    'window_ms':        '[ms] The spectrogram algorithm\'s moving window parameter.',
    'step_ms':          '[ms] Controls the spectrogram algorithm\'s overlap parameter, but expressed as a step size.',
    'kernel_target':    'Choose whether to convolve with the beat digest ("digest") or the spectral accumulator ("accumulator").',
    'kernel_type':      'Choose a kernel that responds to a rising edge ("rising") or local loudness ("loudest").',
    'magic_offset_ms':  '[ms] Add a constant value to the time of maximum kernel response. I haven\'t tracked the cause of this down yet. Might be related to attack perception?',
    'full_spectrogram': 'Analyze the full spectrogram in one go - this will make the program run slower...'
}

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
    'consider_null':    True,
    'consider_p9ms':    True,
    'tolerance':        3.0,
    'fingerprint_ms':   50,
    'window_ms':        10,
    'step_ms':          0.2,
    'kernel_target':    KernelTarget.DIGEST,
    'kernel_type':      BiasKernel.RISING,
    'magic_offset_ms':  0.0,
    'full_spectrogram': False
}

def timestamp():
    return dt.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]


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


def plot_fingerprint(fingerprint, target_axes, **kwargs):
    # Set up visuals to show the user what's going on.
    times_ms = fingerprint['time_values']
    frequencies_kHz = fingerprint['frequencies']
    acc = fingerprint['freq_domain']
    digest = fingerprint['beat_digest']
    sync_bias = fingerprint['bias_result']
    post_kernel_flat = fingerprint['convolution']
    post_kernel = fingerprint['post_kernel']
    plot_title = fingerprint['plots_title']
    fingerprint_ms = kwargs.get('fingerprint_ms', 50)
    magic_offset_ms = kwargs.get('magic_offset_ms', 0.0)
    kernel_target = kwargs.get('kernel_target', KernelTarget.DIGEST)
    hide_yticks = kwargs.get('hide_yticks', False)

    edge_discard = 5        # TODO: pull in from calling function I guess
    digest_axis = np.arange(digest.shape[0])

    time_ticks = np.hstack((
        np.arange(0, times_ms[ 0], -10),
        np.arange(0, times_ms[-1],  10)
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
            digest.shape[0] * 0.2,
            digest.shape[0] * 0.8
        ))

    # Accumulator in frequency domain
    ax = target_axes[0]
    ax.clear()
    pcm = ax.pcolormesh(times_ms, frequencies_kHz, acc)
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlabel('Time [msec]')
    ax.plot(times_ms + magic_offset_ms, post_kernel_over_freq, 'w-')
    ax.plot(frequency_line, frequencies_kHz, 'r-')
    ax.set_xticks(time_ticks)
    if hide_yticks:
        ax.set_yticks([])
    ax.set_xlim(-fingerprint_ms, fingerprint_ms)
    ax.get_figure().suptitle(plot_title)

    # Digest in beat domain
    ax = target_axes[1]
    ax.clear()
    pcm = ax.pcolormesh(times_ms, digest_axis, digest)
    pcm.set_clim(np.percentile(digest[:], 10), np.percentile(digest[:], 90))
    ax.set_ylabel('Beat Index')
    ax.set_xlabel('Time [msec]')
    ax.plot(times_ms + magic_offset_ms, post_kernel_over_beat, 'w-')
    ax.plot(beatindex_line, digest_axis, 'r-')
    ax.set_xticks(time_ticks)
    if hide_yticks:
        ax.set_yticks([])
    ax.set_xlim(-fingerprint_ms, fingerprint_ms)
    ax.get_figure().suptitle(plot_title)

    # Post-convolution plot
    ax = target_axes[2]
    ax.clear()
    if kernel_target == KernelTarget.ACCUMULATOR:
        pcm = ax.pcolormesh(times_ms, frequencies_kHz, post_kernel)
        ax.set_ylabel('Frequency [kHz]')
        ax.plot(times_ms + magic_offset_ms, post_kernel_over_freq, 'w-')
        ax.plot(frequency_line, frequencies_kHz, 'r-')
    else: # kernel_target == KernelTarget.DIGEST
        pcm = ax.pcolormesh(times_ms, digest_axis, post_kernel)
        ax.set_ylabel('Beat Index')
        ax.plot(times_ms + magic_offset_ms, post_kernel_over_beat, 'w-')
        ax.plot(beatindex_line, digest_axis, 'r-')
    pcm.set_clim(np.percentile(post_kernel[:], 3), np.percentile(post_kernel[:], 97))
    ax.set_xlabel('Time [msec]')
    ax.set_xticks(time_ticks)
    if hide_yticks:
        ax.set_yticks([])
    ax.set_xlim(-fingerprint_ms, fingerprint_ms)
    ax.get_figure().suptitle(plot_title)


def get_full_title(base_simfile):
    simfile_artist   = base_simfile.artisttranslit   or base_simfile.artist
    simfile_title    = base_simfile.titletranslit    or base_simfile.title
    simfile_subtitle = base_simfile.subtitletranslit or base_simfile.subtitle
    return f'{simfile_title}{simfile_subtitle and (" " + simfile_subtitle) or ""}'


def check_sync_bias(simfile_dir, base_simfile, chart=None, report_path=None, save_plots=True, show_intermediate_plots=False, **kwargs):
    fingerprint = {
        'beat_digest': None,    # Beat digest fingerprint (beat index vs. time)
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

    simfile_artist   = base_simfile.artisttranslit   or base_simfile.artist
    simfile_title    = base_simfile.titletranslit    or base_simfile.title
    simfile_subtitle = base_simfile.subtitletranslit or base_simfile.subtitle

    # Default to first chart
    if chart is None:
        chart = base_simfile.charts[0]

    # Account for split audio
    if not hasattr(chart, 'music') or chart.music is None:
        audio_path = os.path.join(simfile_dir, base_simfile.music)
    else:
        audio_path = os.path.join(simfile_dir, chart.music)

    engine = TimingEngine(TimingData(base_simfile, chart))

    ###################################################################
    # Load audio using pydub
    audio_ext = os.path.splitext(audio_path)[1]
    audio = AudioSegment.from_file(audio_path, format=audio_ext[1:])
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
            audio_data[:, 0],       # Mono left channel please
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
    while True:
        t = engine.time_at(b)
        b += 1
        if (t < 0):
            # Too early
            continue
        if (t > audio.duration_seconds):
            # Too late
            break

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
                audio_data[t_sample_s:t_sample_f, 0],       # Mono left channel please
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
    sync_bias_ms = fingerprint_times_ms[np.argmax(post_kernel_flat[edge_discard:-edge_discard]) + edge_discard] + magic_offset_ms
    probable_bias = guess_paradigm(sync_bias_ms, short_paradigm=False, **kwargs)
    # print(f'Sync bias: {sync_bias:0.3f} ({probable_bias})')

    full_title = get_full_title(base_simfile)

    plot_tag_vars = kwargs.get('tag_vars', {}) 
    if len(plot_tag_vars) == 0:
        plot_tag = ''
        plot_tag_filename = ''
    else:
        plot_tag = ' (' + ', '.join(f'{k} = {v.format(kwargs.get(k))}' for k, v in plot_tag_vars.items()) + ')'
        plot_tag_filename = '-' + '-'.join(f'{k}_{v.format(kwargs.get(k))}' for k, v in plot_tag_vars.items())

    fingerprint['sample_rate'] = audio.frame_rate
    fingerprint['beat_digest'] = digest
    fingerprint['freq_domain'] = acc
    fingerprint['post_kernel'] = post_kernel
    fingerprint['convolution'] = post_kernel_flat
    fingerprint['frequencies'] = frequencies * 1e-3
    fingerprint['time_values'] = fingerprint_times_ms
    fingerprint['bias_result'] = sync_bias_ms
    fingerprint['plots_title'] = f'Sync fingerprint{plot_tag}\n{simfile_artist} - "{full_title}"\nSync bias: {sync_bias_ms:+0.1f} ms ({probable_bias})'
    
    sanitized_title = slugify(full_title, allow_unicode=False)
    target_axes = []
    target_figs = []
    for i in range(3):
        fig = plt.figure(figsize=(6, 6))
        target_figs.append(fig)
        target_axes.append(fig.add_subplot(1, 1, 1))
    
    plot_fingerprint(fingerprint, target_axes, **kwargs)

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
                simfile_dirs.append(r)
    
    simfile_dirs = sorted(list(set(simfile_dirs)))
    fingerprints = {}
    logging.info(f'Found {len(simfile_dirs)} simfiles in {root_path}')
    for d in simfile_dirs:
        logging.info(f'\t{os.path.relpath(d, root_path)}')

    time_start = dt.utcnow()
    for i, p in enumerate(simfile_dirs):
        # Open simfile
        test_simfile_path = None
        for f in os.listdir(p):
            if os.path.splitext(f)[1] in ['.ssc', '.sm']:
                if (test_simfile_path is None) or (os.path.splitext(test_simfile_path)[1] == '.sm'):
                    test_simfile_path = os.path.join(p, f)
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
                gui_hook.SetStatusText(f'({i+1:d}/{len(simfile_dirs):d}: {time_elapsed_str}) Checking sync bias on {os.path.relpath(p, root_path)}...')
                gui_hook.allow_to_update()
            base_simfile = simfile.open(test_simfile_path)
            fingerprints[p] = check_sync_bias(p, base_simfile, chart=None, save_plots=True, show_intermediate_plots=False, **kwargs)
            sync_bias_ms = fingerprints[p]['bias_result']
            logging.info(f'\t{p}')
            logging.info(f'\tderived sync bias = {sync_bias_ms:+0.1f} ms ({guess_paradigm(sync_bias_ms, short_paradigm=False, **kwargs)})')
            if gui_hook is not None:
                gui_hook.grid_results.InsertRows(i, 1)
                gui_hook.grid_results.SetCellValue(i, 0, os.path.relpath(p, root_path))
                gui_hook.grid_results.SetCellValue(i, 1, '----')
                gui_hook.grid_results.SetCellValue(i, 2, f'{sync_bias_ms:+0.1f}')
                gui_hook.grid_results.SetCellValue(i, 3, guess_paradigm(sync_bias_ms, **kwargs))
                gui_hook.grid_results.MakeCellVisible(i, 3)
                for j in range(4):
                    gui_hook.grid_results.SetReadOnly(i, j)
                gui_hook.grid_results.ForceRefresh()
                gui_hook.allow_to_update()
            if csv_hook is not None:
                row = {
                    'path': os.path.relpath(p, root_path),
                    'slot': '----',
                    'bias': f'{sync_bias_ms:0.3f}',
                    'paradigm': guess_paradigm(sync_bias_ms, **kwargs),
                    'timestamp': timestamp(),
                    'sample_rate': fingerprints[p].get('sample_rate', None)
                }
                for simfile_attr in ['title', 'titletranslit', 'subtitle', 'subtitletranslit', 'artist', 'artisttranslit', 'credit']:
                    row[simfile_attr] = base_simfile[simfile_attr.upper()]
                for param in ['fingerprint_ms', 'window_ms', 'step_ms', 'kernel_type', 'kernel_target']:
                    row[param] = kwargs.get(param, None)
                csv_hook.writerow(row)
        except Exception as e:
            logging.exception(e)

    return fingerprints
    