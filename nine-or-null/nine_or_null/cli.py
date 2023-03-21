import argparse
import csv
import os
import re
import sys
import logging
import textwrap

from . import FloatRange, BiasKernel, KernelTarget, batch_process, guess_paradigm, _VERSION, _CSV_FIELDNAMES, _PARAMETERS
from .gui import start_gui

def start_cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""
            +9ms or Null? v{_VERSION} is a StepMania simfile unbiasing utility.

            This utility can determine whether the sync bias of a simfile or a pack is +9ms (In The Groove) or null (general StepMania). A future version will also offer to unify it under one of those two options.
            It is not meant to perform a millisecond-perfect sync!

            You can read more about the origins of the +9ms sync bias here:
            Club Fantastic Wiki's explanation
                https://wiki.clubfantastic.dance/Sync#itg-offset-and-the-9ms-bias
            Ash's discussion of solutions @ meow.garden
                https://meow.garden/killing-the-9ms-bias"""),
        epilog=textwrap.dedent("""
            Sync bias algorithm and program written by Telperion.
            Credit to beware for sprouting the idea of an aligned audio fingerprint to examine sync."""),
    )
    parser.add_argument('-v', '--version',
        action='version',
        version=_VERSION
    )
    parser.add_argument('root_path',
        help=_PARAMETERS['root_path'],
        nargs='?'
    )
    parser.add_argument('-r', '--report_path',
        help=_PARAMETERS['report_path']
    )
    parser.add_argument('--consider_null', '--cn',
        help=_PARAMETERS['consider_null'],
        action='store_false',
    )
    parser.add_argument('--consider_p9ms', '--c9',
        help=_PARAMETERS['consider_p9ms'],
        action='store_false',
    )
    parser.add_argument('-t', '--tolerance',
        help=_PARAMETERS['tolerance'],
        choices=FloatRange(0, 3.5),
        default=3.0,
        type=float
    )
    parser.add_argument('-f', '--fingerprint',
        help=_PARAMETERS['fingerprint_ms'],
        dest='fingerprint_ms',
        choices=FloatRange(10, 50),
        default=50,
        type=float
    )
    parser.add_argument('-w', '--window',
        help=_PARAMETERS['window_ms'],
        dest='window_ms',
        choices=FloatRange(2, 10),
        default=10,
        type=float
    )
    parser.add_argument('-s', '--step',
        help=_PARAMETERS['step_ms'],
        dest='step_ms',
        choices=FloatRange(0.01, 1.0),
        default=0.2,
        type=float
    )
    parser.add_argument('--magic-offset',
        help=_PARAMETERS['magic_offset_ms'],
        dest='magic_offset_ms',
        choices=FloatRange(-5.0, 5.0),
        default=2.0,
        type=float
    )
    parser.add_argument('--kernel-target',
        help=_PARAMETERS['kernel_target'],
        default=0,
        type=lambda k: isinstance(k, str) and {'digest': 0, 'acc': 1, 'accumulator': 1}[k.lower()] or KernelTarget(k)
    )
    parser.add_argument('--kernel-type',
        help=_PARAMETERS['kernel_type'],
        default=0,
        type=lambda k: isinstance(k, str) and {'rising': 0, 'loudest': 1}[k.lower()] or BiasKernel(k)
    )
    parser.add_argument('--full-spectrogram',
        dest='full_spectrogram',
        action='store_true',
        help=_PARAMETERS['full_spectrogram']
    )
    args = parser.parse_args()
    params = {k: getattr(args, k) for k in _PARAMETERS}

    if args.root_path is None:
        print('No path to simfile or pack provided. Entering GUI mode...')
        start_gui()
        print('Exiting GUI mode. Thank you for playing!')
    else:
        print('Entering CLI mode...')
        
        root_path = args.root_path

        # Verify existence of root path
        if not os.path.isdir(root_path):
            sys.exit(f'Root directory doesn\'t exist: {root_path}')
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
                sys.exit(f'Report directory can\'t be created: {report_path}')
        else:
            print(f"Report directory exists: {report_path}")

        # Set up logging
        log_path = os.path.join(report_path, 'nine-or-null.log')
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

        csv_path = os.path.join(report_path, 'nine-or-null.csv')


        # Recall parameters.
        header_str = f'+9ms or Null? v{_VERSION} (CLI)'
        logging.info(f"{'=' * 20}{header_str:^32s}{'=' * 20}")
        logging.info('Parameter settings:')
        for k, v in params.items():
            logging.info(f'\t{k} = {v}')

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDNAMES, extrasaction='ignore')
            writer.writeheader()
            params['csv_hook'] = writer

            fingerprints = batch_process(**params)

        logging.info('-' * 72)
        logging.info(f"Sync bias report: {len(fingerprints)} fingerprints processed in {root_path}")

        paradigm_count = {}
        for paradigm in ['+9ms', 'null', '????']:
            paradigm_map = {k: v for k, v in fingerprints.items() if guess_paradigm(v['bias_result']) == paradigm}
            logging.info(f"Files sync'd to {paradigm}: {len(paradigm_map)}")
            for k, v in paradigm_map.items():
                logging.info(f"\t{k:>50s}")
                logging.info(f"\t\tderived sync bias = {v['bias_result']:+0.1f} ms")
            paradigm_count[paradigm] = len(paradigm_map)

        paradigm_most = sorted([k for k in paradigm_count], key=lambda k: paradigm_count.get(k, 0))
        logging.info('=' * 72)
        logging.info(f'Pack sync paradigm: {paradigm_most[-1]}')
        logging.info('-' * 72)

        # Done!
        print('Exiting CLI mode. Thank you for playing!')
    

if __name__ == '__main__':
    start_cli()