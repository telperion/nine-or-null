import os
import sys
import logging

from . import BiasKernel, KernelTarget, batch_process, guess_paradigm, _VERSION
from .gui import start_gui

def start_cli():
    print(f'+9ms or Null? v{_VERSION}')

    if len(sys.argv) <= 1:
        print('No path to simfile or pack provided. Entering GUI mode...')
        start_gui()
        print('Exiting GUI mode. Thank you for playing!')
    else:
        print('Entering CLI mode...')
        
        root_path = sys.argv[1]

        # Verify existence of root path
        if not os.path.isdir(root_path):
            sys.exit(f'Root directory doesn\'t exist: {root_path}')
        else:
            print(f"Root directory exists: {root_path}")

        # Verify existence of root path
        report_path = os.path.join(root_path, '__bias-check')
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

        # Default parameters.
        params = {}
        params['root_path']      = root_path
        params['report_path']    = report_path
        params['consider_null']  = True
        params['consider_p9ms']  = True
        params['tolerance']      = 3.0
        params['fingerprint_ms'] = 50
        params['window_ms']      = 10
        params['step_ms']        = 0.20
        params['kernel_target']  = KernelTarget.DIGEST
        params['kernel_type']    = BiasKernel.RISING
        params['magic_offset']   = 2.0

        # Recall parameters.
        header_str = f'+9ms or Null? v{_VERSION} (CLI)'
        logging.info(f"{'=' * 20}{header_str:^32s}{'=' * 20}")
        logging.info('Parameter settings:')
        for k, v in params.items():
            logging.info(f'\t{k} = {v}')

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