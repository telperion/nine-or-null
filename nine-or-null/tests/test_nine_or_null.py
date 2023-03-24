import csv
import logging
import os
import sys

import nine_or_null

_TEST_DIR = r"C:\Games\ITGmania\Songs\ephemera v0.2"

def test_window_size():
    root_path = _TEST_DIR

    params = {k: v for k, v in nine_or_null._PARAM_DEFAULTS.items()}
    params['root_path'] = root_path
    params['report_path'] = os.path.join(root_path, '__bias-tests')
    params['fingerprint_ms'] = 40
    params['step_ms'] = 0.2
    params['tag_vars'] = {'window_ms': '{:d}'}
    
    try:
        nine_or_null.check_paths(params)
    except Exception as e:
        sys.exit(sys.exc_info())
    log_info = nine_or_null.setup_logging(params['report_path'])

    for window_ms in [5]: #[5, 10, 15, 20]:
        params['window_ms'] = window_ms

        # Recall parameters.
        header_str = f'+9ms or Null? v{nine_or_null._VERSION} (testbench)'
        logging.info(f"{'=' * 20}{header_str:^32s}{'=' * 20}")
        logging.info('Parameter settings:')
        for k, v in params.items():
            logging.info(f'\t{k} = {v}')

        with open(log_info['csv_path'], 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=nine_or_null._CSV_FIELDNAMES, extrasaction='ignore')
            writer.writeheader()
            params['csv_hook'] = writer

            fingerprints = nine_or_null.batch_process(**params)

        logging.info('-' * 72)
        logging.info(f"Sync bias report: {len(fingerprints)} fingerprints processed in {root_path}")

        paradigm_count = {}
        for paradigm in ['+9ms', 'null', '????']:
            paradigm_map = {k: v for k, v in fingerprints.items() if nine_or_null.guess_paradigm(v['bias_result']) == paradigm}
            logging.info(f"Files sync'd to {paradigm}: {len(paradigm_map)}")
            for k, v in paradigm_map.items():
                logging.info(f"\t{k:>50s}")
                logging.info(f"\t\tderived sync bias = {v['bias_result']:+0.1f} ms")
            paradigm_count[paradigm] = len(paradigm_map)

        paradigm_most = sorted([k for k in paradigm_count], key=lambda k: paradigm_count.get(k, 0))
        logging.info('=' * 72)
        logging.info(f'Pack sync paradigm: {paradigm_most[-1]}')
        logging.info('-' * 72)


if __name__ == '__main__':
    test_window_size()
    