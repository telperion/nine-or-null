import os
import sys
import logging

from . import process_pack, guess_paradigm, slugify

if __name__ == '__main__':
    pack_dir = os.getcwd()
    if len(sys.argv) > 1:
        pack_dir = sys.argv[1]
    
    # I need logging lol
    log_path = os.path.join(pack_dir, '__bias-check', f'__{slugify(os.path.split(pack_dir)[1])}-results.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        encoding='utf-8',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    sync_bias_map = process_pack(pack_dir)

    logging.info('-' * 72)
    logging.info(f'Sync bias report: {len(sync_bias_map)} simfiles processed in {pack_dir}')

    paradigm_count = {}
    for paradigm in ['probably +9ms', 'probably null', 'unclear paradigm']:
        paradigm_map = {k: v for k, v in sync_bias_map.items() if guess_paradigm(v) == paradigm}
        logging.info(f"Files sync'd to {paradigm:^16s}: {len(paradigm_map)}")
        for k, v in paradigm_map.items():
            logging.info(f'{k:>50s}: derived sync bias = {v:+0.3f}')
        paradigm_count[paradigm] = len(paradigm_map)

    paradigm_most = sorted([k for k in paradigm_count], key=lambda k: paradigm_count.get(k, 0))
    logging.info('=' * 72)
    logging.info(f'Pack sync paradigm: {paradigm_most[-1]}')
    logging.info('-' * 72)

    # process_pack(r"C:\Games\ITGmania\Songs\ITGAlex's Compilation 4")
    # process_pack(r"C:\Games\ITGmania\Songs\DDR XX (2023-02-09)")