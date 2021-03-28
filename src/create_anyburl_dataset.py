import logging
from argparse import ArgumentParser
from pathlib import Path

from data.anyburl.anyburl_dir import AnyburlDir
from data.anyburl.facts_tsv import Fact
from data.ryn.split.split_dir import SplitDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()
    
    create_anyburl_dataset(args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) Ryn Split Directory')

    parser.add_argument('anyburl_dir', metavar='anyburl-dir',
                        help='Path to (output) AnyBURL Directory')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('anyburl-dir', args.anyburl_dir))

    return args


def create_anyburl_dataset(args):
    split_dir_path = args.split_dir
    anyburl_dir_path = args.anyburl_dir

    #
    # Check that (input) Ryn Split Directory exists
    #

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()
    
    #
    # Create (output) AnyBURL Directory
    #
    
    anyburl_dir = AnyburlDir(Path(anyburl_dir_path))
    anyburl_dir.create()
    
    #
    # Load facts from Ryn Split Directory, transform, and save to AnyBURL Directory
    #

    ent_to_lbl = split_dir.ent_labels_txt.load()
    rel_to_lbl = split_dir.rel_labels_txt.load()
    
    def stringify_ent(ent):
        return f"{ent}_{ent_to_lbl[ent].replace(' ', '_')}"
    
    def stringify_rel(rel):
        return f"{rel}_{rel_to_lbl[rel].replace(' ', '_')}"

    cw_train_triples = split_dir.cw_train_triples_txt.load()
    cw_train_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                      for head, rel, tail in cw_train_triples]
    anyburl_dir.cw_train_facts_tsv.save(cw_train_facts)

    cw_valid_triples = split_dir.cw_valid_triples_txt.load()
    cw_valid_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                      for head, rel, tail in cw_valid_triples]
    anyburl_dir.cw_valid_facts_tsv.save(cw_valid_facts)

    ow_valid_triples = split_dir.ow_valid_triples_txt.load()
    ow_valid_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                      for head, rel, tail in ow_valid_triples]
    anyburl_dir.ow_valid_facts_tsv.save(ow_valid_facts)

    ow_test_triples = split_dir.ow_test_triples_txt.load()
    ow_test_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                      for head, rel, tail in ow_test_triples]
    anyburl_dir.ow_test_facts_tsv.save(ow_test_facts)
    

if __name__ == '__main__':
    main()
