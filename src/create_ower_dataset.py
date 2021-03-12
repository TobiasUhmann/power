import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
from typing import Dict, Set, List, Tuple

from dao.ower.ower_dir import OwerDir
from dao.ryn.ryn_dir import RynDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    ## Parse args

    args = parse_args()

    ryn_dir_path = args.ryn_dir
    ower_dir_path = args.ower_dir

    class_count = args.class_count
    sent_count = args.sent_count

    ## Assert that (input) Ryn Directory exists

    ryn_dir = RynDir(Path(ryn_dir_path))
    ryn_dir.check()

    ## Create (output) OWER Directory if it does not exist yet

    ower_dir = OwerDir(Path(ower_dir_path))
    ower_dir.create()

    ## Load Ryn Triples TXTs

    logging.info('Load Ryn Triples TXTs...')

    split_dir = ryn_dir.split_dir
    cw_train_triples: List[Tuple[int, int, int]] = split_dir.cw_train_triples_txt.load()
    cw_valid_triples: List[Tuple[int, int, int]] = split_dir.cw_valid_triples_txt.load()
    ow_valid_triples: List[Tuple[int, int, int]] = split_dir.ow_valid_triples_txt.load()
    ow_test_triples: List[Tuple[int, int, int]] = split_dir.ow_test_triples_txt.load()

    train_triples = cw_train_triples + cw_valid_triples
    valid_triples = ow_valid_triples
    test_triples = ow_test_triples

    ## Save triples to OWER Triples DBs

    logging.info('Save triples to OWER Triples DBs...')

    train_triples_db = ower_dir.tmp_dir.train_triples_db
    train_triples_db.create_triples_table()
    train_triples_db.insert_triples(train_triples)

    valid_triples_db = ower_dir.tmp_dir.valid_triples_db
    valid_triples_db.create_triples_table()
    valid_triples_db.insert_triples(valid_triples)

    test_triples_db = ower_dir.tmp_dir.test_triples_db
    test_triples_db.create_triples_table()
    test_triples_db.insert_triples(test_triples)

    ## Copy Ryn Label TXTs to OWER Dir

    copyfile(ryn_dir.split_dir.ent_labels_txt.path, ower_dir.ent_labels_txt.path)
    copyfile(ryn_dir.split_dir.rel_labels_txt.path, ower_dir.rel_labels_txt.path)

    ## Query most common classes and write them to Classes TSV

    rel_tail_supps = ower_dir.tmp_dir.train_triples_db.select_top_rel_tails(class_count)

    ent_to_label = ower_dir.ent_labels_txt.load()
    rel_to_label = ower_dir.rel_labels_txt.load()

    ent_count = len(ent_to_label)
    rel_tail_freq_labels = [(rel, tail, supp / ent_count, f'{rel_to_label[rel]} {ent_to_label[tail]}')
                            for rel, tail, supp in rel_tail_supps]

    ower_dir.classes_tsv.save(rel_tail_freq_labels)

    ## Query classes' entities

    logging.info("Query classes' entities...")

    train_class_ents = []
    valid_class_ents = []
    test_class_ents = []

    for rel, tail, _ in rel_tail_supps:
        class_ents = ower_dir.tmp_dir.train_triples_db.select_heads_with_rel_tail(rel, tail)
        train_class_ents.append(class_ents)

    for rel, tail, _ in rel_tail_supps:
        class_ents = ower_dir.tmp_dir.valid_triples_db.select_heads_with_rel_tail(rel, tail)
        valid_class_ents.append(class_ents)

    for rel, tail, _ in rel_tail_supps:
        class_ents = ower_dir.tmp_dir.test_triples_db.select_heads_with_rel_tail(rel, tail)
        test_class_ents.append(class_ents)

    ## Create OWER Sample TSVs

    logging.info('Create OWER Sample TSVs...')

    train_ent_to_sents: Dict[int, Set[str]] = ryn_dir.text_dir.cw_train_sents_txt.load()
    valid_ent_to_sents: Dict[int, Set[str]] = ryn_dir.text_dir.ow_valid_sents_txt.load()
    test_ent_to_sents: Dict[int, Set[str]] = ryn_dir.text_dir.ow_test_sents_txt.load()

    def get_samples(ent_to_sents, class_ents):
        """
        :param ent_to_sents: {ent: {sent}}
        :param class_ents: [[ent]]
        :return: [(ent, label, [has class], [sent])
        """

        ent_lbl_classes_sents_list = []

        for ent, sents in ent_to_sents.items():

            ent_classes = []
            for class_ in range(len(class_ents)):
                ent_classes.append(int(ent in class_ents[class_]))

            if len(sents) < sent_count:
                logging.warning(f"Entity '{ent_to_label[ent]}' ({ent}) has less than {sent_count} sentences. Skipping")
                continue

            ent_lbl_classes_sents_list.append((ent, ent_to_label[ent], ent_classes, sents))

        return ent_lbl_classes_sents_list

    train_samples = get_samples(train_ent_to_sents, train_class_ents)
    valid_samples = get_samples(valid_ent_to_sents, valid_class_ents)
    test_samples = get_samples(test_ent_to_sents, test_class_ents)

    ower_dir.train_samples_tsv.save(train_samples)
    ower_dir.valid_samples_tsv.save(valid_samples)
    ower_dir.test_samples_tsv.save(test_samples)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('ryn_dir', metavar='ryn-dir',
                        help='Path to (input) Ryn Directory')

    parser.add_argument('ower_dir', metavar='ower-dir',
                        help='Path to (output) OWER Directory')

    default_class_count = 100
    parser.add_argument('--class-count', dest='class_count', type=int, metavar='INT', default=default_class_count,
                        help='Number of classes (default: {})'.format(default_class_count))

    default_sent_count = 5
    parser.add_argument('--sent-count', dest='sent_count', type=int, metavar='INT', default=default_sent_count,
                        help='Number of sentences per entity. Entities for which not enough sentences'
                             ' are availabe are dropped. (default: {})'.format(default_sent_count))

    args = parser.parse_args()

    ## Log applied config

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ryn-dir', args.ryn_dir))
    logging.info('    {:24} {}'.format('ower-dir', args.ower_dir))
    logging.info('    {:24} {}'.format('--class-count', args.class_count))
    logging.info('    {:24} {}'.format('--sent-count', args.sent_count))

    return args


if __name__ == '__main__':
    main()
