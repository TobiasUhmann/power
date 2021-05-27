from collections import defaultdict
from pathlib import Path
from typing import List

import streamlit as st

from data.irt.text.text_dir import TextDir
from data.power.ruler_pkl import RulerPkl
from data.power.split.split_dir import SplitDir
from data.power.texter_pkl import TexterPkl
from models.fact import Fact
from power.aggregator import Aggregator


def render_predict_page():
    st.title('Power')

    split_dir_path = st.text_input('Path to Power Split', value='data/power/split/cde-50/')
    text_dir_path = st.text_input('Path to IRT Text Dir', value='data/irt/text/cde-irt-5-marked/')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()
    ent_to_lbl = split_dir.test_entities_tsv.load()

    st.write(ent_to_lbl)

    ent = st.number_input('Entity ID', min_value=0)

    #
    # Test Facts
    #

    st.title('Test Facts')

    test_facts_known: List[Fact] = split_dir.test_facts_known_tsv.load()

    ent_facts_known = [fact for fact in test_facts_known if fact.head == ent]

    for fact in ent_facts_known:
        st.write(fact)

    #
    # Texts
    #

    st.title('Texts')

    text_dir = TextDir(Path(text_dir_path))
    text_dir.check()

    ent_to_texts = text_dir.ow_test_sents_txt.load()
    ent_texts = ent_to_texts[ent]

    for ent_text in ent_texts:
        st.write(ent_text)

    #
    # Predict
    #

    ruler_pkl_path = st.text_input('Path to Ruler PKL', value='data/power/ruler-v2/final/cde-50-test.pkl')
    texter_pkl_path = st.text_input('Path to Texter PKL', value='data/power/texter-v2/cde-irt-5-marked_base.pkl')

    ruler_pkl = RulerPkl(ruler_pkl_path)
    texter_pkl = TexterPkl(texter_pkl_path)

    ruler_pkl.check()
    texter_pkl.check()

    ruler = ruler_pkl.load()
    texter = texter_pkl.load()

    power = Aggregator(texter, ruler, alpha=1.0)

    preds = power.predict(ent, list(ent_texts))
    st.write(preds)


def get_defaultdict():
    return defaultdict(list)
