from collections import defaultdict
from pprint import pformat
from typing import List

import streamlit as st

from app.common import add_sidebar_param_split_dir, add_sidebar_param_text_dir, add_sidebar_param_ruler_pkl, \
    add_sidebar_param_texter_pkl
from data.irt.text.text_dir import TextDir
from data.power.ruler_pkl import RulerPkl
from data.power.split.split_dir import SplitDir
from data.power.texter_pkl import TexterPkl
from models.fact import Fact
from power.aggregator import Aggregator


def add_predict_page():
    st.sidebar.header('Config')
    split_dir, text_dir, ruler_pkl, texter_pkl = _add_sidebar()

    st.title('Predict')
    _add_main_page(split_dir, text_dir, ruler_pkl, texter_pkl)


def _add_sidebar():
    """ Create sidebar and return all its contained parameters """

    split_dir = add_sidebar_param_split_dir()
    text_dir = add_sidebar_param_text_dir()
    ruler_pkl = add_sidebar_param_ruler_pkl()
    texter_pkl = add_sidebar_param_texter_pkl()

    return split_dir, text_dir, ruler_pkl, texter_pkl


def _add_main_page(split_dir: SplitDir, text_dir: TextDir, ruler_pkl: RulerPkl, texter_pkl: TexterPkl) -> None:
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

    ent_to_texts = text_dir.ow_test_sents_txt.load()
    ent_texts = ent_to_texts[ent]

    for ent_text in ent_texts:
        st.write(ent_text)

    #
    # Predict
    #

    ruler = ruler_pkl.load()
    texter = texter_pkl.load().cpu()

    power = Aggregator(texter, ruler, alpha=1.0)

    preds = power.predict(ent, list(ent_texts))
    for pred in preds:
        st.write(pred.fact.head, pred.fact.rel, pred.fact.tail, pred.conf)
