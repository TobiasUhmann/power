from typing import List, Set, Tuple

import streamlit as st

from app.common import add_sidebar_param_split_dir, add_sidebar_param_text_dir, add_sidebar_param_ruler_pkl, \
    add_sidebar_param_texter_pkl
from data.irt.text.text_dir import TextDir
from data.power.ruler_pkl import RulerPkl
from data.power.split.facts_tsv import Fact
from data.power.split.split_dir import SplitDir
from data.power.texter_pkl import TexterPkl
from models.ent import Ent
from power.aggregator import Aggregator


def add_predict_page():
    st.sidebar.header('Config')
    split_dir, text_dir, ruler_pkl, texter_pkl = _add_sidebar()

    st.title('Predict')
    _create_main_page(split_dir, text_dir, ruler_pkl, texter_pkl)


def _add_sidebar():
    """ Create sidebar and return all its contained parameters """

    split_dir = add_sidebar_param_split_dir()
    text_dir = add_sidebar_param_text_dir()
    ruler_pkl = add_sidebar_param_ruler_pkl()
    texter_pkl = add_sidebar_param_texter_pkl()

    return split_dir, text_dir, ruler_pkl, texter_pkl


def _create_main_page(split_dir: SplitDir, text_dir: TextDir, ruler_pkl: RulerPkl, texter_pkl: TexterPkl) -> None:
    st.write("Select a valid or test entity, view its known and unknown facts"
             " as well as its texts, and observe the Power model's predictions.")

    st.write("Ideally, the top predictions should match the entity's unknown"
             " facts. The predictions can be expanded to see the model's eplanation"
             " for the predicted fact, i.e. the rules and/or text prioritization,"
             " depending on whether the fact has been predicted by Ruler and/or"
             " Texter.")

    ent, subset = _select_entity(split_dir)

    _show_entity_facts(split_dir, ent, subset)

    ent_texts = _show_entity_texts(text_dir, ent, subset)

    _show_predictions(ruler_pkl, texter_pkl, ent, ent_texts)


def _select_entity(split_dir: SplitDir) -> Tuple[Ent, str]:
    """
    Show
    - Radio buttons that allows switching between valid and test entities
    - Spinner to select entity by ID
    - Text field that shows the selected entity's label
    """

    st.header('Entity')

    valid_test_selection = st.radio('', ['Valid Entity', 'Test Entity'])

    if valid_test_selection == 'Valid Entity':
        subset = 'valid'
        ent_to_lbl = split_dir.valid_entities_tsv.load()

    elif valid_test_selection == 'Test Entity':
        subset = 'test'
        ent_to_lbl = split_dir.test_entities_tsv.load()

    else:
        raise ValueError()

    min_id = min(ent_to_lbl.keys())
    max_id = max(ent_to_lbl.keys())

    cols = st.beta_columns([25, 75])
    ent_id = cols[0].number_input('ID', key='_ent_id', min_value=min_id, max_value=max_id, value=min_id)
    cols[1].text_input('Label', key='_ent_lbl', value=ent_to_lbl[ent_id])

    ent = Ent(ent_id, ent_to_lbl[ent_id])

    return ent, subset


def _show_entity_facts(split_dir: SplitDir, ent: Ent, subset: str) -> None:
    if subset == 'valid':
        known_facts: List[Fact] = split_dir.valid_facts_known_tsv.load()
        unknown_facts: List[Fact] = split_dir.valid_facts_unknown_tsv.load()
    elif subset == 'test':
        known_facts: List[Fact] = split_dir.test_facts_known_tsv.load()
        unknown_facts: List[Fact] = split_dir.test_facts_unknown_tsv.load()
    else:
        raise ValueError()

    known_ent_facts = [fact for fact in known_facts if fact.head == ent.id]
    unknown_ent_facts = [fact for fact in unknown_facts if fact.head == ent.id]

    st.header('Known Facts')
    for fact in known_ent_facts:
        st.write(f'{fact.head_lbl}, {fact.rel_lbl}, {fact.tail_lbl}')

    st.header('Unknown Facts')
    for fact in unknown_ent_facts:
        st.write(f'{fact.head_lbl}, {fact.rel_lbl}, {fact.tail_lbl}')


def _show_entity_texts(text_dir: TextDir, ent: Ent, subset: str) -> Set[str]:
    if subset == 'valid':
        ent_to_texts = text_dir.ow_valid_sents_txt.load()
    elif subset == 'test':
        ent_to_texts = text_dir.ow_test_sents_txt.load()
    else:
        raise ValueError()

    ent_texts = ent_to_texts[ent.id]

    st.header('Texts')
    for i, text in enumerate(ent_texts):
        st.text_area(label='', value=text, key=f'{ent.id}_{i}')

    return ent_texts


def _show_predictions(ruler_pkl: RulerPkl, texter_pkl: TexterPkl, ent: Ent, ent_texts: Set[str]):
    """
    Load the Power model, make predictions for the selected entity,
    and list the predictions as expandable sections that reveal the
    predictions' rules and text prioritizations.
    """

    st.header('Predictions')

    ruler = ruler_pkl.load()
    texter = texter_pkl.load().cpu()

    power = Aggregator(texter, ruler, alpha=0.5)

    preds = power.predict(ent, list(ent_texts))

    for pred in preds:
        expander_title = '{:.2f}% - {}, {}, {}'.format(
            pred.conf * 100, pred.fact.head.lbl, pred.fact.rel.lbl, pred.fact.tail.lbl)

        with st.beta_expander(expander_title, expanded=False):
            st.subheader('Rules')
            if pred.rules:
                for rule in pred.rules:
                    st.write(rule)
            else:
                st.write('None')

            st.subheader('Texts')
            if pred.sents:
                for sent in pred.sents:
                    st.write(sent)
            else:
                st.write('None')
