from typing import Dict, List, Set

import streamlit as st

from app.common import add_sidebar_param_split_dir, add_sidebar_param_text_dir
from data.irt.text.text_dir import TextDir
from data.power.split.split_dir import SplitDir
from models.fact import Fact


def add_browse_dataset_page():
    st.sidebar.header('Config')
    split_dir, text_dir = _add_sidebar()

    st.title('Browse Dataset')
    _add_main_page(split_dir, text_dir)


def _add_sidebar():
    """ Create sidebar and return all its contained parameters """

    split_dir = add_sidebar_param_split_dir()
    text_dir = add_sidebar_param_text_dir()

    return split_dir, text_dir


def _add_main_page(split_dir: SplitDir, text_dir: TextDir):
    _add_main_section_train_entities(split_dir, text_dir)
    _add_main_section_valid_entities(split_dir, text_dir)
    _add_main_section_test_entities(split_dir, text_dir)


def _add_main_section_train_entities(split_dir: SplitDir, text_dir: TextDir) -> None:
    """
    Add main section that allows selecting a train entity to show
    its label, facts, and texts
    """

    # Load train entities
    train_ent_to_lbl: Dict[int, str] = split_dir.train_entities_tsv.load()

    with st.beta_expander('Train Entities ({})'.format(len(train_ent_to_lbl)), expanded=False):
        ent = prompt_entity('Entity', train_ent_to_lbl)

        # Load all train facts and show selected entity's facts
        train_facts: List[Fact] = split_dir.train_facts_tsv.load()
        show_entity_facts('Facts', train_facts, ent)

        # Load all train texts and show selected entity's texts
        cw_train_texts: Dict[int, Set[str]] = text_dir.cw_train_sents_txt.load()
        show_entity_texts('Texts', cw_train_texts, ent)


def _add_main_section_valid_entities(split_dir: SplitDir, text_dir: TextDir) -> None:
    """
    Add main section that allows selecting a train entity to show
    its label, facts, and texts
    """

    # Load valid entities
    valid_ent_to_lbl: Dict[int, str] = split_dir.valid_entities_tsv.load()

    with st.beta_expander('Valid Entities ({})'.format(len(valid_ent_to_lbl)), expanded=False):
        ent = prompt_entity('Entity', valid_ent_to_lbl)

        # Load all valid facts and show selected entity's known and unknown valid facts
        known_valid_facts: List[Fact] = split_dir.valid_facts_known_tsv.load()
        unknown_valid_facts: List[Fact] = split_dir.valid_facts_unknown_tsv.load()
        show_entity_facts('Known Facts', known_valid_facts, ent)
        show_entity_facts('Unknown Facts', unknown_valid_facts, ent)

        # Load all valid texts and show selected entity's texts
        ow_valid_texts: Dict[int, Set[str]] = text_dir.ow_valid_sents_txt.load()
        show_entity_texts('Texts', ow_valid_texts, ent)


def _add_main_section_test_entities(split_dir: SplitDir, text_dir: TextDir) -> None:
    """
    Add main section that allows selecting a test entity to show
    its label, facts, and texts
    """

    # Load test entities
    test_ent_to_lbl: Dict[int, str] = split_dir.test_entities_tsv.load()

    with st.beta_expander('Test Entities ({})'.format(len(test_ent_to_lbl)), expanded=False):
        ent = prompt_entity('Entity', test_ent_to_lbl)

        # Load all test facts and show selected entity's known and unknown test facts
        known_test_facts: List[Fact] = split_dir.test_facts_known_tsv.load()
        unknown_test_facts: List[Fact] = split_dir.test_facts_unknown_tsv.load()
        show_entity_facts('Known Facts', known_test_facts, ent)
        show_entity_facts('Unknown Facts', unknown_test_facts, ent)

        # Load all test texts and show selected entity's texts
        ow_test_texts: Dict[int, Set[str]] = text_dir.ow_test_sents_txt.load()
        show_entity_texts('Texts', ow_test_texts, ent)


def prompt_entity(subheader, ent_to_lbl: Dict[int, str]) -> int:
    st.subheader(subheader)

    min_id = min(ent_to_lbl.keys())
    max_id = max(ent_to_lbl.keys())

    cols = st.beta_columns([25, 75])
    ent = cols[0].number_input('ID', key='_train_ent_id', min_value=min_id, max_value=max_id, value=min_id)
    cols[1].text_input('Label', key='_train_ent_lbl', value=ent_to_lbl[ent])

    return ent


def show_entity_facts(subheader: str, facts: List[Fact], ent: int) -> None:
    st.subheader(subheader)

    ent_train_facts = [fact for fact in facts if fact.head == ent]

    for fact in ent_train_facts:
        st.write(fact)


def show_entity_texts(subheader: str, ent_to_texts: Dict[int, Set[str]], ent: int) -> None:
    st.subheader(subheader)

    ent_texts = ent_to_texts[ent]

    for i, text in enumerate(ent_texts):
        st.text_area('', key=f'_{ent}_text_{i}', value=text)
