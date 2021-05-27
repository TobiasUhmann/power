from pathlib import Path

import streamlit as st

from data.irt.text.text_dir import TextDir
from data.power.ruler_pkl import RulerPkl
from data.power.split.split_dir import SplitDir
from data.power.texter_pkl import TexterPkl


def add_sidebar_param_split_dir() -> SplitDir:
    """ Add text input for path to Power Split Dir to sidebar, check Power Split Dir, and return handle """

    split_dir_path = st.sidebar.text_input('Path to Power Split Directory',
                                           value='data/power/split/cde-50/')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    return split_dir


def add_sidebar_param_text_dir() -> TextDir:
    """ Add text input for path to IRT Text Dir to sidebar, check IRT Text Dir, and return handle """

    text_dir_path = st.sidebar.text_input('Path to IRT Text Directory',
                                          value='data/irt/text/cde-irt-5-marked/')

    text_dir = TextDir(Path(text_dir_path))
    text_dir.check()

    return text_dir


def add_sidebar_param_ruler_pkl() -> RulerPkl:
    """ Add text input for path to Power Ruler PKL to sidebar, check Ruler PKL, and return handle """

    ruler_pkl_path = st.sidebar.text_input('Path to Power Ruler PKL',
                                          value='data/power/ruler/cde-50-test.pkl')

    ruler_pkl = RulerPkl(Path(ruler_pkl_path))
    ruler_pkl.check()

    return ruler_pkl


def add_sidebar_param_texter_pkl() -> TexterPkl:
    """ Add text input for path to Power Texter PKL to sidebar, check Texter PKL, and return handle """

    texter_pkl_path = st.sidebar.text_input('Path to Power Texter PKL',
                                           value='data/power/texter/cde-irt-5-marked.pkl')

    texter_pkl = TexterPkl(Path(texter_pkl_path))
    texter_pkl.check()

    return texter_pkl
