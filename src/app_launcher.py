from collections import defaultdict

import streamlit as st

from app.browse_dataset import add_browse_dataset_page
from app.predict import add_predict_page


def main():
    """ Render common sidebar navigation and continue rendering selected page """

    st.sidebar.header('Navigation')

    navigate_to = st.sidebar.radio('', [
        'Browse Dataset',
        'Predict'
    ])

    if navigate_to == 'Browse Dataset':
        add_browse_dataset_page()
    elif navigate_to == 'Predict':
        add_predict_page()


def get_defaultdict():
    return defaultdict(list)


if __name__ == '__main__':
    main()
