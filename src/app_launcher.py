import streamlit as st

from app.browse_dataset import run_browse_dataset_page
from app.predict import render_predict_page


def main():
    """ Render common sidebar navigation and continue rendering selected page """

    st.sidebar.header('Navigation')

    navigate_to = st.sidebar.radio('', [
        'Browse Dataset',
        'Predict'
    ])

    if navigate_to == 'Browse Dataset':
        run_browse_dataset_page()
    elif navigate_to == 'Predict':
        render_predict_page()


if __name__ == '__main__':
    main()
