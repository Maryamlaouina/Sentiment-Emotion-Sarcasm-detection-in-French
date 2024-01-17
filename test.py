import streamlit as st
import h5py

model_path = "models/Emotions.h5"

# Open the HDF5 file in read-only mode
with h5py.File(model_path, "r") as file:
    # Get a list of keys (names of objects) in the HDF5 file
    keys = list(file.keys())

    # Display the keys in the Streamlit app
    st.write("Keys in the HDF5 file:", keys)
