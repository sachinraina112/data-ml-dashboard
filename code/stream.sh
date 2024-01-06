#!/bin/sh
echo "Running Streamlit server"
#/bin/bash
echo "Serving Inference"
streamlit run streamlit.py --server.address 0.0.0.0 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false --client.showErrorDetails false --client.toolbarMode minimal
