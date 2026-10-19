FROM ganeshhealthaireg.azurecr.io/healthai-app:ui-fixed
COPY src/19_streamlit/01_healthai_virtual_hospital.py /app/src/19_streamlit/01_healthai_virtual_hospital.py
EXPOSE 8501
CMD ["streamlit", "run", "/app/src/19_streamlit/01_healthai_virtual_hospital.py", "--server.address=0.0.0.0", "--server.port=8501"]
