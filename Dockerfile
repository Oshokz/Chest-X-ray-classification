FROM python:3.9.19
COPY . /flask_app
WORKDIR /flask_app
RUN pip install -r requirements.txt
CMD streamlit run flask_app.py