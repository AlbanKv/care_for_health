# write some code to build your image
FROM python:3.8.12-buster

COPY api /api
COPY care_for_health /care_for_health
COPY data /data
# COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
# COPY /home/albankv/code/AlbanKv/gcp/possible-tape-337815-de50173e79d8.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast_care_for_health:app --host 0.0.0.0 --port $PORT
