FROM python:3.8.0
COPY rest_api.py /.
COPY predict_value.py /.
COPY model.json /.
WORKDIR /.
RUN pip install --upgrade pip
RUN pip install flask
RUN pip install pandas
RUN pip install xgboost
EXPOSE 8088
ENTRYPOINT ["python3"]
CMD ["rest_api.py"]