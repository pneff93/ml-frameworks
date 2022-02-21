FROM python:3.8.0
COPY rest_api.py /.
COPY predict.py /.
COPY reviews.h5 /.
WORKDIR /.
RUN pip install --upgrade pip
RUN pip install flask
RUN pip install keras
RUN pip install tensorflow
RUN pip install numpy
EXPOSE 8088
ENTRYPOINT ["python3"]
CMD ["rest_api.py"]