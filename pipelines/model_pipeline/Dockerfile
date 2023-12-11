FROM python:3.8

RUN pip install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml:${PATH}"

COPY . /opt/ml
WORKDIR /opt/ml
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3 model.py"]