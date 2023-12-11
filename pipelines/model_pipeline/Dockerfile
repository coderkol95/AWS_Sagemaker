FROM ubuntu
RUN apt-get update
RUN apt-get install -y python3 python3-pip
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT python3 code/train.py