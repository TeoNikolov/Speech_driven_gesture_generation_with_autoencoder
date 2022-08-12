FROM python:3.7.12

RUN apt-get update
RUN apt-get -y install python3-pip libsndfile1 ffmpeg
  
WORKDIR /app
COPY ./ ./

RUN pip3 install tensorflow==1.15.2
RUN pip3 install -r ./requirements.txt
RUN pip3 install protobuf==3.19.0