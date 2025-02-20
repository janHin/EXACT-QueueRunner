# pull official base image
#FROM ubuntu:18.04 as builder
#FROM nvidia/cuda:11.2.1-devel-ubuntu20.04
#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
FROM nvcr.io/nvidia/pytorch:21.02-py3
#FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu22.04
#FROM nvidia/cuda:10.2.0-devel-ubuntu18.04
#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda:10.2-cudnn7-devel
# set work directory
WORKDIR /usr/src/exact

ENV DEBIAN_FRONTEND="noninteractive" TZ="SystemV"

#RUN apt-key del 7fa2af80
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
#RUN apt-get update

#RUN apt-get update && apt-get install -y python3-pip dos2unix python3-openslide python3-opencv  libvips libvips-dev netcat libpq-dev\
RUN apt-get update && apt-get install -y python3-opencv python3-openslide
#    && rm -rf /var/lib/apt/lists/*

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
#RUN pip3 install --upgrade pip
#RUN pip3 install numpy==1.19.5
#numpy==1.19.4
COPY ./requirements.txt /usr/src/exact/requirements.txt

RUN pip3 install -U -r requirements.txt

#RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
#RUN pip3 install --upgrade opencv-python
RUN pip3 uninstall -y opencv-python
RUN pip3 install -U opencv-python==4.5.5.64
#RUN pip3 install -U opencv-python-headless
RUN pip3 install -U openslide-python
# copy entrypoint.sh and convert to linux format 
COPY ./entrypoint.sh /usr/src/exact/entrypoint.sh

# copy settingsfile 
#COPY ./exact/settings.py.example /usr/src/exact/exact/settings.py
#RUN cat /usr/src/exact/exact/settings.py


#RUN dos2unix /usr/src/exact/entrypoint.sh

RUN chmod +x /usr/src/exact/entrypoint.sh
#RUN cat /usr/src/exact/entrypoint.sh

#RUN pip install --upgrade git+git://github.com/ubernostrum/django-registration.git#egg=django-registration
RUN pip list

# copy project
COPY . /usr/src/exact/

#RUN mv /usr/src/exact/exact/settings.py.example /usr/src/exact/exact/settings.py
#RUN cat /usr/src/exact/exact/settings.py

# run entrypoint.sh
ENTRYPOINT ["/usr/src/exact/entrypoint.sh"]
