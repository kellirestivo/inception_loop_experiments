FROM sinzlab/pytorch:v3.8-torch1.4.0-cuda10.1-dj0.12.4
    
WORKDIR /src

ADD ./monkey_mei /src/monkey_mei
RUN pip3 install -e monkey_mei

ADD ./neuralpredictors /src/neuralpredictors
RUN pip3 install -e neuralpredictors
RUN pip install --upgrade pip
RUN pip3 install six>=1.10

ADD ./nnfabrik /src/nnfabrik
RUN pip3 install -e nnfabrik

ADD ./mei /src/mei
RUN pip3 install -e mei

ADD ./nnvision /src/nnvision
RUN pip3 install -e nnvision

ADD ./CORnet /src/CORnet
RUN pip3 install CORnet

ADD ./ptrnets /src/ptrnets
RUN pip3 install -e ptrnets

RUN pip3 install gitpython

RUN pip3 install jupyterlab

RUN pip install numpy==1.19.5 scikit-image==0.18.3 h5py==2.10.0 pandas==1.3.5 datajoint scipy==1.7.3

RUN pip3 install einops

RUN pip3 install --upgrade --force-reinstall torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html

WORKDIR /notebooks