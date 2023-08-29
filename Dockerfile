FROM python:3.9

RUN mkdir dash-a
WORKDIR /dash-a
ADD data/ /dash-a/data/
ADD code/ /dash-a/code/

ADD models/ /dash-a/models/
COPY requirements.txt /dash-a/requirements.txt

RUN pip3 install --no-cache-dir -r /dash-a/requirements.txt
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8501
WORKDIR "/dash-a/code/"

CMD ["/bin/bash", "stream.sh"]
