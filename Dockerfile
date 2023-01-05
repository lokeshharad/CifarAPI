FROM continuumio/anaconda3
COPY . /usr/app/
EXPOSE 7075
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD python Cifar10API.py