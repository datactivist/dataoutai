FROM python:3.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /dataoutai
WORKDIR /dataoutai
COPY . /dataoutai/
RUN pip install -r requirements.txt
WORKDIR /dataoutai/visualization
EXPOSE 5006
CMD ["sh", "entrypoint.sh"]