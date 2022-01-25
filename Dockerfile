FROM python:3.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /dataoutai
WORKDIR /dataoutai
COPY . /dataoutai/
RUN pip install --no-cache-dir bokeh~=2.4.2 networkx~=2.6.3 pickleshare==0.7.5
WORKDIR /dataoutai/visualization
EXPOSE 5006
CMD ["sh", "entrypoint.sh"]