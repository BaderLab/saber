FROM python:3.6
WORKDIR /app
COPY . /app
RUN pip install -e .
CMD ["python", "-m", "saber.cli.app"]
EXPOSE 5000
