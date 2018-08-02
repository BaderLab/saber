FROM python:3
COPY . /usr/src/app
WORKDIR /usr/src/app/saber
RUN pip install --no-cache-dir -r ../requirements.txt
CMD ["python -m", "saber.app"]
EXPOSE 5000
