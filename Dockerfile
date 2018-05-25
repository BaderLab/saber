FROM python:3
COPY . /usr/src/app
WORKDIR /usr/src/app/saber
RUN pip install --no-cache-dir -r ../requirements.txt
CMD ["python", "app.py" ]
EXPOSE 5000