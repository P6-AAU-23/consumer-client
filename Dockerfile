FROM python:3

EXPOSE 420/udp
ENV port 420
ENV ip "localhost"

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./main.py" ]
