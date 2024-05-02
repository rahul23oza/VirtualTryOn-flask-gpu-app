FROM python:alpine3.19
WORKDIR /app
COPY . /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENTRYPOINT [ "python" ]

# For Test Usage
CMD [ "/app/main.py" ]
