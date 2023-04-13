FROM python:3.10.3-slim-bullseye

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

# Copy your project files
COPY . /app

# Install your packages
RUN pip3 install --no-cache-dir \
    flask \
    numpy \
    pillow \
    face_recognition

# Set the working directory
WORKDIR /app

# Install face_recognition
RUN cd /app && \
    pip3 install -r requirements.txt

# Expose port 80 if your application uses it
EXPOSE 5000


# Add the command to start your app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--log-level", "debug", "app:app"]
