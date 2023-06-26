FROM python:3.10.3-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y --fix-missing \
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
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install dlib
RUN mkdir -p ~/dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git ~/dlib/ && \
    cd  ~/dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

# Set the working directory
WORKDIR /app

# Copy your project files
COPY . /app

# Install your Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port 5000 if your application uses it
EXPOSE 5000

# Add the command to start your app using Flask development server
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
