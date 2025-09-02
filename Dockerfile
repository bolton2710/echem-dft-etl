#Airflow
FROM apache/airflow:3.0.6
ADD requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt
#Compile jdftx
USER root
WORKDIR /opt
RUN apt-get -y update && apt-get -y --no-install-recommends install g++ cmake libgsl0-dev libopenmpi-dev openmpi-bin libfftw3-dev libatlas-base-dev liblapack-dev wget unzip ca-certificates make && \
	wget https://github.com/shankar1729/jdftx/archive/refs/heads/master.zip && unzip master.zip && rm master.zip && \
	cd jdftx-master && mkdir build && cd build && \
	cmake ../jdftx && make install -j4 && \
	apt-get -y purge g++ cmake wget unzip ca-certificates make
