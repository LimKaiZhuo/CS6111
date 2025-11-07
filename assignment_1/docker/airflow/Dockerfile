FROM apache/airflow:2.9.1

# Install runtime dependencies for PySpark (Java) and utilities
USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

COPY requirements.txt ./

USER airflow

RUN pip install --no-cache-dir -r requirements.txt

VOLUME /app
