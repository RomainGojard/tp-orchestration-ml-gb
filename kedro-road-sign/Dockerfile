ARG BASE_IMAGE=python:3.13.2
FROM $BASE_IMAGE as runtime-environment

RUN apt-get update && \
    apt-get install -y libgl1 tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# update pip and install uv
RUN python -m pip install -U "pip>=21.2"
RUN pip install uv
RUN tesseract --version

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_road_sign -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_road_sign

WORKDIR /home/kedro_road_sign
USER kedro_road_sign

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

CMD ["python", "app.py"]

EXPOSE 8000