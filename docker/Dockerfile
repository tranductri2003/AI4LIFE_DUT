FROM jupyter/minimal-notebook

ENV GRANT_SUDO=yes
USER root
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl -y

USER jovyan
ADD ./notebooks/requirements.txt "/home/jovyan"
RUN pip install -r requirements.txt

COPY --chown=jovyan "." "/home/jovyan"
WORKDIR "/home/jovyan/notebooks"
RUN mkdir output

CMD ["jupyter", "lab", "--ip=0.0.0.0"]
