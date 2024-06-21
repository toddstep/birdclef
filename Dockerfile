# https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-tensorflow-models-in-aws-lambda/
FROM public.ecr.aws/lambda/python:3.10
# ADD https://github.com/toddstep/birdclef#main ./
# WORKDIR birdclef


COPY requirements.txt ./
# https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
RUN python3.10 -m pip install librosa pandas tensorflow-cpu==2.13.1 tensorflow-io==0.34.0 tensorflow-probability==0.21.0 --no-cache-dir
COPY *.py ./
COPY birdmodeling ./birdmodeling
COPY birdtraining ./birdtraining
COPY birddata ./birddata
COPY competition_classes.txt audio_stats.pickle score_threshold.pickle ./
COPY export_model ./export_model

# https://stackoverflow.com/questions/59290386/runtimeerror-at-cannot-cache-function-shear-dense-no-locator-available-fo
# https://github.com/numba/numba/issues/4032
RUN mkdir /tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

CMD ["top_birds.lambda_handler"]

# https://devicetests.com/send-base64-encoded-images-curl
# echo '{"body": "'$(base64 ../train_audio_test/amecro/XC180081.mp3)'"}' > /tmp/a
# curl -XPOST  "http://localhost:9000/2015-03-31/functions/function/invocations" -d @/tmp/a

