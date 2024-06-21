# https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-tensorflow-models-in-aws-lambda/
FROM public.ecr.aws/lambda/python:3.12


RUN dnf install xz tar -y
RUN curl https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz > ffmpeg-release-amd64-static.tar.xz
RUN curl https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz.md5 > ffmpeg-release-amd64-static.tar.xz.md5
RUN md5sum -c ffmpeg-release-amd64-static.tar.xz.md5
RUN tar xf ffmpeg-release-amd64-static.tar.xz
RUN mv ffmpeg*/ffmpeg /usr/local/bin/
RUN rm -r ffmpeg*
RUN dnf remove xz tar -y

COPY requirements.txt ./
# https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
RUN python3.12 -m pip install -r requirements.txt --no-cache-dir
COPY app.py ./
COPY templates ./templates

CMD ["app.handler"]

# https://devicetests.com/send-base64-encoded-images-curl
# echo '{"body": "'$(base64 ../train_audio_test/amecro/XC180081.mp3)'"}' > /tmp/a
# curl -XPOST  "http://localhost:9000/2015-03-31/functions/function/invocations" -d @/tmp/a

