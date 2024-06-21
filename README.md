# birdclef

## Algorithm overview for the [Cornell Birdcall Identification](https://www.kaggle.com/competitions/birdsong-recognition) data:
* data split of audio recordings:
    * train: 19151 recordings
    * test: 2222 recordings
* outputs for 264 bird species
* pretrained [Inception](https://openaccess.thecvf.com/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) neural network backbone
* recording-level metric: identify the _primary_ bird in a recording
* loss based on [Deep CNN framework for audio event recognition using weakly labeled web data](https://deepai.org/publication/deep-cnn-framework-for-audio-event-recognition-using-weakly-labeled-web-data)
* training augmentation: remove a random amount of audio from the beginning of each recording
* hyperparameter tuning:
    *  learning rate (of Adam optimizer)
    *  drop rate (of input spectrogram features)
* recording-level accuracy: 73.7%

## Results:

#### Setup training:
* Note: the training was done using an [Amazon EC2 g5 instance](https://aws.amazon.com/ec2/instance-types/g5/)
  with an [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html):
  ```amazon/Deep Learning OSS Nvidia Driver AMI GPU TensorFlow 2.13 (Ubuntu 20.04) 20231212```
* Install package
```
sudo apt update
sudo apt install -y ffmpeg
```
* Clone this repository:
```
git clone https://github.com/toddstep/birdclef.git
```
* Install python packages (instead of those provided by the AMI). The second `pip install` is for packages used in training but not in inference:
```
python3 -m venv ~/.python/env/birdclef
source ~/.python/env/birdclef/bin/activate
pip install -r birdclef/requirements.txt
pip install -r birdclef/requirements_trainingOnly.txt
```
* Download Kaggle token from [Settings](https://www.kaggle.com/settings)->Account->API to `~/.kaggle/kaggle.json`
* Agree to [Kaggle birdsong data rules](https://www.kaggle.com/competitions/birdsong-recognition/data).
* Download and extract Kaggle birdsong data:
```
kaggle competitions download -c birdsong-recognition
unzip -q -d ./birdsong-recognition birdsong-recognition.zip
```
* Convert MP3 to OGG:
```
mp3_files=`find birdsong-recognition -name '*.mp3'`
for mp3 in ${mp3_files}; do
    base=`echo $mp3 | cut -f1 -d.`
    ffmpeg -i ${mp3} -ar 22050 -codec:a libvorbis -qscale:a 4 -ac 1 ${base}.ogg < /dev/null
    echo ${?} Exit ${mp3}
done
```


#### Train model:
* Run [Birdclef Training Notebook](birdclef-modeling.ipynb)
```
cd birdclef
export BASE_DIR=../birdsong-recognition
jupyter-lab
```
    * tuned hyperparameters:
        * learning rate 0.00021564782468849842
        * drop rate 0.05
    * 40 training epochs

## Prepare web demo:
#### Export model for inference:
```
python export_model.py ../birdclef-checkpoints/spectrogram_model_TUNE
```

#### Analyze model:
* Generate recording-level scores using [Test Scores Generation](test_scores.ipynb).
* Estimate optimal threshold level for a 1% false-positive rate using [Test Scores Analysis](analyze.ipynb).

#### Deploy on AWS Serverless Application Model (SAM):
* Install Docker:
    * On Amazon Linux 2023, follow [Creating a container image for use on Amazon ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/create-container-image.html).
    * On Ubuntu, follow [Installing Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
* Follow the [AWS SAM prerequisites](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/prerequisites.html).
* Install [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
* Set up a domain and hosted zone in [Route 53](https://us-east-1.console.aws.amazon.com/route53/v2/home) for the demo website.
* Create a certificate in [AWS Certificate Manager(ACM)](https://us-east-1.console.aws.amazon.com/acm/home) for the domain.
* Build stack and deploy (see Steps 2 and 3 of [Tutorial: Deploying a Hello World application](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started-hello-world.html)):
    * Answer `y` when asked:
      ```FlaskFunction Function Url has no authentication. Is this okay? ```
```
sam build -u
sam deploy --guided
```
* The deployed CloudFormation output `BirdFrontUrl` contains the URL for the demo.
