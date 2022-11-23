**Homework №2**
==============================

## *Launch instructions*

### ***Build docker image***
Activate your env and then run in `online_inference/`:
~~~
docker build -t matthewiskornev/online_inference:v2 .
~~~
You can check all docker images by typing:
~~~
docker image ls
~~~


### ***Run docker container***
Run in `online_inference/`:
~~~
docker run -p8000:8000 -it --name service matthewiskornev/online_inference:v2
~~~
It will create docker container named service. <br />
You can check all running docker containers by typing (in new console window):
~~~
docker container ls
~~~
After this you can go to the __service__ at http://127.0.0.1:8000 and follow the instructions below.

### ***Using Docker Hub***
You don't have to build the image by yourself using Dockerfile. It's enough to download image from
__Docker__ __Hub__.
You cun do it by running in `online_inference/`:
~~~
docker pull matthewiskornev/online_inference:v2
~~~
After you can run container using command from last point.

### ***Making requests***
After we have started server we need to make sure that it is working correctly.
We can do it by running script `make_requests.py`. <br />
For this you have to switch to new window in console and run:
~~~
python microservice/make_requests.py
~~~

### ***Tests***
Run in `online_inference/`:
~~~
python -m pytest tests
~~~


## *Additional part*