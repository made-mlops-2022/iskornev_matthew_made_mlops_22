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

### ***Size optimization of Docker image***
1. `.dockerignore` __file__. <br />
    Used to exclude useless files and dirs such us .pytest_cache, build, etc.
2. __Ordering__ __layers__ from the less frequently changed (to ensure the build cache is reusable)
   to the more frequently changed. <br />
    For example, last layers deal with paths to model and transformer, used port and 
    commands to run within the container. It is assumed that these items change most often
3. __Ignoring unnecessary packages__. <br />
    File `requirements.txt` contains only necessary packages. It contains 25 lines less compared 
    to the same file in the last homework.
4. __Minimizing the number of layers__. <br />
    I tried to union instructions like `COPY` and `RUN`. For example <br />`COPY requirements.txt ./service/requirements.txt`
    and `COPY setup.py ./service/` <br /> can be combined into <br />
    `COPY requirements.txt setup.py ./service/`.
    To union `RUN` instructions I used "&&".
5. __Using lighter basic images__. <br />
    For example, `python:3.9-slim` instead of `python:3.9`

Using all the steps, size of image decreased from __1.25__ GB to __503.42__ MB. <br />
However the main contribution is made by step 5.
    