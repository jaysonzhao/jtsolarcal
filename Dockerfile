FROM registry.redhat.io/ubi7/python-36:latest
# 使用官方提供的 Python 开发镜像作为基础镜像
# 指定"python:2.7-slim"这个官方维护的基础镜像，从而免去安装 Python 等语言环境的操作。否则，这一段就得这么写了：
##FROM ubuntu:latest
##RUN apt-get update -yRUN apt-get install -y python-pip python-dev build-essential

WORKDIR /app
# 将工作目录切换为 /app
# 意思是在这一句之后，Dockerfile 后面的操作都以这一句指定的 /app 目录作为当前目录。 

ADD . /app
# 将当前目录下的所有内容复制到 /app 下 
# Dockerfile 里的原语并不都是指对容器内部的操作。比如 ADD，指的是把当前目录（即 Dockerfile 所在的目录）里的文件，复制到指定容器内的目录当中。

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# 使用 pip 命令安装这个应用所需要的依赖
 
EXPOSE 8080
# 允许外界访问容器的 80 端口
 
 
CMD ["python", "app.py"]
# 设置容器进程为：python app.py，即：这个 Python 应用的启动命令
# 这里app.py 的实际路径是 /app/app.py。CMD ["python", "app.py"] 等价于 "docker run python app.py"。
# 在使用 Dockerfile 时，可能还会看到一个叫作 ENTRYPOINT 的原语。它和 CMD 都是 Docker 容器进程启动所必需的参数，完整执行格式是："ENTRYPOINT CMD"。
# 但是，默认，Docker 会提供一个隐含的 ENTRYPOINT，即：/bin/sh -c。所以，在不指定 ENTRYPOINT 时，比如在这个例子里，实际上运行在容器里的完整进程是：/bin/sh -c "python app.py"，即 CMD 的内容就是 ENTRYPOINT 的参数。
# 基于以上原因，后面会统一称 Docker 容器的启动进程为 ENTRYPOINT，而不是 CMD。
