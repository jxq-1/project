# 使用官方 nginx 镜像作为基础镜像
FROM nginx:alpine

# 将本地的网页文件复制到容器内的 nginx 网站根目录
COPY . /usr/share/nginx/html

# 暴露端口 80，以便可以访问网页
EXPOSE 80
