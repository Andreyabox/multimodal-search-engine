FROM nginx:alpine

WORKDIR /usr/share/nginx/html
RUN rm -rf ./*

COPY ui/index.html .
COPY ui/style.css .
COPY ui/app.js .

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
