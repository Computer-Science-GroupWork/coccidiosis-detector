# coccidiosis-detector

Everything packaged in docker with requirement.txt, so you can push it to any docker hosted cloud service. Enjoy :)

You can test locally by installing Docker and using the following command:

docker build -t coccidiosis-detector . && docker run --rm -it -p 8080:8080 coccidiosis-detector