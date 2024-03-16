FROM ollama/ollama

COPY conf.sh /
RUN sh /conf.sh

ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]