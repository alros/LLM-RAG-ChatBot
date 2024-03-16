#!/bin/sh

/bin/ollama serve &
sleep 3
/bin/ollama pull mistral