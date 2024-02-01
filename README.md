# LLM-RAG-ChatBot

LLM-based chatbot using Retrieval-Augmented Generation (RAG) to give answers based on local files.

## Usage

Store txt documents in `./kb` and run the application with

```shell
ollama serve
python -m streamlit run app.py
```

Connect with a browser to [http://localhost:8501/](http://localhost:8501/).

## Performance

Tested with 107 files in markdown for a total of 596K, on Mac M1 with 32 GB of RAM.

The application is memory intensive and the answer may take 30+ seconds.

The following diagram shows the memory pressure during a query. The phases of document search and answer generation are clearly visible.

![memory pressure](docs/img/memory.png)