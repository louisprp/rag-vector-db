# RAG Assited Abstract Question Answering

This repository contains a program built with Pinecone vector storage, Groq, and Streamlit. The program is designed to parse documents, split them into chunks, convert these chunks into vectors, store them in a Pinecone vector database, and allow for efficient querying and retrieval of similar matches and their corresponding metadata. Additionally, a Streamlit frontend has been created to facilitate user interaction with the database.

## Showcase

<video width="full" controls>
  <source src="showcase/screenrecording.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Features

- **Document Parsing and Chunking:** A Jupyter Notebook is used to parse documents and split them into manageable chunks.
- **Vector Conversion and Storage:** These chunks are then converted into vectors using the model `all-mpnet-base-v2` and stored in a Pinecone vector database.
- **Efficient Querying:** The database can be queried using a search query to retrieve similar matches along with their corresponding metadata.
- **Streamlit Frontend:** A user-friendly frontend built with Streamlit allows users to query the database and view the sources of the retrieved information.
- **LLM Integration:** The context from the retrieved data can be passed to an LLM (LLaMA 3-8b-8192) to provide an answer to the user's question using the available context from the vector storage.
- **Source Filtering:** Optionally, the information to be retrieved can be filtered by its source (e.g., book or article title).

## How It Works

1. **Document Parsing:**
   - Documents are parsed and split into smaller chunks using a Jupyter Notebook.

2. **Vector Conversion:**
   - The chunks are converted into vectors using the `all-mpnet-base-v2` model.

3. **Vector Storage:**
   - The vectors are stored in a Pinecone vector database.

4. **Querying the Database:**
   - Users can query the database to retrieve similar matches and their metadata.

5. **Streamlit Frontend:**
   - The Streamlit frontend allows users to interact with the database and view the results.

6. **LLM Integration:**
   - The context from the retrieved data is passed to an LLM (LLaMA 3-8b-8192) to provide answers based on the available context.

7. **Source Filtering:**
   - Users can filter the information to be retrieved by its source (e.g., book or article title).

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/document-vector-search.git
cd document-vector-search
pip install -r requirements.txt
```

## Usage
1. **Update API Keys:**
   - Copy `.env.example` to `.env` and update your API keys

2. **Run the Jupyter Notebook:**
   - Place your documents in the `articles` folder
   - Parse and split your documents into chunks.
   - Convert the chunks into vectors and store them in the Pinecone database.

3. **Run the Streamlit Frontend:**
   - Start the Streamlit application to query the database and view the results.
  
  ```bash
  streamlit run app.py
  ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Pinecone](https://www.pinecone.io/) for the vector database.
- [Hugging Face](https://huggingface.co/) for the `all-mpnet-base-v2` model.
- [Streamlit](https://www.streamlit.io/) for the frontend framework.
- [Groq](https://www.groq.com/) for the computation acceleration.

---

Happy querying!
