# PTI Chatbot

## Overview

The PTI Chatbot is an intelligent conversational agent designed to provide comprehensive information about the Petroleum Training Institute (PTI) in Nigeria.
Methodologies that the system utilizes and offers currently:

1. **CAG (Contextual Adaptive Generation)**
2. **LightRAG (Lightweight Retrieval-Augmented Generation)**
3. **ILLAMA INDEX**

## Methodologies

### 1. CAG (Contextual Adaptive Generation)

CAG employs advanced natural language processing techniques to generate responses based on user input and historical conversation context. It utilizes a powerful language model to adapt its responses dynamically, ensuring that the information provided is relevant and accurate.

- **Key Features**:
  - Contextual understanding of user queries.
  - Ability to generate detailed responses based on internal knowledge and user history.
  - Integration with Google GenAI for enhanced language capabilities.

### 2. LightRAG (Lightweight Retrieval-Augmented Generation)

LightRAG combines retrieval-based methods with generative capabilities to provide accurate answers. It retrieves relevant documents from a knowledge base and uses them to inform its responses, ensuring that the information is both accurate and contextually appropriate.

- **Key Features**:
  - Efficient retrieval of relevant documents from a structured database.
  - Augmentation of generated responses with factual data.
  - Optimized for performance and scalability.

### 3. ILLAMA INDEX

ILLAMA INDEX is a sophisticated indexing system that organizes and retrieves information efficiently. It allows the chatbot to access a wide range of data quickly, ensuring that users receive timely and relevant answers to their queries.

- **Key Features**:
  - Fast and efficient indexing of large datasets.
  - Seamless integration with the chatbot for real-time information retrieval.
  - Supports complex queries and provides structured responses.

## Installation

To set up the PTI Chatbot, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/jonathan-ikpen/pti-bot-v1.git
   cd pti-bot-v1
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory and add your API keys:
     ```
     GOOGLE_API_KEY=your_google_api_key
     FIRECRAWL_API_KEY=your_firecrawl_api_key
     LLMA_INDEX_API_KEY=your_llama-index-cloud_api_key
     ```

## Usage

To run the chatbot, execute the following command:
```bash
streamlit run main.py
```

Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of the libraries and frameworks used in this project.
- Thanks to the Petroleum Training Institute for the data (website - `pti.edu.ng`).
