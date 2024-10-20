This README file is created by this app itself.
---

# Chat with Your Code

## Overview

"Chat with Your Code" is an application designed to facilitate interaction with code repositories. It allows users to parse documents by file types, create vector store indices, and manage chat sessions effectively.

## Features

- **Document Parsing**: Supports parsing of various file types, including Markdown and code files, with specific handling for each type.
- **Index Creation**: Utilizes Qdrant for creating vector store indices from parsed documents.
- **Session Management**: Maintains session states for unique user interactions.

## Installation

To set up the application, ensure you have the necessary dependencies installed. You can typically do this using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Start the Application**: Run the main application script.
   ```bash
   streamlit run app.py
   ```

2. **Parse Documents**: The application can parse documents based on file extensions and language settings. It supports recursive directory reading.

3. **Create Index**: Use the `create_index` function to generate a vector store index from parsed nodes.

4. **Manage Sessions**: The application maintains session states, allowing for persistent interactions.

## Functions

- `parse_docs_by_file_types(ext, language, input_dir_path)`: Parses documents based on file type and language.
- `create_index(nodes, client)`: Creates a Qdrant vector store index from document nodes.
- `reset_chat()`: Resets the chat session, clearing messages and context.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
