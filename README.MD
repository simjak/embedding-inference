## Overview

This repository contains a Rust workspace with three main components: embedding_generator, inf_server, and models_hf. The primary function is to generate embeddings from text data using a BERT model and serve these embeddings through a web server for similarity queries.

### Components

* embedding_generator: A Rust application that reads text data from a CSV file, generates embeddings using a BERT model, and serializes these embeddings for later use.
* inf_server: A web server built with Axum that loads pre-generated embeddings and provides an endpoint to find similar texts based on a query.
* models_hf: A library that interfaces with Hugging Face models, specifically BERT, for generating embeddings and performing similarity scoring.

## Getting Started

### Installation

1. Clone the repository.
2. Navigate to the root of the workspace.
3. Build the workspace using Cargo:

```cargo build --release```

### Usage

#### Generating Embeddings

1. Navigate to embedding_generator.
2. Run the application with a path to your CSV file:

``` cargo run --release -- <path_to_your_csv> ```

#### Starting the Inference Server

1. Navigate to inf_server.
2. Run the server:

``` cargo run --release ```

3. The server will start on 0.0.0.0:3000. Use the /similar endpoint to query for similar texts.

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.