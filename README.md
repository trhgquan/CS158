# CS158 - Xử lý ngôn ngữ tự nhiên ứng dụng
Tóm tắt lí thuyết & lab môn học Xử lý ngôn ngữ tự nhiên ứng dụng - CSC15008.

## Chủ đề
### Lí thuyết
- Topic 1: language-models aka mô hình ngôn ngữ
    - W02.
- Topic 2: text-classification aka phân lớp văn bản.
    - W03: Naive Bayes.
    - W04: Opinion Mining & Sentiment Analysis.
- Topic 3: Deep Learning
    - W05: first glance at deep learning.
- Topic 4: Machine Translation
    - W06 + 07: Machine Translation (chua viet notebook)

### Lab
- Lab-01: Tìm hiểu công cụ KenLM (code + report LaTeX). 
    - Tài nguyên sử dụng: [bản dịch tiếng Anh truyện ngắn Con Đầm Pích (*The Queen of Spades* - Aleksandr Sergeyevich Pushkin)](https://www.gutenberg.org/cache/epub/23058/pg23058.txt)

- Lab-02: Cài đặt thuật toán Naive Bayes (from scratch) và tiến hành phân lớp văn bản theo yêu cầu (code + report LaTeX)
    - Data stats
        | Data | Classes | Avg. sentence length | Dataset size | Vocab size | Num. words present in word2vec | Test size |
        |------|---------|----------------------|--------------|------------|--------------------------------|-----------|
        | TREC | 6       | 10                   | 5952         | 9592       | 9125                           | 500       |

    - Training & testing stats:
        | label | P (precision) | R (recall) | f1-score |
        |-------|---------------|------------|----------|
        | 0 | 0.83 | 0.75 | 0.79 |
        | 1 | 0.62 | 0.61 | 0.61 |
        | 2 | 0.88 | 0.78 | 0.82 |
        | 3 | 0.81 | 0.95 | 0.87 |
        | 4 | 0.68 | 0.94 | 0.79 |
        | 5 | 0.97 | 0.73 | 0.83 |
        | Accuracy |  |  | 0.55 |
        | Macro AVG | 0.80 | 0.79 | 0.79 |
        | Weighted AVG | 0.79 | 0.78 | 0.78 |

    - Train:
        ```
        python train.py --input=<input_file> --model=<output_model_name> [--stopwords=<stopwords_file>]
        ```

    - Test:
        ```
        python test.py --input=<input_file> --model=<input_model_name>
        ```

- Lab-03: Tìm hiểu công cụ [jupyter-text2code](https://github.com/deepklarity/jupyter-text2code) (notebook + report LaTeX)
    - Tài nguyên sử dụng: [COVID-19 data from kaggle](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)
- Lab-04: Tìm hiểu bài toán phân lớp văn bản phân tầng (Hierarchical Multi-label Text Classification), sử dụng mô hình [RandolphVI/Hierarchical-Multi-Label-Text-Classification](https://github.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification) (report LaTeX)

## LICENSE
This project is licensed under the terms of [The GNU GPL v3.0 License](LICENSE)

VNUHCM - University of Science, mùa Xuân 2022.
