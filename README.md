# ProfAgro RAG System

Этот проект представляет собой систему, основанную на подходе **Retrieval-Augmented Generation (RAG)**, которая использует библиотеку `LangChain` для обработки и генерации ответов на основе предварительно извлеченных документов и контекста. Система интегрирует различные методы поиска и генерации, включая использование векторных баз данных (FAISS) и модели для обработки естественного языка.

Он сделан для  быстрого нахождения нужной документации по оборудованию для сотрудников компании (генерация ответа занимает от 10 до 15 секунд).
Интерфейс реализован через streamlit.

## Описание
Система выполнена в архитектуре **Retrieval-Augmented Generation (RAG)**, что позволяет эффективно интегрировать процесс извлечения информации и генерации ответов на основе контекста. Проект использует несколько ключевых технологий, включая:

- **FAISS** для поиска ближайших векторных представлений документов.
- **Jina** для создания эмбеддингов и работы с векторными базами данных.
- **LangChain** для создания цепочек обработки запросов и данных.
- **GigaChat API** для генерации ответов на основе извлеченной информации и истории чатов.

## Установка

Чтобы установить и запустить проект, выполните следующие шаги:

# В случае когда нет возможности использовать Docker:

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/feelness/profagro-rag-system.git
   cd profagro-rag-system

2. Установка зависимостей: Установите необходимые зависимости:
    ```python
    pip install -r requirements.txt

3. (Важно!)
 Настройка конфигурации:
 Создайте файл .env в корне проекта и добавьте необходимые API ключи для используемых сервисов:
   ```python
   jina_ai_api_key=your_jina_api_key
   gigachat_api_key=your_gigachat_api_key


4. Запуск проекта:
   ```python
   streamlit run  src/app.py
только (через src/app.py)

# Используя Docker
В корне проекта приложен Docker файла для вашего удобства

Как собрать:
   ```python
     docker build -t profagro-rag-system .
     docker run -p 8501:8501 --env-file .env profagro-rag-system
   


  




