   # Используем официальный образ Python в качестве базового образа
   FROM python:3.12.8-slim

   # Устанавливаем зависимости для системы
   RUN apt-get update && \
       apt-get install -y --no-install-recommends \
       build-essential \
       && rm -rf /var/lib/apt/lists/*

   # Устанавливаем рабочую директорию внутри контейнера
   WORKDIR /app

   # Копируем директорию src/ в контейнер
   COPY src/ /app/

   # Устанавливаем Python зависимости
   RUN pip install --no-cache-dir -r requirements.txt

   # Указываем команду для запуска приложения
   CMD ["streamlit", "run", "app.py"]
   