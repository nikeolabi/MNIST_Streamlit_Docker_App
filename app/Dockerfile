FROM python:3.11

WORKDIR /app

# Copy files from the correct paths (relative to root directory)
COPY app/ /app/
COPY backend_db_api/ /backend_db_api/
COPY SavedModels/ /SavedModels/
COPY requirements.txt /app/requirements.txt

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1

ENV PYTHONPATH="/backend_db_api"

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "Docker_Streamlit_app.py"]