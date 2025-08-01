#FROM python:3.11
#WORKDIR /app
#COPY requirements.txt .
#RUN pip install --upgrade pip
##RUN pip install --no-cache-dir -r requirements.txt
#COPY . .
#EXPOSE 8000


#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
