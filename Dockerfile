# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container to /app
# This is where the application files will be stored in the container
WORKDIR /app

# This will bring your Flask application files, requirements.txt, and other project files into the container
COPY . /app

# Install the dependencies specified in requirements.txt
# This installs all the necessary Python packages for your Flask app to run
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 to the outside world
# Flask runs by default on port 5000, so this makes it accessible from outside the container
EXPOSE 5000

# Set the command to run your Flask application
CMD ["python", "flask_app.py"]
