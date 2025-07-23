# Use the official Qdrant image as a starting point
FROM qdrant/qdrant:latest

# Switch to the root user to install packages
USER root

# Install curl and then clean up to keep the image small
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*