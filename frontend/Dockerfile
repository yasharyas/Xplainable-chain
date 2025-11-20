# Node.js
FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install system build deps required for some native npm modules
# Provide python and build tools for node-gyp to work in Alpine
RUN apk add --no-cache python3 make g++ build-base \
	&& ln -sf /usr/bin/python3 /usr/bin/python \
	&& npm install

# Copy application
COPY . .

# Expose port
EXPOSE 3000

# Run application
CMD ["npm", "run", "dev"]
