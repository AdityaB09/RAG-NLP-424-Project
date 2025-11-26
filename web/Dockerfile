FROM node:20-alpine AS base
WORKDIR /app

# Install deps
COPY package.json package-lock.json ./
RUN npm install --legacy-peer-deps

# Build
COPY . .
RUN npm run build

# Run
EXPOSE 3000
CMD ["npm", "run", "start"]
