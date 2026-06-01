FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm install && npm run build 2>/dev/null || echo "build pendiente"
CMD ["sh", "-c", "echo 'frontend placeholder'"]