# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod ./
# COPY go.sum ./ # Uncomment if go.sum is added
RUN go mod download

COPY . .

# Build a statically linked binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -installsuffix cgo -ldflags="-w -s" -o nearby ./cmd/server

# Final stage
FROM scratch

# Copy the binary from the builder stage
COPY --from=builder /app/nearby /nearby

# Expose the default port
EXPOSE 6379

# Entry point
ENTRYPOINT ["/nearby"]
