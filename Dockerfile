# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY go.mod ./
RUN go mod download

# Copy source code
COPY cmd/ ./cmd/
COPY internal/ ./internal/
COPY pkg/ ./pkg/

# Build static binary
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o nearby ./cmd/server

# Final scratch image (< 15MB)
FROM scratch

COPY --from=builder /app/nearby /nearby

EXPOSE 6379

ENTRYPOINT ["/nearby"]
