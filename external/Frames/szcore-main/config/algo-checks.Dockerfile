# Dockerfile
# Provides dependencies to verify algorithm definition files 
# and validate them against schema

FROM alpine:3.20

# install dependencies
RUN apk add --no-cache \
    curl \
    cargo \
    yq

ENV PATH="${PATH}:/root/.cargo/bin"

RUN cargo install jsonschema-cli
