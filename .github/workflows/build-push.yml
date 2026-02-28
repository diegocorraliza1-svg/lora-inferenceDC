name: Build & Push Inference Image

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  IMAGE_NAME: diegocorraliza/lora-inferencedc

jobs:
  build-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3

      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.IMAGE_NAME }}:latest
            ${{ env.IMAGE_NAME }}:v1-${{ github.run_number }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
