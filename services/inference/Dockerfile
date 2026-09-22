FROM debian:bookworm-slim AS build

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
      build-essential \
      ca-certificates \
      cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY services/inference ./

RUN cmake -S . -B /build -DBUILD_TESTING=OFF -DVERITAS_FETCH_ONNXRUNTIME=ON \
    && cmake --build /build --parallel

FROM debian:bookworm-slim

RUN groupadd --gid 10001 veritas \
    && useradd --uid 10001 --gid veritas --create-home --shell /usr/sbin/nologin veritas

COPY --from=build /build/veritas-inference /usr/local/bin/veritas-inference
COPY --from=build /build/_deps/veritas_onnxruntime-src/lib/libonnxruntime.so.1.20.1 /usr/local/lib/libonnxruntime.so.1.20.1

RUN ldconfig

USER veritas

EXPOSE 8080

CMD ["veritas-inference", "--host", "0.0.0.0", "--port", "8080"]
