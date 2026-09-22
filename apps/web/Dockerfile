FROM node:22.14-bookworm-slim AS dependencies

WORKDIR /app
COPY package.json package-lock.json ./
COPY apps/web/package.json ./apps/web/package.json
RUN npm ci

FROM node:22.14-bookworm-slim AS build

WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=dependencies /app/apps/web/node_modules ./apps/web/node_modules
COPY package.json package-lock.json ./
COPY apps/web ./apps/web

ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

RUN npm run build --workspace=@veritas-face/web

FROM node:22.14-bookworm-slim AS runtime

WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/package.json ./package.json
COPY --from=build --chown=node:node /app/apps/web ./apps/web

USER node

EXPOSE 3000

CMD ["npm", "run", "start", "--workspace=@veritas-face/web", "--", "--hostname", "0.0.0.0", "--port", "3000"]
