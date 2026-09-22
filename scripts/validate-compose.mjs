import fs from "node:fs";
import yaml from "js-yaml";

const manifest = yaml.load(fs.readFileSync(new URL("../compose.yaml", import.meta.url), "utf8"));

if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
  throw new Error("compose.yaml must contain a document object");
}

const services = requiredObject(manifest.services, "services");
for (const name of ["web", "api", "inference", "postgres", "redis", "object-storage"]) {
  requiredObject(services[name], `services.${name}`);
}

const api = services.api;
if (api.build?.dockerfile !== "services/api/Dockerfile") {
  throw new Error("API service must use its container build");
}
if (api.environment?.VERITAS_FACE_ARTIFACT_DIRECTORY !== "/var/lib/veritas-face/artifacts") {
  throw new Error("API service must use the private temporary artifact directory");
}
if (!api.volumes?.includes("temporary-artifacts:/var/lib/veritas-face/artifacts")) {
  throw new Error("API service must mount the temporary artifact volume");
}
if (api.depends_on?.postgres?.condition !== "service_healthy" || api.depends_on?.redis?.condition !== "service_healthy") {
  throw new Error("API service must wait for Redis and Postgres health checks");
}

if (services.web.build?.dockerfile !== "apps/web/Dockerfile" || services.web.depends_on?.api?.condition !== "service_healthy") {
  throw new Error("web service must build the web app and wait for API health");
}
if (services.inference.build?.dockerfile !== "services/inference/Dockerfile") {
  throw new Error("inference service must use its container build");
}
if (services["object-storage"].tmpfs?.includes("/data") !== true) {
  throw new Error("object storage must keep its data temporary");
}

for (const name of ["inference", "postgres", "redis", "object-storage"]) {
  if ("ports" in services[name]) {
    throw new Error(`${name} must stay private to the Compose network`);
  }
}

if (!("temporary-artifacts" in requiredObject(manifest.volumes, "volumes"))) {
  throw new Error("temporary artifact volume is missing");
}

console.log("Compose topology valid: web, API/worker, inference, Redis, Postgres, and temporary object storage");

function requiredObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value;
}
