import { UploadForm } from "./upload-form";

const evidenceSteps = [
  "Face quality and primary-face selection",
  "C2PA / Content Credentials and metadata inspection",
  "Fine-tuned ONNX model and independent baseline",
  "Calibrated verdict with explicit uncertainty",
];

export default function HomePage() {
  return (
    <main>
      <section className="upload-panel" aria-label="Portrait upload">
        <UploadForm />
      </section>
      <section className="evidence" aria-label="Analysis evidence">
        <h2>Every verdict will show its work</h2>
        <ol>
          {evidenceSteps.map((step, index) => (
            <li key={step}>
              <span>0{index + 1}</span>
              {step}
            </li>
          ))}
        </ol>
      </section>
      <aside>
        V1 assesses fully synthetic portraits, not face swaps, identity verification, or AI-edited photos.
      </aside>
    </main>
  );
}
