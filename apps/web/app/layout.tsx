import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "Veritas Face",
  description: "Evidence-first synthetic portrait analysis."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
